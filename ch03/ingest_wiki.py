"""
ingest_wiki.py
================
한국어 위키백과 데이터를 vectorstore에 적재하는 파이프라인.

Hugging Face `wikimedia/wikipedia` 데이터셋(이미 정제되어 있음)을
스트리밍으로 읽어 Clean Text -> Markdown(.md) -> Chunking -> Embedding
-> Vector DB 순서로 적재한다:
   load_dataset("wikimedia/wikipedia", "20231101.ko") -> Clean Text -> Markdown(.md)
   -> Chunking -> Embedding -> Vector DB

6.2절 (1)의 "일반 도메인 코퍼스(한국어 위키피디아)"에 대응하는 데이터를
만들기 위한 모듈로, 기존 ch04 PDF 코퍼스("pdf" 컬렉션)와는
분리된 별도 컬렉션에 적재한다.

실행 방법: ch04 디렉터리에서 실행하며(ch04/data 상대경로 때문), Ollama와
Qdrant(ch02/docker-compose.yml)가 떠 있어야 한다.
    python3 ingest_wiki.py --limit N
    python3 ingest_wiki.py --hf-config 20231101.ko --limit N

for i in {1..10}; do
    echo "===== Run $i : $(date) ====="
    python3 ingest_wiki.py --hf-config 20231101.ko --limit 100
done

위 반복문처럼 여러 번 실행하면, 매번 실행이 끝날 때 다음 읽을 위치(offset)를
STATE_FILE(.ingest_wiki_state.json, (config, 컬렉션)별로 따로 저장)에
기록해두고 다음 실행에서 그 지점부터 이어서 읽으므로, 10번 반복하면 서로
겹치지 않는 문서 1000개(10 x 100)가 만들어진다. 처음부터 다시 읽고 싶으면
--reset-offset을 추가한다.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterator, List, Optional

from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from qdrant_utils import get_vectorstore

# --------------------------------------------------
# 0. 설정
# --------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent / "data"
MARKDOWN_DIR = DATA_DIR / "wikipedia"  # Clean Text -> Markdown 단계 결과물이 저장되는 위치

COLLECTION_NAME = "korean_wikipedia"  # 기존 PDF 코퍼스("pdf")와 분리된 일반 도메인 컬렉션

MIN_ARTICLE_CHARS = 200  # 이보다 짧으면 넘겨주기(redirect)/스텁 문서로 간주해 제외

MAX_FILES_PER_DIR = 1000  # 한 폴더에 이 개수 이상 쌓이면 part_0002, part_0003 ... 로 넘어감
# ch04/static은 ch04/data를 가리키는 심볼릭 링크이므로(rag_run.py 참고),
# data/wikipedia와 ch04/static/wikipedia는 실제로는 같은 디렉터리다.
# MARKDOWN_DIR 하나만 관리하면 두 경로 모두에 자동으로 반영된다.


# --------------------------------------------------
# 1. Hugging Face `wikimedia/wikipedia` 데이터셋: 이미 정제된 텍스트
# --------------------------------------------------
# Wikimedia가 공식으로 미리 정제해둔 덤프라, 별도 정제 도구 없이 바로 각
# row의 title/text를 {"title", "text"} 형태로 변환해 내놓는다. 전체 행 수
# (20231101.ko 기준 647,897개, 약 6.8GB)가 매우 크므로 스트리밍으로 읽고
# limit으로 잘라낸다.
HF_DATASET_NAME = "wikimedia/wikipedia"
HF_DEFAULT_CONFIG = "20231101.ko"

# offset(어디까지 읽었는지)을 실행 간에 기억해두는 상태 파일. config+컬렉션별로
# 따로 관리해서, 나중에 같은 명령을 다시 실행하면 이어서 다음 구간부터 읽는다.
STATE_FILE = DATA_DIR / ".ingest_wiki_state.json"


def _state_key(config: str, collection_name: str) -> str:
    return f"{config}|{collection_name}"


def load_offset(config: str, collection_name: str) -> int:
    """마지막으로 저장된 offset(다음에 읽어야 할 위치)을 가져온다. 없으면 0."""
    if not STATE_FILE.exists():
        return 0
    try:
        state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return 0
    return state.get(_state_key(config, collection_name), 0)


def save_offset(config: str, collection_name: str, offset: int) -> None:
    state = {}
    if STATE_FILE.exists():
        try:
            state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            state = {}
    state[_state_key(config, collection_name)] = offset
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def iter_huggingface_articles(
    config: str = HF_DEFAULT_CONFIG, limit: Optional[int] = None, offset: int = 0
) -> Iterator[dict]:
    """wikimedia/wikipedia 데이터셋을 스트리밍으로 읽어 문서를 하나씩 꺼낸다.

    offset개만큼 앞부분을 건너뛰고, limit개(None이면 끝까지)를 가져온다.
    skip/take 모두 데이터셋 순서상 정확히 offset~offset+limit-1번째 행을
    가리키므로, 다음 호출 때 offset+limit부터 이어서 읽으면 중복 없이 계속
    다음 구간을 처리할 수 있다.
    """
    from datasets import load_dataset

    dataset = load_dataset(HF_DATASET_NAME, config, streaming=True, split="train")
    if offset:
        dataset = dataset.skip(offset)
    if limit is not None:
        dataset = dataset.take(limit)
    for row in dataset:
        yield {"title": row["title"], "text": row["text"]}


# --------------------------------------------------
# 2. Clean Text 추가 정제
# --------------------------------------------------
_MULTI_BLANK_RE = re.compile(r"\n{3,}")
_TRAILING_WS_RE = re.compile(r"[ \t]+\n")


def clean_text(text: str) -> str:
    """본문에 남는 잔여 공백(줄 끝 공백, 3줄 이상 빈 줄)을 정리한다."""
    text = _TRAILING_WS_RE.sub("\n", text)
    text = _MULTI_BLANK_RE.sub("\n\n", text)
    return text.strip()


# --------------------------------------------------
# 3. Markdown(.md) 저장
# --------------------------------------------------
_UNSAFE_FILENAME_RE = re.compile(r'[\\/:*?"<>|]')


def safe_filename(title: str) -> str:
    return _UNSAFE_FILENAME_RE.sub("_", title).strip() or "untitled"


def _next_shard_dir(markdown_dir: Path, shard_index: int) -> Path:
    """markdown_dir/part_0001, part_0002 ... 형태의 하위 폴더 (처음부터 1000개 단위로 분할)."""
    shard_dir = markdown_dir / f"part_{shard_index:04d}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    return shard_dir


def write_markdown_files(
    articles: Iterator[dict], markdown_dir: Path, limit: Optional[int] = None
) -> List[Path]:
    """추출된 문서를 "# 제목\\n\\n본문" 형태의 .md 파일로 저장한다.

    본문이 MIN_ARTICLE_CHARS보다 짧은 문서는 넘겨주기/스텁으로 보고 건너뛴다.
    처음부터 markdown_dir/part_0001에 저장하며, MAX_FILES_PER_DIR개가 차면
    part_0002, part_0003 ... 순서로 새 폴더를 만들어 이어서 저장한다
    (폴더 하나에 파일이 너무 많이 쌓이는 것을 방지).
    """
    markdown_dir.mkdir(parents=True, exist_ok=True)

    shard_index = 1
    current_dir = _next_shard_dir(markdown_dir, shard_index)
    current_count = len(list(current_dir.glob("*.md")))

    written: List[Path] = []
    for article in articles:
        if limit is not None and len(written) >= limit:
            break
        body = clean_text(article["text"])
        if len(body) < MIN_ARTICLE_CHARS:
            continue

        while current_count >= MAX_FILES_PER_DIR:
            shard_index += 1
            current_dir = _next_shard_dir(markdown_dir, shard_index)
            current_count = len(list(current_dir.glob("*.md")))

        title = article["title"]
        md_path = current_dir / f"{safe_filename(title)}.md"
        md_path.write_text(f"# {title}\n\n{body}\n", encoding="utf-8")
        written.append(md_path)
        current_count += 1

    return written


# --------------------------------------------------
# 4. Chunking
# --------------------------------------------------
def load_markdown_documents(md_paths: List[Path]) -> List[Document]:
    """저장된 .md 파일을 다시 읽어 metadata(source, title)가 붙은 Document로 만든다.

    source는 DATA_DIR.parent(=이 스크립트가 있는 디렉터리) 기준 상대경로로 저장한다.
    PDF 코퍼스의 source("data/pdf/...")와 같은 관례이며, 절대경로를 저장하면
    저장소 디렉터리 이름이 바뀔 때(예: 과거 ch05 -> ch04 개편) DB에 이미 적재된
    metadata가 존재하지 않는 경로를 가리키게 되어 rag_run.py의 출처 링크 생성이
    조용히 실패하는 문제가 있었다.
    """
    documents = []
    base = DATA_DIR.parent
    for path in md_paths:
        text = path.read_text(encoding="utf-8")
        title = text.splitlines()[0].lstrip("# ").strip()
        rel_source = str(path.resolve().relative_to(base))
        documents.append(
            Document(page_content=text, metadata={"source": rel_source, "title": title})
        )
    return documents


def chunk_documents(
    documents: List[Document], chunk_size: int = 800, chunk_overlap: int = 150
) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)


# --------------------------------------------------
# 5. Embedding + 6. Vector DB 적재
# --------------------------------------------------
def build_vectorstore(collection_name: str = COLLECTION_NAME) -> QdrantVectorStore:
    embedding = OllamaEmbeddings(model="kure-v1")
    return get_vectorstore(collection_name, embedding)


# --------------------------------------------------
# 파이프라인 전체 실행
# --------------------------------------------------
def ingest_from_huggingface(
    config: str = HF_DEFAULT_CONFIG,
    limit: Optional[int] = 100,
    collection_name: str = COLLECTION_NAME,
    reset_offset: bool = False,
) -> dict:
    """HF wikimedia/wikipedia -> Clean Text -> Markdown -> Chunking -> Embedding -> Vector DB

    limit 없이 전체(647,897개, 약 6.8GB)를 다 받으면 매우 오래 걸리므로
    기본값을 100건으로 제한해두었다. 실제 정식 코퍼스를 만들 때는 필요한
    만큼 limit을 늘려서 실행한다.

    같은 (config, collection_name) 조합으로 다시 실행하면 STATE_FILE에 저장된
    offset을 읽어 이어서 다음 구간부터 처리한다(중복 없이 매번 새 문서를
    가져오기 위함). reset_offset=True면 0부터 다시 시작한다.
    """
    offset = 0 if reset_offset else load_offset(config, collection_name)

    print(
        f"[1/3] Hugging Face '{HF_DATASET_NAME}' ({config})에서 "
        f"offset={offset}부터 최대 {limit}건 스트리밍 중..."
    )

    print("[2/3] Clean Text 정제 및 Markdown(.md) 저장 중...")
    md_paths = write_markdown_files(
        iter_huggingface_articles(config, limit=limit, offset=offset), MARKDOWN_DIR
    )
    print(f"       -> {len(md_paths)}개 문서를 {MARKDOWN_DIR}에 저장했습니다.")

    print("[3/3] Chunking -> Embedding -> Vector DB 적재 중...")
    documents = load_markdown_documents(md_paths)
    chunks = chunk_documents(documents)
    print(f"       -> {len(documents)}개 문서를 {len(chunks)}개 청크로 분할했습니다.")

    vectorstore = build_vectorstore(collection_name)
    if chunks:
        vectorstore.add_documents(chunks)
    print(f"       -> 컬렉션 '{collection_name}'에 {len(chunks)}개 벡터를 적재했습니다.")

    next_offset = offset + limit if limit is not None else None
    if next_offset is not None:
        save_offset(config, collection_name, next_offset)
        print(f"       -> 다음 실행을 위해 offset={next_offset}을 {STATE_FILE}에 저장했습니다.")

    return {
        "articles": len(md_paths),
        "chunks": len(chunks),
        "collection": collection_name,
        "offset": offset,
        "next_offset": next_offset,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hf-config", dest="hf_config", default=HF_DEFAULT_CONFIG,
        help=f"Hugging Face {HF_DATASET_NAME} 데이터셋 설정 (기본값: {HF_DEFAULT_CONFIG})",
    )
    parser.add_argument("--limit", type=int, default=100, help="적재할 최대 문서 수 (테스트용)")
    parser.add_argument("--collection", default=COLLECTION_NAME, help="Qdrant 컬렉션 이름")
    parser.add_argument(
        "--reset-offset", action="store_true",
        help="저장된 offset을 무시하고 처음(0)부터 다시 읽는다.",
    )
    args = parser.parse_args()

    stats = ingest_from_huggingface(
        config=args.hf_config,
        limit=args.limit,
        collection_name=args.collection,
        reset_offset=args.reset_offset,
    )

    print(f"\n=== 완료: {stats} ===")

    print("\n=== 동작 확인: 적재된 컬렉션에서 검색 테스트 ===")
    vs = build_vectorstore(args.collection)
    for doc in vs.similarity_search("인공지능이란 무엇인가?", k=2):
        print(f"- {doc.metadata.get('title')} ({doc.metadata.get('source')})")
        print(f"  {doc.page_content[:100]}...")
