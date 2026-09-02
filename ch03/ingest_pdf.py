"""
ingest_pdf.py
=============
`ch03/data/pdf/staging/`에 놓인 PDF를 청킹해 Qdrant 컬렉션
("pdf")에 적재하고, 성공한 문서는 `ch03/data/pdf/part_XXXX/`
보관 폴더로 옮기는 수작업(사용자가 필요할 때 직접 실행) 파이프라인.

동작 순서(각 PDF마다):
  1. vectorstore에 동일 파일명이 이미 적재돼 있는지 확인한다(중복 적재 방지).
     이미 있으면 청킹·임베딩 없이 보관 폴더로만 옮긴다.
  2. 없으면 이 문서가 최종적으로 위치할 보관 경로(part_XXXX/파일명)를 먼저
     정하고, staging의 실제 파일을 읽어 청킹하되 각 청크 metadata의
     source는 (아직 옮기기 전이라도) 이 최종 경로로 기록한다.
  3. vectorstore.add_documents()가 성공한 뒤에야 실제 파일을 staging에서
     그 최종 경로로 옮긴다.
  이 순서를 지키는 이유: 적재 전에 먼저 옮기거나, source를 staging
  경로로 기록해두면 "DB에 기록된 출처 경로"와 "파일의 실제 위치"가
  어긋나서 UI의 출처 링크가 깨지는 문제가 실제로 있었다(예전에 성경
  PDF가 이 문제로 "File not found"가 떴던 사례). 적재 실패 시 파일은
  staging에 그대로 남으므로 재실행하면 안전하게 이어서 처리된다.

보관 폴더 규칙: part_XXXX 폴더 하나에 PDF 1,000개가 차면 다음 번호
(part_XXXX+1)를 새로 만들어 그 다음부터 채운다. 이미 최상위
data/pdf/에 낱개로 있는 예전 파일(예: 최초 코퍼스 3종)은 이 개수
계산에 포함하지 않는다 — staging에서 이동되는 파일만 part_XXXX
안으로 들어간다.

실행 방법: ch04 디렉터리에서 실행하며, Ollama와 Qdrant(ch02/docker-compose.yml)가
떠 있어야 한다.
    python3 ingest_pdf.py                 # staging의 모든 PDF 처리
    python3 ingest_pdf.py --limit 3        # 앞 3건만(테스트용)
    python3 ingest_pdf.py --dry-run        # 무엇을 할지만 보여주고 실제 적재·이동은 안 함

처리 후에는 BM25 캐시가 새 문서를 반영하도록
`python3 rag_retriever.py --rebuild-bm25 pdf`를 별도로 실행해야 한다
(이 모듈은 캐시를 자동으로 갱신하지 않는다).
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path
from typing import Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from qdrant_utils import find_source_by_filename, get_vectorstore, update_source_by_filename

# --------------------------------------------------
# 0. 설정
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent  # ch04/
PDF_DIR = BASE_DIR / "data" / "pdf"
STAGING_DIR = PDF_DIR / "staging"
MAX_FILES_PER_PART = 1000

COLLECTION_NAME = "pdf"

_PART_DIR_RE = re.compile(r"^part_(\d{4,})$")


# --------------------------------------------------
# 1. 보관 폴더(part_XXXX) 관리
# --------------------------------------------------
def _existing_part_dirs() -> list[tuple[int, Path]]:
    parts = []
    for p in PDF_DIR.iterdir():
        if p.is_dir():
            m = _PART_DIR_RE.match(p.name)
            if m:
                parts.append((int(m.group(1)), p))
    return sorted(parts)


def _current_target_dir() -> Path:
    """PDF를 넣을 보관 폴더를 정한다. 가장 번호가 큰 part_XXXX가 아직
    MAX_FILES_PER_PART 미만이면 그 폴더를, 꽉 찼거나 아예 없으면 다음
    번호의 새 폴더를 만들어 반환한다."""
    parts = _existing_part_dirs()
    if not parts:
        target = PDF_DIR / "part_0001"
        target.mkdir(parents=True, exist_ok=True)
        return target

    last_num, last_dir = parts[-1]
    count = sum(1 for _ in last_dir.glob("*.pdf"))
    if count < MAX_FILES_PER_PART:
        return last_dir

    target = PDF_DIR / f"part_{last_num + 1:04d}"
    target.mkdir(parents=True, exist_ok=True)
    return target


# --------------------------------------------------
# 2. 중복 적재 확인
# --------------------------------------------------
def _find_existing_source(file_name: str) -> Optional[str]:
    """vectorstore(pdf)에 이 파일명을 source로 가진 청크가 이미
    있으면 그 청크에 기록된 source 경로 문자열을 그대로 반환하고, 없으면
    None을 반환한다. part_XXXX 하위든 최상위든 경로와 무관하게 파일명
    기준으로 매칭한다(실제 구현: qdrant_utils.find_source_by_filename)."""
    return find_source_by_filename(COLLECTION_NAME, file_name)


def _update_existing_source(file_name: str, new_source: str) -> int:
    """이미 적재된 문서를 새 part_XXXX 위치로 옮길 때, vectorstore에 남아있는
    모든 청크의 metadata.source를 새 경로로 갱신한다. 파일을 옮기기만 하고
    이 갱신을 빼먹으면 "DB에 기록된 출처 경로"와 "파일의 실제 위치"가 어긋나
    UI 출처 링크가 깨진다(과거 실제로 겪은 문제). 갱신된 청크 수를 반환한다
    (실제 구현: qdrant_utils.update_source_by_filename)."""
    return update_source_by_filename(COLLECTION_NAME, file_name, new_source)


# --------------------------------------------------
# 3. 청킹 (rag_archiver.ipynb가 my_documents를 구축한 방식과 동일:
#    반복 머리말/꼬리말 제거 -> 페이지 병합 -> 절/조항 번호를 최우선
#    분할 기준으로 하는 RecursiveCharacterTextSplitter, chunk_size=800,
#    chunk_overlap=150)
# --------------------------------------------------
def load_and_chunk_pdf(pdf_path: Path, source_override: str) -> list[Document]:
    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()
    if not pages:
        return []

    # 일부 PDF(예: arXiv에서 뽑은 LLMLingua 논문)는 손상되거나 임베딩된
    # 폰트/바이너리 아티팩트로 인해 추출된 텍스트에 NUL(0x00) 바이트가
    # 섞여 나온다. PostgreSQL의 text 컬럼은 NUL 바이트를 절대 저장할 수
    # 없어(psycopg.DataError) 이 상태로 add_documents()를 호출하면 배치
    # 전체가 죽으므로, 텍스트를 쓰기 가장 이른 시점에 미리 제거한다.
    for page in pages:
        if "\x00" in page.page_content:
            page.page_content = page.page_content.replace("\x00", "")

    total_pages = len(pages)
    line_counts = Counter()
    for page in pages:
        for line in {l.strip() for l in page.page_content.splitlines() if l.strip()}:
            line_counts[line] += 1
    boilerplate_lines = {line for line, cnt in line_counts.items() if cnt >= max(total_pages * 0.3, 2)}
    page_number_re = re.compile(r"^\d+$")

    def clean_page_text(text: str) -> str:
        kept = [
            stripped for line in text.splitlines()
            if (stripped := line.strip()) and stripped not in boilerplate_lines
            and not page_number_re.match(stripped)
        ]
        return "\n".join(kept)

    full_text_parts = []
    offset = 0
    page_index = []  # [(offset, metadata), ...]
    for page in pages:
        cleaned = clean_page_text(page.page_content)
        if not cleaned:
            continue
        page_index.append((offset, {"source": source_override, "page": page.metadata.get("page", 0)}))
        full_text_parts.append(cleaned)
        offset += len(cleaned) + 1
    full_text = "\n".join(full_text_parts)
    if not full_text.strip():
        return []

    def metadata_for_offset(off: int) -> dict:
        result = page_index[0][1]
        for start, meta in page_index:
            if start <= off:
                result = meta
            else:
                break
        return result

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=[r"\n(?=\d{1,3}[.\s])", "\n\n", "\n", ". ", " ", ""],
        is_separator_regex=True,
    )

    chunks = []
    cursor = 0
    for chunk_text in text_splitter.split_text(full_text):
        idx = full_text.find(chunk_text, max(0, cursor - 200))
        if idx == -1:
            idx = cursor
        meta = dict(metadata_for_offset(idx))
        chunks.append(Document(page_content=chunk_text, metadata=meta))
        cursor = idx + len(chunk_text)
    return chunks


# --------------------------------------------------
# 4. 메인 처리 루프
# --------------------------------------------------
def process_staging(limit: Optional[int] = None, dry_run: bool = False) -> None:
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    pdf_paths = sorted(STAGING_DIR.glob("*.pdf"))
    if not pdf_paths:
        print(f"staging 폴더({STAGING_DIR})에 처리할 PDF가 없습니다.")
        return
    if limit:
        pdf_paths = pdf_paths[:limit]

    embedding = OllamaEmbeddings(model="kure-v1")
    vectorstore = get_vectorstore(COLLECTION_NAME, embedding)

    total_chunks = 0
    for i, pdf_path in enumerate(pdf_paths, start=1):
        file_name = pdf_path.name
        print(f"[{i}/{len(pdf_paths)}] {file_name}")

        target_dir = _current_target_dir()
        # source는 rag_run.py/build_source_link()가 ch04(프로세스 cwd) 기준
        # 상대경로로 해석하므로, 반드시 BASE_DIR(ch04/) 기준으로 계산해야 한다
        # — repo 루트 기준으로 계산하면(예: "ch04/data/pdf/...") 앱이 경로를
        # 이중으로("ch04/ch04/...") 붙여 출처 링크가 깨진다(실제로 겪은 문제).
        target_rel_source = str((target_dir / file_name).relative_to(BASE_DIR))

        existing_source = _find_existing_source(file_name)
        if existing_source is not None:
            if existing_source == target_rel_source and pdf_path.parent == target_dir:
                print("    -> vectorstore에 이미 존재하고 이미 제자리(이동 불필요)")
                continue
            print(f"    -> vectorstore에 이미 존재, 청킹 없이 {target_rel_source}(으)로 이동 + "
                  f"기존 청크의 출처 정보 갱신")
            if dry_run:
                continue
            target_path = BASE_DIR / target_rel_source
            if target_path.exists() and target_path != pdf_path:
                print(f"    -> 경고: 대상 경로에 파일이 이미 있어 건너뜀: {target_rel_source}")
                continue
            updated = _update_existing_source(file_name, target_rel_source)
            pdf_path.rename(target_path)
            print(f"    -> 청크 {updated}건 출처 갱신, 이동 완료: {target_rel_source}")
            continue

        if dry_run:
            print(f"    -> [dry-run] 청킹·적재 후 {target_rel_source} 로 이동 예정")
            continue

        try:
            chunks = load_and_chunk_pdf(pdf_path, target_rel_source)
        except Exception as e:
            print(f"    -> 파싱 실패, staging에 남겨두고 건너뜀: {e}")
            continue

        if not chunks:
            print("    -> 추출된 텍스트 없음(스캔본 이미지 PDF일 수 있음), staging에 남겨둠")
            continue

        try:
            vectorstore.add_documents(chunks)
        except Exception as e:
            # 여기서 실패해도 파일은 아직 staging에 있으므로(위로 옮기기 전),
            # 이 문서 하나 때문에 나머지 staging 파일 처리까지 멈추지 않는다.
            print(f"    -> 적재 실패, staging에 남겨두고 건너뜀: {e}")
            continue
        total_chunks += len(chunks)
        print(f"    -> {len(chunks)}개 청크 적재 완료")

        # 적재가 끝난 뒤에만 실제로 옮긴다 — 이 순서를 지켜야 중간에 실패해도
        # staging에 파일이 남아 다음 실행에서 안전하게 재시도된다.
        pdf_path.rename(target_dir / file_name)
        print(f"    -> 이동 완료: {target_rel_source}")

    print(f"\n총 {total_chunks}개 청크를 '{COLLECTION_NAME}' 컬렉션에 새로 적재했습니다.")
    if total_chunks:
        print("BM25 캐시에는 아직 반영되지 않았습니다. 필요하면 다음을 실행하세요:")
        print("  python3 rag_retriever.py --rebuild-bm25 pdf")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="staging 앞 N건만 처리(테스트용)")
    parser.add_argument("--dry-run", action="store_true", help="실제 적재·이동 없이 계획만 출력")
    args = parser.parse_args()
    process_staging(limit=args.limit, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
