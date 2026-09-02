"""
ingest_allganize.py
====================
Hugging Face의 allganize/RAG-Evaluation-Dataset-KO(금융·공공·의료·법률·커머스
5개 도메인, 58건 문서 색인 "documents.csv")를 원문 PDF로 내려받아
`ch03/data/allganize_rag_ko/<domain>/<file_name>`에 저장하고, 청킹하여
별도의 Qdrant 컬렉션("allganize_rag_ko")에 적재하는 파이프라인.

이 데이터셋은 QA 정답 쌍이 아니라 "documents.csv"에 문서별 원본 URL만
제공하며, 그 URL 상당수는 PDF 파일이 아니라 정부·기관 게시판의 상세보기
페이지(HTML)다. 따라서 다운로드는 두 단계 best-effort 방식으로 시도한다:
  1) URL을 그대로 GET해서 응답이 PDF(매직 바이트 %PDF- 또는 Content-Type에
     pdf 포함)이면 그대로 저장한다.
  2) 아니면 응답을 HTML로 간주하고, href 속성 중 ".pdf"가 포함된 링크를
     정규식으로 찾아 그 URL을 다시 시도한다(다운로드 버튼이 게시판 상세
     페이지에 걸려 있는 흔한 패턴).
사이트별 첨부파일 구조(로그인/세션/자바스크립트 렌더링 등)가 제각각이라
일부는 이 두 단계로도 실패할 수 있으며, 실패 건은 화면 출력과
download_report.csv에 명시적으로 남긴다(조용히 누락시키지 않는다).

실행 방법: ch03 디렉터리에서 실행하며(상대경로 때문), Ollama와
Qdrant(ch02/docker-compose.yml)가 떠 있어야 한다.
    python3 ingest_allganize.py                 # 다운로드 + 임베딩 적재 모두 수행
    python3 ingest_allganize.py --limit 5        # 앞 5건만(테스트용)
    python3 ingest_allganize.py --skip-download   # 이미 받아둔 PDF만 재청킹·재적재
    python3 ingest_allganize.py --skip-ingest     # 다운로드만 수행, 벡터 적재는 생략
    python3 ingest_allganize.py --reset-state     # 임베딩 완료 기록을 무시하고 전부 재적재
    python3 ingest_allganize.py --skip-download --skip-domains finance  # finance 폴더(대형 사업보고서) 제외하고 재적재
    python3 ingest_allganize.py --skip-download --chunk-size 1500 --chunk-overlap 200  # 청크 크기 조절(기본 800/150은 다른 코퍼스와 통일된 값이므로 바꿀 때 유의)
    python3 ingest_allganize.py --skip-download --domains medical       # medical 폴더만 재적재(예: 프로브 수정 후 재검증)
    python3 ingest_allganize.py --skip-download --domains law,medical         # 여러 도메인
    python3 ingest_allganize.py --skip-download --domains law,medical --skip-domains law  # 조합

cd ~/dev/workspace/koarf/ch03
# 터미널을 닫아도 안 죽도록 nohup + 백그라운드로 실행, 로그는 파일로
nohup python3 ingest_allganize.py --skip-download > ingest.log 2>&1 &
nohup python3 ingest_allganize.py --domains finance --skip-download --chunk-size 1500 --chunk-overlap 200 --chunk-workers 2 > ingest.log 2>&1 &
# 진행 상황 확인
tail -f ingest.log   
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional

import requests
from huggingface_hub import hf_hub_download
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from qdrant_utils import get_vectorstore

# --------------------------------------------------
# 0. 설정
# --------------------------------------------------
REPO_ID = "allganize/RAG-Evaluation-Dataset-KO"

BASE_DIR = Path(__file__).resolve().parent  # ch03/
DATA_DIR = BASE_DIR / "data" / "allganize_rag_ko"
STATE_FILE = DATA_DIR / ".ingest_state.json"
REPORT_FILE = DATA_DIR / "download_report.csv"
PROBE_SKIP_REPORT_FILE = DATA_DIR / "probe_skip_report.csv"

COLLECTION_NAME = "allganize_rag_ko"

# 정부·기관 사이트 다수가 기본 User-Agent(python-requests/…)를 차단하므로
# 일반 브라우저처럼 보이는 UA를 사용한다.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}
TIMEOUT = 20

_HREF_RE = re.compile(r'href\s*=\s*["\']([^"\']+\.pdf[^"\']*)["\']', re.IGNORECASE)
# href="" 속성이 아니라 onclick="fileDown('...')"나 순수 텍스트로만 박혀 있는
# 절대경로 PDF 링크(예: 이미지 갤러리 대신 별도 다운로드 스크립트를 쓰는
# 게시판)까지 잡아내기 위한 보조 패턴.
_BARE_PDF_URL_RE = re.compile(r'(https?://[^\s"\'<>()]+\.pdf)', re.IGNORECASE)

# scourt.go.kr(법원 게시판) 계열은 <a href="javascript:download('내부파일명','표시파일명')">
# 형태로 첨부를 걸어두고, 실제 파일은 자바스크립트가 채우는 hidden form을
# file.scourt.go.kr/AttachDownload에 POST해야 받아진다(GET으로는 못 받음).
# 페이지 HTML에서 이 두 조각(다운로드 함수 인자, 폼 action 주소)을 그대로
# 파싱해 같은 POST를 코드로 재현한다.
_SCOURT_DOWNLOAD_RE = re.compile(r"javascript:download\('([^']+)','([^']+)'\)")
_FORM_ACTION_RE = re.compile(r'form\.action\s*=\s*"([^"]+)"')


# --------------------------------------------------
# 1. documents.csv 확보
# --------------------------------------------------
def load_document_index() -> list[dict]:
    """HF 저장소의 documents.csv(domain, file_name, pages, url)를 내려받아 읽는다."""
    csv_path = hf_hub_download(repo_id=REPO_ID, filename="documents.csv", repo_type="dataset")
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows


# --------------------------------------------------
# 2. PDF 다운로드 (best-effort 2단계)
# --------------------------------------------------
def _truncate_utf8(text: str, max_bytes: int) -> str:
    """UTF-8로 인코딩했을 때 max_bytes를 넘지 않도록, 멀티바이트 문자를
    끊지 않고 자른다(한글은 문자당 3바이트라 글자 수 기준으로 자르면
    파일시스템의 바이트 단위 파일명 길이 제한을 넘길 수 있다)."""
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode("utf-8", errors="ignore")


def sanitize_filename(name: str) -> str:
    """경로 구분자·제어문자를 제거하고, 대부분의 파일시스템이 허용하는
    255바이트 제한 안에 들어오도록 필요하면 줄인다(법률 도메인 문서명처럼
    100자를 넘는 한글 파일명이 있어 그대로 두면 OSError가 난다)."""
    name = name.replace("/", "_").replace("\\", "_")
    name = "".join(ch for ch in name if ch.isprintable()).strip() or "unnamed.pdf"

    stem, dot, ext = name.rpartition(".")
    if not dot:
        stem, ext = name, "pdf"
    # 확장자(+ 여유)를 뺀 나머지를 stem 예산으로 주고, stem이 넘치면 자른다.
    stem_budget = 255 - len(ext.encode("utf-8")) - 1  # 1 = '.'
    stem = _truncate_utf8(stem, stem_budget)
    return f"{stem}.{ext}"


def _looks_like_pdf(resp: requests.Response) -> bool:
    content_type = resp.headers.get("Content-Type", "").lower()
    return "pdf" in content_type or resp.content[:5] == b"%PDF-"


def _save(resp: requests.Response, dest_path: Path) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    dest_path.write_bytes(resp.content)


def _get_with_retry(url: str, session: requests.Session) -> requests.Response:
    """일부 정부·기관 사이트 특유의 일시적 오류에 대응해 최대 3회 시도한다.
      - SSLError: 중간 인증서 체인이 불완전한 사이트가 있어(공개 정부 문서
        다운로드 목적이므로) 검증을 끄고 한 번 더 시도한다.
      - Timeout/ConnectionError: 네트워크 혼잡으로 인한 일시적 실패일 수 있어
        짧은 대기 후 재시도한다.
    """
    last_exc: Exception | None = None
    for attempt in range(3):
        verify = attempt == 0  # 1차만 인증서 검증, 실패하면 이후 시도는 검증 없이
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                resp = session.get(url, headers=HEADERS, timeout=TIMEOUT,
                                    allow_redirects=True, verify=verify)
            resp.raise_for_status()
            return resp
        except requests.exceptions.SSLError as e:
            last_exc = e
            continue  # 다음 시도부터 verify=False
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            last_exc = e
            time.sleep(2)
            continue
        except requests.RequestException as e:
            raise  # 404 등 재시도해도 의미 없는 오류는 즉시 전파
    raise last_exc


def _find_pdf_links_in_html(html: str, base_url: str) -> list[str]:
    from urllib.parse import urljoin

    candidates = []
    seen = set()
    for pattern in (_HREF_RE, _BARE_PDF_URL_RE):
        for href in pattern.findall(html):
            absolute = urljoin(base_url, href)
            if absolute not in seen:
                seen.add(absolute)
                candidates.append(absolute)
    return candidates


def _try_scourt_download(html: str, session: requests.Session) -> Optional[requests.Response]:
    """scourt.go.kr 계열 게시판 전용: javascript:download() 호출 인자와 폼
    action 주소를 HTML에서 파싱해, 브라우저가 자바스크립트로 하는 POST를
    그대로 재현한다. 매칭되는 패턴이 없으면 None을 반환한다."""
    dl_match = _SCOURT_DOWNLOAD_RE.search(html)
    action_match = _FORM_ACTION_RE.search(html)
    if not dl_match or not action_match:
        return None
    internal_name, display_name = dl_match.group(1), dl_match.group(2)
    action_url = action_match.group(1)
    try:
        resp = session.post(
            action_url,
            data={"file": internal_name, "path": "003", "downFile": display_name},
            headers=HEADERS,
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
    except requests.RequestException:
        return None
    return resp


# 사이트별 전용 핸들러 목록. 각 함수는 (html, session) -> Response|None 시그니처를
# 가지며, 페이지 구조를 인식하면 실제 첨부 응답을, 아니면 None을 반환한다.
# 새 사이트를 지원하려면 이 목록에 함수만 추가하면 된다.
_SITE_HANDLERS = [_try_scourt_download]


def download_one(url: str, dest_path: Path, session: requests.Session) -> tuple[str, str]:
    """반환값: (status, reason). status는 downloaded / downloaded_via_html /
    downloaded_via_site_handler / failed."""
    try:
        resp = _get_with_retry(url, session)
    except requests.RequestException as e:
        return "failed", f"1차 요청 실패: {e}"

    if _looks_like_pdf(resp):
        _save(resp, dest_path)
        return "downloaded", "직접 다운로드"

    html = resp.text

    # 1) 사이트별 전용 핸들러(자바스크립트로만 동작하는 첨부 다운로드 재현)
    for handler in _SITE_HANDLERS:
        site_resp = handler(html, session)
        if site_resp is not None and _looks_like_pdf(site_resp):
            _save(site_resp, dest_path)
            return "downloaded_via_site_handler", f"{handler.__name__} 재현 성공"

    # 2) HTML 게시판 상세 페이지로 간주하고, 페이지 내 .pdf 링크를 찾아 재시도한다.
    for candidate in _find_pdf_links_in_html(html, resp.url):
        try:
            resp2 = _get_with_retry(candidate, session)
        except requests.RequestException:
            continue
        if _looks_like_pdf(resp2):
            _save(resp2, dest_path)
            return "downloaded_via_html", f"HTML 내 링크 발견: {candidate}"

    return "failed", "PDF 응답도, 페이지 내 .pdf 링크도 찾지 못함(수동 확인 필요)"


def download_all(rows: list[dict], limit: Optional[int] = None) -> list[dict]:
    """documents.csv의 각 행을 내려받는다. 이미 파일이 있으면 건너뛴다(재실행 안전).

    한 행에서 예기치 못한 예외(파일명 길이 제한 등 사이트별 변수)가 나더라도
    전체 배치가 죽지 않도록 행 단위로 감싸고, 보고서는 매 행마다 즉시 기록해
    도중에 중단돼도 그때까지의 결과가 남도록 한다.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    report = []

    target_rows = rows[:limit] if limit else rows
    with open(REPORT_FILE, "w", encoding="utf-8", newline="") as report_f:
        writer = csv.DictWriter(report_f, fieldnames=["domain", "file_name", "url", "status", "reason"])
        writer.writeheader()

        for i, row in enumerate(target_rows, start=1):
            domain = row["domain"]
            file_name = sanitize_filename(row["file_name"])
            url = row["url"]
            dest_path = DATA_DIR / domain / file_name

            try:
                if dest_path.exists():
                    print(f"[{i}/{len(target_rows)}] 이미 존재, 건너뜀: {domain}/{file_name}")
                    entry = {"domain": domain, "file_name": file_name, "url": url,
                              "status": "exists", "reason": "이미 다운로드됨"}
                else:
                    print(f"[{i}/{len(target_rows)}] 다운로드 시도: {domain}/{file_name}")
                    status, reason = download_one(url, dest_path, session)
                    print(f"    -> {status}: {reason}")
                    entry = {"domain": domain, "file_name": file_name, "url": url,
                              "status": status, "reason": reason}
            except OSError as e:
                print(f"    -> failed: 파일 시스템 오류: {e}")
                entry = {"domain": domain, "file_name": file_name, "url": url,
                          "status": "failed", "reason": f"파일 시스템 오류: {e}"}

            report.append(entry)
            writer.writerow(entry)
            report_f.flush()

    ok = sum(1 for r in report if r["status"] in ("downloaded", "downloaded_via_html", "exists"))
    failed = [r for r in report if r["status"] == "failed"]
    print(f"\n다운로드 결과: 성공/기보유 {ok}/{len(report)}건, 실패 {len(failed)}건")
    if failed:
        print("실패 목록(수동으로 원문 사이트에서 내려받아 해당 경로에 저장하면 다음 실행에서 자동으로 포함됩니다):")
        for r in failed:
            print(f"  - [{r['domain']}] {r['file_name']}  <- {r['url']}")
    print(f"상세 결과: {REPORT_FILE}")
    return report


# --------------------------------------------------
# 3. 청킹 (rag_archiver.ipynb의 PDF 코퍼스 구축 방식과 동일:
#    반복 머리말/꼬리말 제거 -> 페이지 병합 -> 절/조항 번호를 최우선
#    분할 기준으로 하는 RecursiveCharacterTextSplitter, chunk_size=800,
#    chunk_overlap=150)
# --------------------------------------------------
_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")


def _strip_surrogates(text: str) -> str:
    """PyPDFLoader(pypdf)가 손상된 CID 폰트 매핑이 있는 PDF에서 짝이 안 맞는
    유니코드 서로게이트(예: U+DA23)를 그대로 반환하는 경우가 있다. 파이썬
    문자열 자체는 이를 담을 수 있지만 유효한 UTF-8로 인코딩할 수 없어,
    Ollama 임베딩 호출(httpx의 JSON 인코딩)에서 UnicodeEncodeError로
    죽으므로 청킹 전에 제거한다."""
    return _SURROGATE_RE.sub("", text)


def probe_has_extractable_text(pdf_path: Path, sample_pages: int = 8, min_chars: int = 50) -> bool:
    """PDF 전체를 파싱하지 않고 앞 몇 쪽만 열어 텍스트 추출 가능 여부를 빠르게 확인한다.

    스캔 이미지 PDF(예: public 도메인의 100MB급 문서)는 어차피 전체를 파싱해도 빈
    텍스트로 판정되어 load_and_chunk_pdf가 빈 리스트를 반환하지만, 그 판정에
    도달하기까지 페이지마다 이미지 스트림을 디코딩하느라 문서당 수 분이 걸릴 수 있다.
    이 프로브로 미리 걸러 청킹 큐에 아예 올리지 않는다.

    sample_pages 기본값을 8로 둔 이유: 표지·목차 페이지가 이미지로만 구성돼 앞
    1~2쪽은 텍스트가 0자여도 3쪽째부터 본문이 시작되는 정상 문서가 실제로 다수
    존재함을 확인했다(예: 32쪽짜리 문서가 0쪽 2자·1쪽 25자였지만 2쪽부터 179자,
    이후 매 쪽 100자 이상). sample_pages=2였을 때 이런 문서들이 오탐(false
    positive)으로 걸러져 실제로는 유효한 청크가 나오는 문서를 통째로 누락시켰다.
    반면 실제 스캔 이미지 문서는 앞 12쪽까지 검사해도 예외 없이 0자였다.

    프로브 자체가 실패하면(암호화 등 텍스트 유무와 무관한 다른 원인일 수 있으므로)
    통과시켜 본 파싱 단계에서 판단하게 둔다(정상 문서를 오탐으로 건너뛰는 것을
    피하기 위함)."""
    try:
        reader = PdfReader(str(pdf_path))
        pages = reader.pages[:sample_pages]
        text = "".join(_strip_surrogates(page.extract_text() or "") for page in pages)
        return len(text.strip()) >= min_chars
    except Exception:
        return True


def load_and_chunk_pdf(
    pdf_path: Path, domain: str, chunk_size: int = 800, chunk_overlap: int = 150,
) -> list[Document]:
    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()
    if not pages:
        return []
    for page in pages:
        page.page_content = _strip_surrogates(page.page_content)

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
    # rag_run.py의 build_source_link()가 source를 ch03(BASE_DIR) 기준 상대경로로
    # 해석하므로 반드시 BASE_DIR 기준이어야 한다 — 저장소 루트 기준으로 계산하면
    # ("ch03/data/...") 앱이 경로를 이중으로("ch03/ch03/...") 붙여 출처 링크가
    # 깨진다(ingest_pdf.py에서 이미 겪었던 문제와 동일).
    rel_source = str(pdf_path.relative_to(BASE_DIR))
    for page in pages:
        cleaned = clean_page_text(page.page_content)
        if not cleaned:
            continue
        page_index.append((offset, {"source": rel_source, "domain": domain,
                                     "page": page.metadata.get("page", 0)}))
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
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
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
# 4. Qdrant 적재 (이미 적재한 파일은 상태 파일로 기록해 재실행 시 건너뜀)
# --------------------------------------------------
def load_state() -> set[str]:
    if STATE_FILE.exists():
        return set(json.loads(STATE_FILE.read_text(encoding="utf-8")))
    return set()


def save_state(ingested: set[str]) -> None:
    STATE_FILE.write_text(json.dumps(sorted(ingested), ensure_ascii=False, indent=2), encoding="utf-8")


DEFAULT_CHUNK_WORKERS = 2  # 4코어 환경에서 메인 프로세스(임베딩 요청/DB 삽입)용 2코어를 남겨둔다.
# rag_archiver.ipynb의 PDF 코퍼스 구축 방식과 동일한 값(3장 참조). 다른 코퍼스(PDF
# 25종, 위키백과)와 청킹 방식을 통일하기 위한 값이므로, 여기서만 바꾸면 코퍼스 간
# 일관성이 깨질 수 있다 — 바꾸기 전에 다른 코퍼스도 함께 바꿀지 확인할 것.
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 150


def ingest_to_vectorstore(
    reset_state: bool = False,
    chunk_workers: int = DEFAULT_CHUNK_WORKERS,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    domains: Optional[set[str]] = None,
    skip_domains: Optional[set[str]] = None,
    skip_text_probe: bool = False,
    probe_pages: int = 8,
    probe_min_chars: int = 50,
) -> None:
    """PDF를 벡터 DB에 적재한다.

    청킹(PyPDFLoader 파싱 + 텍스트 분할)은 CPU 작업이고 Ollama와 무관하므로
    ProcessPoolExecutor로 여러 문서를 동시에 청킹한다. 반면 임베딩+DB삽입
    (vectorstore.add_documents)은 실측 결과 하나의 배치 호출 안에서는
    Ollama가 청크를 계속 슬롯 0번 하나로만 순차 처리해(OLLAMA_NUM_PARALLEL을
    올려도 배치 내부가 갈라지지 않음), 메인 프로세스에서 순차로 수행한다.
    즉 "다음 문서 청킹"과 "현재 문서 임베딩"이 겹쳐 돌아가는 파이프라인
    구조로, 청킹에 걸리는 시간만 임베딩 대기 시간 뒤에 숨긴다(임베딩 자체의
    처리량은 그대로다).

    domains: 지정하면 finance/law/medical/public/commerce 중 이 이름들에 해당하는
    domain 폴더만 처리한다(그 외 폴더는 이번 실행에서 완전히 제외). 예: {"medical"}면
    medical 폴더만 적재한다.

    skip_domains: finance/law/medical/public/commerce 등 domain 폴더명 중
    이번 실행에서 건너뛸 이름들. 예: {"finance"}면 사업보고서(대형 문서라
    청크 수가 압도적으로 많음)만 제외하고 나머지를 처리한다. domains와 함께 주면
    domains로 추린 뒤 skip_domains로 다시 제외한다.

    chunk_size/chunk_overlap: RecursiveCharacterTextSplitter에 그대로 전달된다.
    기본값(800/150)은 PDF 25종·위키백과 등 다른 코퍼스와 동일하게 맞춘 값이므로,
    이 컬렉션만 다른 값으로 재적재하면 코퍼스 간 청킹 방식이 달라진다는 점에 유의.

    skip_text_probe가 False(기본)이면, 청킹 큐에 올리기 전 각 PDF의 앞
    probe_pages쪽만 훑어 텍스트 추출량이 probe_min_chars 미만이면 건너뛴다.
    스캔 이미지 PDF는 어차피 load_and_chunk_pdf가 빈 리스트를 반환해
    최종 결과는 같지만, 그 판정에 이르기까지 전체 페이지의 이미지 스트림을
    디코딩하느라 문서당 수 분이 걸릴 수 있어 미리 걸러낸다.
    """
    embedding = OllamaEmbeddings(model="kure-v1")
    vectorstore = get_vectorstore(COLLECTION_NAME, embedding)

    ingested = set() if reset_state else load_state()
    pdf_paths = sorted(DATA_DIR.glob("*/*.pdf"))
    if domains:
        pdf_paths = [p for p in pdf_paths if p.parent.name in domains]
    if skip_domains:
        pdf_paths = [p for p in pdf_paths if p.parent.name not in skip_domains]
    if not pdf_paths:
        print("적재할 PDF가 없습니다. 먼저 다운로드를 수행하세요.")
        return

    todo = [p for p in pdf_paths if str(p.relative_to(BASE_DIR.parent)) not in ingested]
    skipped = len(pdf_paths) - len(todo)
    if skipped:
        print(f"이미 적재된 {skipped}건은 건너뜁니다.")
    if not todo:
        print("적재할 새 PDF가 없습니다.")
        return

    if not skip_text_probe:
        probe_skipped = []
        kept = []
        for p in todo:
            if probe_has_extractable_text(p, sample_pages=probe_pages, min_chars=probe_min_chars):
                kept.append(p)
            else:
                rel_source = str(p.relative_to(BASE_DIR.parent))
                probe_skipped.append(rel_source)
                ingested.add(rel_source)
        if probe_skipped:
            save_state(ingested)
            PROBE_SKIP_REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(PROBE_SKIP_REPORT_FILE, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["path", "reason"])
                writer.writeheader()
                for s in probe_skipped:
                    writer.writerow({
                        "path": s,
                        "reason": f"앞 {probe_pages}쪽 추출 텍스트 {probe_min_chars}자 미만(스캔 이미지 추정)",
                    })
            print(f"텍스트 프로브 결과 {len(probe_skipped)}건은 텍스트가 거의 없어(스캔 이미지 추정) 건너뜁니다"
                  f"(목록: {PROBE_SKIP_REPORT_FILE}):")
            for s in probe_skipped:
                print(f"  - {s}")
        todo = kept
        if not todo:
            print("프로브 이후 적재할 새 PDF가 없습니다.")
            return

    total_chunks = 0
    with ProcessPoolExecutor(max_workers=chunk_workers) as executor:
        # 모든 문서의 청킹 작업을 한꺼번에 제출한다. ProcessPoolExecutor가
        # chunk_workers개까지만 동시에 실행하고 나머지는 내부 큐에서 기다리므로
        # 여기서 직접 슬라이딩 윈도우를 관리할 필요는 없다.
        jobs = [
            (p, executor.submit(load_and_chunk_pdf, p, p.parent.name, chunk_size, chunk_overlap))
            for p in todo
        ]

        for i, (pdf_path, future) in enumerate(jobs, start=1):
            rel_source = str(pdf_path.relative_to(BASE_DIR.parent))
            print(f"[{i}/{len(todo)}] 임베딩 적재 중: {rel_source}")
            try:
                chunks = future.result()
            except Exception as e:
                # 암호화된 PDF, 손상된 파일 등 개별 문서의 파싱 실패가 전체 배치를
                # 죽이지 않도록 한다 — 실패 문서는 건너뛰고 계속 진행한다.
                print(f"    -> 파싱 실패, 건너뜀: {e}")
                ingested.add(rel_source)
                save_state(ingested)
                continue
            if not chunks:
                print("    -> 추출된 텍스트 없음, 건너뜀(스캔본 이미지 PDF일 수 있음)")
                ingested.add(rel_source)
                save_state(ingested)
                continue

            vectorstore.add_documents(chunks)
            total_chunks += len(chunks)
            print(f"    -> {len(chunks)}개 청크 적재")
            ingested.add(rel_source)
            save_state(ingested)  # 매 문서마다 저장해 중간에 중단돼도 이어서 재개 가능

    print(f"\n총 {total_chunks}개 청크를 '{COLLECTION_NAME}' 컬렉션에 새로 적재했습니다.")


# --------------------------------------------------
# 5. 실행
# --------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="documents.csv 앞 N건만 처리(테스트용)")
    parser.add_argument("--skip-download", action="store_true", help="다운로드를 건너뛰고 기존 PDF만 적재")
    parser.add_argument("--skip-ingest", action="store_true", help="다운로드만 수행하고 벡터 적재는 생략")
    parser.add_argument("--reset-state", action="store_true", help="적재 완료 기록을 무시하고 전부 재적재")
    parser.add_argument(
        "--chunk-workers", type=int, default=DEFAULT_CHUNK_WORKERS,
        help=f"PDF 청킹을 동시에 수행할 프로세스 수(기본 {DEFAULT_CHUNK_WORKERS})",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
        help=f"청크 최대 문자 수(기본 {DEFAULT_CHUNK_SIZE}). PDF 25종·위키백과 등 다른 "
             "코퍼스와 동일한 값이 기본이므로, 바꾸면 코퍼스 간 청킹 방식이 달라짐에 유의.",
    )
    parser.add_argument(
        "--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP,
        help=f"인접 청크 간 중복 문자 수(기본 {DEFAULT_CHUNK_OVERLAP})",
    )
    parser.add_argument(
        "--domains", type=str, default=None,
        help="이번 실행에서 처리할 domain 폴더명만 지정(쉼표로 구분). 예: --domains medical "
             "또는 --domains law,medical. 지정하지 않으면 전체 도메인을 처리한다.",
    )
    parser.add_argument(
        "--skip-domains", type=str, default=None,
        help="이번 실행에서 건너뛸 domain 폴더명(쉼표로 구분). 예: --skip-domains finance",
    )
    parser.add_argument(
        "--skip-text-probe", action="store_true",
        help="사전 텍스트 프로브를 건너뛰고 모든 PDF를 그대로 청킹 시도(기본은 프로브 수행)",
    )
    parser.add_argument(
        "--probe-pages", type=int, default=8,
        help="사전 텍스트 프로브 시 확인할 앞쪽 페이지 수(기본 8, 표지·목차 이미지 페이지를 지나쳐 본문 시작을 잡기 위함)",
    )
    parser.add_argument(
        "--probe-min-chars", type=int, default=50,
        help="사전 텍스트 프로브 통과 기준 최소 추출 문자 수(기본 50, 미만이면 스캔 이미지로 간주해 건너뜀)",
    )
    args = parser.parse_args()
    domains = set(args.domains.split(",")) if args.domains else None
    skip_domains = set(args.skip_domains.split(",")) if args.skip_domains else None

    if not args.skip_download:
        rows = load_document_index()
        print(f"documents.csv에서 {len(rows)}건의 문서를 확인했습니다.")
        download_all(rows, limit=args.limit)

    if not args.skip_ingest:
        ingest_to_vectorstore(
            reset_state=args.reset_state, chunk_workers=args.chunk_workers,
            chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap,
            domains=domains, skip_domains=skip_domains,
            skip_text_probe=args.skip_text_probe, probe_pages=args.probe_pages,
            probe_min_chars=args.probe_min_chars,
        )


if __name__ == "__main__":
    main()
