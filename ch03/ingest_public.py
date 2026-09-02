"""
ingest_public.py
========================
대한민국 정책브리핑(korea.kr) "정책자료 > 전문자료"(정부 각 부처가 작성한
정책 보고서·백서·가이드북 등) 게시판에서 최신 게시물 순으로 첨부 PDF를
받아 `ch03/data/allganize_rag_ko/public/`에 저장하는 파이프라인.

(AI Hub 공공행정 QA 데이터셋은 계정별 API 키·데이터셋 이용신청 승인이
필요해 이번 파이프라인에서는 제외했다 — 사용자 확인 결과 정부 정책
PDF만으로 100건을 채우기로 함.)

절차:
  1) 목록은 `/archive/expDocList.do?group=S`를 POST로 페이지네이션
     (pageIndex, 페이지당 20건)해서 docId 목록을 모은다.
  2) 각 게시물 상세 페이지(`/archive/expDocView.do?docId=...&group=S`)의
     "첨부파일" 영역에서 .pdf 확장자인 첫 번째 첨부파일의
     (tblKey, fileId)를 파싱한다.
  3) `/common/download.do?tblKey=...&fileId=...`로 바로 실제 PDF를 받는다
     (KDCA와 마찬가지로 세션·Referer 흉내가 필요 없다).

실행 방법: ch03 디렉터리에서 실행.
    python3 ingest_public.py                 # 최신 100건 다운로드
    python3 ingest_public.py --limit 5        # 앞 5건만(테스트용)
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path
from typing import Optional

import requests

# --------------------------------------------------
# 0. 설정
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent  # ch03/
PUBLIC_DIR = BASE_DIR / "data" / "allganize_rag_ko" / "public"
STATE_FILE = PUBLIC_DIR / ".ingest_public_state.json"
REPORT_FILE = PUBLIC_DIR / "ingest_public_download_report.csv"

SITE = "https://www.korea.kr"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}
TIMEOUT = 30
TOP_N = 100
PAGE_SIZE = 20

_FILEDOWN_BLOCK_RE = re.compile(r'filedown">(.*?)</dd>', re.S)
_ATTACHMENT_ITEM_RE = re.compile(
    r'<a href="/common/download\.do\?tblKey=([^&]+)&amp;fileId=(\d+)">\s*'
    r'<img[^>]*/>\s*([^<]+)',
)


# --------------------------------------------------
# 1. 게시물 목록(페이지네이션)
# --------------------------------------------------
def fetch_article_ids(n: int, session: requests.Session) -> list[str]:
    """최신순으로 게시물 docId를 n개 이상 모을 때까지 페이지를 넘기며 수집한다."""
    doc_ids: list[str] = []
    seen = set()
    page = 1
    while len(doc_ids) < n and page <= 20:
        resp = session.post(
            f"{SITE}/archive/expDocList.do",
            params={"group": "S"},
            data={"pageIndex": page},
            headers=HEADERS,
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        ids = re.findall(r"docId=(\d+)", resp.text)
        if not ids:
            break
        for did in ids:
            if did not in seen:
                seen.add(did)
                doc_ids.append(did)
        page += 1
        time.sleep(0.2)
    return doc_ids


# --------------------------------------------------
# 2. 상세 페이지에서 제목 + 대표 PDF 첨부 찾기
# --------------------------------------------------
def fetch_article_title(text: str) -> str:
    m = re.search(r"<title>(.*?)</title>", text, re.S)
    title = m.group(1).strip() if m else "제목없음"
    title = title.split(" | ")[0].strip()
    title = re.sub(r"\s*-\s*주제별\s*$", "", title).strip()
    return title


def find_pdf_attachment(doc_id: str, session: requests.Session) -> Optional[tuple[str, str, str]]:
    """상세 페이지의 첨부파일 목록에서 .pdf 확장자인 첫 번째 항목의
    (제목, tblKey, fileId)를 반환한다. 없으면 None."""
    resp = session.get(
        f"{SITE}/archive/expDocView.do", params={"docId": doc_id, "group": "S"},
        headers=HEADERS, timeout=TIMEOUT,
    )
    resp.raise_for_status()
    title = fetch_article_title(resp.text)

    block = _FILEDOWN_BLOCK_RE.search(resp.text)
    if not block:
        return None
    for tbl_key, file_id, file_name in _ATTACHMENT_ITEM_RE.findall(block.group(1)):
        if file_name.strip().lower().endswith(".pdf"):
            return title, tbl_key, file_id
    return None


def download_pdf(tbl_key: str, file_id: str, dest_path: Path, session: requests.Session) -> None:
    resp = session.get(
        f"{SITE}/common/download.do", params={"tblKey": tbl_key, "fileId": file_id},
        headers=HEADERS, timeout=60,
    )
    resp.raise_for_status()
    if resp.content[:5] != b"%PDF-":
        raise ValueError(f"PDF 응답이 아님(첫 5바이트: {resp.content[:5]!r})")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    dest_path.write_bytes(resp.content)


# --------------------------------------------------
# 3. 파일명 정리 (다른 ingest 모듈과 동일한 정책)
# --------------------------------------------------
def sanitize_filename(name: str) -> str:
    name = name.replace("/", "_").replace("\\", "_")
    name = "".join(ch for ch in name if ch.isprintable()).strip() or "unnamed.pdf"
    encoded = name.encode("utf-8")
    if len(encoded) > 250:
        stem, dot, ext = name.rpartition(".")
        budget = 250 - len((dot + ext).encode("utf-8"))
        name = stem.encode("utf-8")[:budget].decode("utf-8", errors="ignore") + dot + ext
    return name


# --------------------------------------------------
# 4. 실행부
# --------------------------------------------------
def load_state() -> set[str]:
    if STATE_FILE.exists():
        return set(json.loads(STATE_FILE.read_text(encoding="utf-8")))
    return set()


def save_state(done: set[str]) -> None:
    STATE_FILE.write_text(json.dumps(sorted(done), ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=TOP_N, help="다운로드할 건수(기본 100)")
    parser.add_argument(
        "--skip-existing", type=lambda s: s.lower() != "false", default=True,
        help="이미 받은 게시물(상태 파일 기준)은 건너뜀(기본 true)",
    )
    args = parser.parse_args()

    PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()

    # PDF가 없는 게시물(HWP만 있거나 미첨부)을 감안해 목표 건수의 1.3배 정도 후보를 모은다.
    candidate_n = max(args.limit + 30, int(args.limit * 1.3))
    print(f"게시물 목록 조회 중 (후보 {candidate_n}건)...")
    doc_ids = fetch_article_ids(candidate_n, session)
    print(f"{len(doc_ids)}건 확보\n")

    done = load_state() if args.skip_existing else set()
    report_rows = []
    ok_count = 0
    rank = 0

    with open(REPORT_FILE, "w", encoding="utf-8", newline="") as report_f:
        writer = csv.DictWriter(
            report_f, fieldnames=["rank", "doc_id", "title", "status", "reason"]
        )
        writer.writeheader()

        for doc_id in doc_ids:
            if ok_count >= args.limit:
                break
            if doc_id in done:
                rank += 1
                ok_count += 1
                print(f"[{rank}] docId={doc_id}: 이미 처리됨, 건너뜀")
                continue

            rank += 1
            try:
                found = find_pdf_attachment(doc_id, session)
                if found is None:
                    raise ValueError("PDF 첨부파일 없음")
                title, tbl_key, file_id = found

                file_name = sanitize_filename(f"{ok_count + 1:03d}_{title}.pdf")
                dest_path = PUBLIC_DIR / file_name
                download_pdf(tbl_key, file_id, dest_path, session)

                print(f"[{rank}] {title[:40]}: 다운로드 완료 ({file_name})")
                entry = {"rank": ok_count + 1, "doc_id": doc_id, "title": title,
                          "status": "downloaded", "reason": file_name}
                ok_count += 1
                done.add(doc_id)
                save_state(done)
            except Exception as e:  # noqa: BLE001 - 한 건 실패가 전체 배치를 죽이지 않도록
                print(f"[{rank}] docId={doc_id}: 실패 - {e}")
                entry = {"rank": "", "doc_id": doc_id, "title": "",
                          "status": "failed", "reason": str(e)}

            report_rows.append(entry)
            writer.writerow(entry)
            report_f.flush()
            time.sleep(0.3)  # korea.kr 서버에 대한 예의상 지연

    failed = [r for r in report_rows if r["status"] == "failed"]
    print(f"\n다운로드 결과: 성공 {ok_count}/{args.limit}건(목표), 실패 {len(failed)}건")
    if failed:
        print("실패 목록:")
        for r in failed:
            print(f"  - docId={r['doc_id']}: {r['reason']}")
    if ok_count < args.limit:
        print(f"목표 건수({args.limit})를 채우지 못했습니다. --limit는 그대로 두고 다시 실행하면 이어받습니다.")
    print(f"상세 결과: {REPORT_FILE}")


if __name__ == "__main__":
    main()
