"""
ingest_medical.py
===================
질병관리청(KDCA) 홈페이지의 "지침" 게시판(bcIdx=55, 감염병 관리지침·예방접종
지침 등)에서 최신 게시물 순으로 첨부 PDF를 내려받아
`ch03/data/allganize_rag_ko/medical/`에 저장하는 파이프라인.

절차:
  1) 게시판이 제공하는 RSS 피드(rssList.do?row=N)로 게시물 목록(제목, 상세
     페이지 링크)을 한 번에 받는다. row 값을 요청 건수보다 넉넉히 잡아
     첨부파일이 없는 게시물(공지성 게시물 등)을 걸러내고도 목표 건수를
     채울 수 있게 한다.
  2) 각 게시물 상세 페이지(artclView.do)에서 첨부파일 목록을 파싱해, 그중
     ".pdf" 확장자인 첫 번째 첨부파일의 다운로드 링크(download.do)를 찾는다.
     한 게시물에 첨부파일이 여러 개(본책/별책 등)인 경우가 있어 그중 대표
     1건만 받는다.
  3) download.do는 DART/law.go.kr과 달리 세션·Referer 없이 바로 실제 PDF를
     반환하므로 별도 흉내 없이 그대로 GET한다.

실행 방법: ch03 디렉터리에서 실행.
    python3 ingest_medical.py                 # 최신 100건 다운로드
    python3 ingest_medical.py --limit 5        # 앞 5건만(테스트용)
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
MEDICAL_DIR = BASE_DIR / "data" / "allganize_rag_ko" / "medical"
STATE_FILE = MEDICAL_DIR / ".ingest_medical_state.json"
REPORT_FILE = MEDICAL_DIR / "ingest_medical_download_report.csv"

BOARD_URL = "https://www.kdca.go.kr/bbs/kdca/55"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}
TIMEOUT = 30
TOP_N = 100

_ATTACHMENT_BLOCK_RE = re.compile(r'class="attachment">(.*?)</ul>', re.S)
_ATTACHMENT_ITEM_RE = re.compile(
    r'<li>\s*(?:<!--.*?-->\s*)?([^<]+?)\s*<div>\s*<a class="attch-down" download href="([^"]+)"',
    re.S,
)


# --------------------------------------------------
# 1. 게시물 목록(RSS)
# --------------------------------------------------
def fetch_article_list(n: int, session: requests.Session) -> list[dict]:
    """최신순으로 게시물 n건(제목, 상세 링크)을 반환한다. 여유분을 두려면 n을
    실제 필요 건수보다 크게 준다."""
    resp = session.get(
        f"{BOARD_URL}/rssList.do", params={"row": n}, headers=HEADERS, timeout=TIMEOUT
    )
    resp.raise_for_status()
    items = re.findall(r"<item>(.*?)</item>", resp.text, re.S)
    articles = []
    for item in items:
        title = re.search(r"<title><!\[CDATA\[(.*?)\]\]></title>", item).group(1)
        # KDCA RSS 피드는 제목 끝에 원본 데이터 자체의 "}" 잔재가 항상 붙어 나온다
        # (사이트 쪽 데이터 이슈로 실측 확인됨, 내용과 무관해 파일명에서 제거한다).
        title = title.rstrip("}").strip()
        link = re.search(r"<link>(.*?)</link>", item).group(1)
        articles.append({"title": title, "url": "https://www.kdca.go.kr" + link})
    return articles


# --------------------------------------------------
# 2. 상세 페이지에서 대표 PDF 첨부 링크 찾기
# --------------------------------------------------
def find_pdf_attachment(article_url: str, session: requests.Session) -> Optional[tuple[str, str]]:
    """상세 페이지의 첨부파일 목록에서 .pdf 확장자인 첫 번째 항목의
    (파일명, 다운로드 URL)을 반환한다. 없으면 None."""
    resp = session.get(article_url, headers=HEADERS, timeout=TIMEOUT)
    resp.raise_for_status()
    block = _ATTACHMENT_BLOCK_RE.search(resp.text)
    if not block:
        return None
    for m in _ATTACHMENT_ITEM_RE.finditer(block.group(1)):
        file_name, href = m.group(1).strip(), m.group(2).strip()
        if file_name.lower().endswith(".pdf"):
            return file_name, "https://www.kdca.go.kr" + href
    return None


def download_pdf(url: str, dest_path: Path, session: requests.Session) -> None:
    resp = session.get(url, headers=HEADERS, timeout=60)
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

    MEDICAL_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()

    # 첨부파일이 없는 게시물(공지 등)을 감안해 목표 건수의 1.3배 정도를 후보로 확보한다.
    candidate_n = max(args.limit + 30, int(args.limit * 1.3))
    print(f"게시물 목록 조회 중 (후보 {candidate_n}건)...")
    articles = fetch_article_list(candidate_n, session)
    print(f"{len(articles)}건 확보\n")

    done = load_state() if args.skip_existing else set()
    report_rows = []
    ok_count = 0
    rank = 0

    with open(REPORT_FILE, "w", encoding="utf-8", newline="") as report_f:
        writer = csv.DictWriter(
            report_f, fieldnames=["rank", "title", "article_url", "status", "reason"]
        )
        writer.writeheader()

        for article in articles:
            if ok_count >= args.limit:
                break
            if article["url"] in done:
                rank += 1
                print(f"[{rank}] {article['title'][:40]}: 이미 처리됨, 건너뜀")
                ok_count += 1  # 이미 받은 것도 목표 건수에 포함
                continue

            rank += 1
            try:
                found = find_pdf_attachment(article["url"], session)
                if found is None:
                    raise ValueError("PDF 첨부파일 없음")
                attach_name, download_url = found

                file_name = sanitize_filename(f"{ok_count + 1:03d}_{article['title']}.pdf")
                dest_path = MEDICAL_DIR / file_name
                download_pdf(download_url, dest_path, session)

                print(f"[{rank}] {article['title'][:40]}: 다운로드 완료 ({file_name})")
                entry = {"rank": ok_count + 1, "title": article["title"],
                          "article_url": article["url"], "status": "downloaded", "reason": file_name}
                ok_count += 1
                done.add(article["url"])
                save_state(done)
            except Exception as e:  # noqa: BLE001 - 한 건 실패가 전체 배치를 죽이지 않도록
                print(f"[{rank}] {article['title'][:40]}: 실패 - {e}")
                entry = {"rank": "", "title": article["title"],
                          "article_url": article["url"], "status": "failed", "reason": str(e)}

            report_rows.append(entry)
            writer.writerow(entry)
            report_f.flush()
            time.sleep(0.3)  # KDCA 서버에 대한 예의상 지연

    failed = [r for r in report_rows if r["status"] == "failed"]
    print(f"\n다운로드 결과: 성공 {ok_count}/{args.limit}건(목표), 실패 {len(failed)}건")
    if failed:
        print("실패 목록:")
        for r in failed:
            print(f"  - {r['title'][:50]}: {r['reason']}")
    if ok_count < args.limit:
        print(
            f"목표 건수({args.limit})를 채우지 못했습니다. 후보를 늘리려면 "
            f"--limit는 그대로 두고 candidate_n 배율을 높이거나 스크립트를 다시 실행하세요."
        )
    print(f"상세 결과: {REPORT_FILE}")


if __name__ == "__main__":
    main()
