"""
ingest_finance.py
===================
금융감독원 전자공시시스템(DART)에서 코스피 시가총액 상위 100개 기업의 최신
사업보고서를 PDF로 내려받아 `ch03/data/allganize_rag_ko/finance/`에 저장하는
파이프라인. (ingest_allganize.py가 채워 둔 finance 도메인 코퍼스를 국내
대표 상장기업의 실제 사업보고서로 보강한다.)

절차:
  1) 네이버 금융 시가총액 순위 페이지에서 코스피 상위 종목을 스크래핑하고,
     우선주(종목명이 "...우"/"...2우B" 형태)를 걸러내 보통주 100개를 고른다.
  2) DART Open API(corpCode.xml)로 종목코드 -> corp_code를 매핑한다.
  3) 기업별로 list.json(pblntf_detail_ty=A001)을 호출해 가장 최근 사업보고서의
     접수번호(rcept_no)를 찾는다.
  4) DART 공시서류 뷰어(dsaf001/main.do)를 방문해 문서 트리(treeData)에서
     사업보고서 본문의 dcm_no를 파싱한다. Open API는 원문을 PDF가 아닌
     HTML/XML 묶음(zip)으로만 제공하므로, PDF 자체는 공시서류 뷰어의 실제
     다운로드 경로(/pdf/download/pdf.do)를 3단계 세션 흐름
     (뷰어 페이지 -> 다운로드 안내 페이지 -> 실제 PDF)으로 그대로 재현해 받는다.

DART Open API 인증키는 .env의 DART_API_KEY로 관리하며 코드에 하드코딩하지 않는다.

실행 방법: ch03 디렉터리에서 실행.
    python3 ingest_finance.py                 # 상위 100개 기업 사업보고서 다운로드
    python3 ingest_finance.py --limit 5        # 앞 5건만(테스트용)
    python3 ingest_finance.py --skip-existing false  # 이미 받은 파일도 재다운로드
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import time
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Optional

import requests

# --------------------------------------------------
# 0. 설정
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent  # ch03/
FINANCE_DIR = BASE_DIR / "data" / "allganize_rag_ko" / "finance"
STATE_FILE = FINANCE_DIR / ".ingest_finance_state.json"
REPORT_FILE = FINANCE_DIR / "ingest_finance_download_report.csv"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}
TIMEOUT = 30
TOP_N = 100

_PREFERRED_STOCK_RE = re.compile(r".*\d?우[A-Z]?$")
_TREE_ROOT_DCM_RE = re.compile(
    r"node1\['id'\]\s*=\s*\"1\";\s*\r?\n\s*node1\['rcpNo'\]\s*=\s*\"\d+\";\s*\r?\n"
    r"\s*node1\['dcmNo'\]\s*=\s*\"(\d+)\""
)


def _dart_api_key() -> str:
    key = os.environ.get("DART_API_KEY")
    if key:
        return key
    env_path = BASE_DIR.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("DART_API_KEY="):
                return line.split("=", 1)[1].strip()
    raise RuntimeError(
        "DART_API_KEY를 찾을 수 없습니다. 저장소 루트 .env에 DART_API_KEY=... 형식으로 넣거나 "
        "환경변수로 export 하세요."
    )


# --------------------------------------------------
# 1. 코스피 시가총액 상위 종목 스크래핑 (네이버 금융, 인증 불필요)
# --------------------------------------------------
def fetch_kospi_top_companies(
    n: int, corp_map: dict[str, dict], session: requests.Session
) -> list[tuple[str, str, dict]]:
    """시가총액 내림차순으로 (종목코드, 네이버표시명, corp_map 항목) n개를 반환한다.

    네이버 금융의 "시가총액" 순위 페이지는 개별 기업뿐 아니라 ETF/펀드도 순자산
    총액 기준으로 함께 순위에 섞여 나온다(예: KODEX 200, TIGER 미국S&P500).
    ETF는 DART에 사업보고서를 내지 않는 별도 등록 체계라 corp_code 매핑이
    없으므로, 이를 rank 배정 "이후"에 걸러내면 최종 순위에 구멍이 생긴다.
    그래서 corp_map에 실제로 매핑되는 종목만 세어 가며 rank를 매겨, 최종
    결과가 항상 "DART에 사업보고서를 내는 진짜 기업" 상위 n개가 되도록 한다.
    """
    from bs4 import BeautifulSoup

    companies: list[tuple[str, str, dict]] = []
    seen_codes = set()
    page = 1
    while len(companies) < n and page <= 20:
        resp = session.get(
            "https://finance.naver.com/sise/sise_market_sum.naver",
            params={"sosok": 0, "page": page},
            headers=HEADERS,
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        table = soup.select_one("table.type_2")
        rows = table.select("a.tltle") if table else []
        if not rows:
            break
        for a in rows:
            name = a.get_text(strip=True)
            m = re.search(r"code=(\d+)", a["href"])
            if not m or _PREFERRED_STOCK_RE.match(name):
                continue
            code = m.group(1)
            if code in seen_codes:
                continue
            seen_codes.add(code)
            corp = corp_map.get(code)
            if corp is None:
                continue  # ETF/펀드 등 DART 미등록 종목 - 순위에서 제외
            companies.append((code, name, corp))
            if len(companies) >= n:
                break
        page += 1
        time.sleep(0.3)
    return companies


# --------------------------------------------------
# 2. DART corpCode.xml: 종목코드 -> corp_code 매핑
# --------------------------------------------------
def fetch_corp_code_map(api_key: str) -> dict[str, dict]:
    """{stock_code: {"corp_code":..., "corp_name":...}} 매핑을 반환한다."""
    import xml.etree.ElementTree as ET

    resp = requests.get(
        "https://opendart.fss.or.kr/api/corpCode.xml",
        params={"crtfc_key": api_key},
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    with zipfile.ZipFile(BytesIO(resp.content)) as z:
        xml_bytes = z.read(z.namelist()[0])
    root = ET.fromstring(xml_bytes)

    mapping: dict[str, dict] = {}
    for el in root.findall("list"):
        stock_code = (el.findtext("stock_code") or "").strip()
        if stock_code:
            mapping[stock_code] = {
                "corp_code": el.findtext("corp_code").strip(),
                "corp_name": el.findtext("corp_name").strip(),
            }
    return mapping


# --------------------------------------------------
# 3. 최신 사업보고서 접수번호(rcept_no) 조회
# --------------------------------------------------
def find_latest_business_report(corp_code: str, api_key: str) -> Optional[dict]:
    """가장 최근 사업보고서 1건을 반환한다.

    list.json의 정렬 파라미터(sort=date&sort_mtd=desc)를 실측해보면 오히려
    접수일자 오름차순으로 반환되는 경우가 있어(문서화된 동작과 다름), 정렬
    파라미터를 신뢰하지 않고 반환된 목록 중 rcept_dt가 가장 큰 항목을
    직접 골라 최신 보고서를 확정한다.
    """
    resp = requests.get(
        "https://opendart.fss.or.kr/api/list.json",
        params={
            "crtfc_key": api_key,
            "corp_code": corp_code,
            "pblntf_detail_ty": "A001",  # 사업보고서
            "bgn_de": "20200101",
            "end_de": "30001231",
            "page_count": 10,
        },
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("status") != "000":
        return None
    items = data.get("list") or []
    if not items:
        return None
    return max(items, key=lambda item: item["rcept_dt"])


# --------------------------------------------------
# 4. 공시서류 뷰어에서 본문 문서의 dcm_no 파싱 + PDF 다운로드
# --------------------------------------------------
def _find_root_dcm_no(rcept_no: str, session: requests.Session) -> Optional[str]:
    resp = session.get(
        "https://dart.fss.or.kr/dsaf001/main.do",
        params={"rcpNo": rcept_no},
        headers=HEADERS,
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    m = _TREE_ROOT_DCM_RE.search(resp.text)
    return m.group(1) if m else None


def download_business_report_pdf(
    rcept_no: str, dcm_no: str, dest_path: Path, session: requests.Session
) -> None:
    """뷰어 페이지 -> 다운로드 안내 페이지 -> 실제 PDF 순서로 세션을 그대로 재현한다.

    각 단계에서 이전 페이지 URL을 Referer로 넘겨야 서버가 팝업 경유 요청으로
    인식한다(그렇지 않으면 "잘못된 다운로드 요청" HTML 오류 페이지를 반환한다).
    """
    viewer_url = f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={rcept_no}"
    landing_url = f"https://dart.fss.or.kr/pdf/download/main.do?rcp_no={rcept_no}&dcm_no={dcm_no}"

    session.get(viewer_url, headers=HEADERS, timeout=TIMEOUT)
    session.get(landing_url, headers={**HEADERS, "Referer": viewer_url}, timeout=TIMEOUT)
    resp = session.get(
        "https://dart.fss.or.kr/pdf/download/pdf.do",
        params={"rcp_no": rcept_no, "dcm_no": dcm_no},
        headers={**HEADERS, "Referer": landing_url},
        timeout=60,
    )
    resp.raise_for_status()
    if resp.content[:5] != b"%PDF-":
        raise ValueError(f"PDF 응답이 아님(첫 5바이트: {resp.content[:5]!r})")

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    dest_path.write_bytes(resp.content)


# --------------------------------------------------
# 5. 파일명 정리 (ingest_allganize.py의 sanitize_filename과 동일한 정책)
# --------------------------------------------------
def sanitize_filename(name: str) -> str:
    name = name.replace("/", "_").replace("\\", "_")
    name = "".join(ch for ch in name if ch.isprintable()).strip() or "unnamed.pdf"
    encoded = name.encode("utf-8")
    if len(encoded) > 250:
        name = encoded[:250].decode("utf-8", errors="ignore")
    return name


# --------------------------------------------------
# 6. 실행부
# --------------------------------------------------
def load_state() -> set[str]:
    import json

    if STATE_FILE.exists():
        return set(json.loads(STATE_FILE.read_text(encoding="utf-8")))
    return set()


def save_state(done: set[str]) -> None:
    import json

    STATE_FILE.write_text(json.dumps(sorted(done), ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=TOP_N, help="상위 N개 기업만 처리(기본 100)")
    parser.add_argument(
        "--skip-existing", type=lambda s: s.lower() != "false", default=True,
        help="이미 받은 회사(상태 파일 기준)는 건너뜀(기본 true)",
    )
    args = parser.parse_args()

    api_key = _dart_api_key()
    FINANCE_DIR.mkdir(parents=True, exist_ok=True)

    print("DART corp_code 매핑 로딩 중...")
    corp_map = fetch_corp_code_map(api_key)

    print("코스피 시가총액 상위 종목(DART 등록 기업만) 조회 중...")
    session = requests.Session()
    companies = fetch_kospi_top_companies(args.limit, corp_map, session)
    print(f"{len(companies)}개 기업 확보")

    done = load_state() if args.skip_existing else set()
    report_rows = []
    ok_count = 0

    with open(REPORT_FILE, "w", encoding="utf-8", newline="") as report_f:
        writer = csv.DictWriter(
            report_f, fieldnames=["rank", "stock_code", "corp_name", "rcept_no", "status", "reason"]
        )
        writer.writeheader()

        for rank, (stock_code, name, corp) in enumerate(companies, start=1):
            corp_code, corp_name = corp["corp_code"], corp["corp_name"]
            if corp_code in done:
                print(f"[{rank}/{len(companies)}] {corp_name}: 이미 처리됨, 건너뜀")
                continue

            try:
                latest = find_latest_business_report(corp_code, api_key)
                if latest is None:
                    raise ValueError("사업보고서 공시 이력 없음")
                rcept_no = latest["rcept_no"]

                dcm_no = _find_root_dcm_no(rcept_no, session)
                if dcm_no is None:
                    raise ValueError("문서 트리에서 dcm_no를 찾지 못함")

                file_name = sanitize_filename(
                    f"{rank:03d}_{corp_name}_사업보고서_{latest['rcept_dt']}.pdf"
                )
                dest_path = FINANCE_DIR / file_name
                download_business_report_pdf(rcept_no, dcm_no, dest_path, session)

                print(f"[{rank}/{len(companies)}] {corp_name}: 다운로드 완료 ({file_name})")
                entry = {"rank": rank, "stock_code": stock_code, "corp_name": corp_name,
                          "rcept_no": rcept_no, "status": "downloaded", "reason": file_name}
                ok_count += 1
                done.add(corp_code)
                save_state(done)
            except Exception as e:  # noqa: BLE001 - 한 회사 실패가 전체 배치를 죽이지 않도록
                print(f"[{rank}/{len(companies)}] {corp_name}: 실패 - {e}")
                entry = {"rank": rank, "stock_code": stock_code, "corp_name": corp_name,
                          "rcept_no": "", "status": "failed", "reason": str(e)}

            report_rows.append(entry)
            writer.writerow(entry)
            report_f.flush()
            time.sleep(0.5)  # DART/뷰어 서버에 대한 예의상 지연

    failed = [r for r in report_rows if r["status"] == "failed"]
    print(f"\n다운로드 결과: 성공 {ok_count}/{len(companies)}건, 실패 {len(failed)}건")
    if failed:
        print("실패 목록:")
        for r in failed:
            print(f"  - [{r['rank']}] {r['corp_name']}: {r['reason']}")
    print(f"상세 결과: {REPORT_FILE}")


if __name__ == "__main__":
    main()
