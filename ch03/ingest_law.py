"""
ingest_law.py
===============
국가법령정보센터(law.go.kr) Open API에서 민법·형법·개인정보 보호법 세 법률의
역대 개정판(시행일자별 연혁) 전문을 받아 PDF로 만들어
`ch03/data/allganize_rag_ko/law/`에 저장하는 파이프라인.

배경: law.go.kr Open API는 검색(lawSearch.do)과 조문 XML 본문(lawService.do,
type=XML)은 데모 키(OC=test)로도 열람할 수 있지만, 사이트 자체의 PDF
내보내기(HTML 렌더링, lsPdfPrint.do)는 정식 등록된 기관코드(OC)에 "법령"
API 신청까지 마쳐야 열람할 수 있어("미신청된 목록/본문에 대한 접근입니다"
오류) 이 파이프라인에서는 사용하지 않는다. 대신 XML로 받은 조문(조/항/호)
구조를 그대로 파싱해 이 스크립트가 직접 HTML -> PDF(WeasyPrint)로 렌더링한다.

대상 3개 법률과 부족분 보충:
    - 민법(법령ID 001706): 역대 개정판 39건
    - 형법(법령ID 001692): 역대 개정판 34건
    - 개인정보 보호법(법령ID 011357): 역대 개정판 22건
    합계 95건이라 100건에 5건 부족하므로, 개인정보 보호법의 시행령
    (법령ID 011468, 역대 개정판 37건) 중 최신 5건으로 채운다.

DART API 키는 필요 없으며(law.go.kr Open API는 OC=test로 접근), 별도
등록 없이 바로 실행 가능하다.

실행 방법: ch03 디렉터리에서 실행.
    python3 ingest_law.py                 # 100건 전체 다운로드
    python3 ingest_law.py --limit 5        # 앞 5건만(테스트용)
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests
import weasyprint

# --------------------------------------------------
# 0. 설정
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent  # ch03/
LAW_DIR = BASE_DIR / "data" / "allganize_rag_ko" / "law"
STATE_FILE = LAW_DIR / ".ingest_law_state.json"
REPORT_FILE = LAW_DIR / "ingest_law_download_report.csv"

OC = "test"  # law.go.kr Open API 데모 키. 검색·XML 본문 열람에는 등록 없이도 동작한다.
API_BASE = "https://www.law.go.kr/DRF"
TIMEOUT = 20

# (법령ID, 검색어, 목표 건수). 목표 건수를 채우지 못하면 있는 만큼만 받는다.
PRIMARY_TARGETS = [
    ("001706", "민법", None),       # 39건 전부
    ("001692", "형법", None),       # 34건 전부
    ("011357", "개인정보 보호법", None),  # 22건 전부
]
# 위 세 법률 역대 개정판 합계(39+34+22=95)가 100에 못 미치는 만큼,
# 개인정보 보호법 시행령의 최신 버전으로 채운다.
PADDING_TARGET = ("011468", "개인정보 보호법 시행령")

TOP_N = 100


# --------------------------------------------------
# 1. 법령 연혁(시행일자별 버전) 목록 조회
# --------------------------------------------------
@dataclass
class LawVersion:
    law_id: str
    law_name: str
    mst: str  # 법령일련번호 - lawService.do 조회 키
    efyd: str  # 시행일자
    promulgation_date: str
    revision_type: str


def fetch_law_history(law_id: str, query: str, session: requests.Session) -> list[LawVersion]:
    """query로 검색되는 법령들 중 law_id와 정확히 일치하는 법령의 모든 시행일자별
    버전(연혁)을 최신순으로 반환한다.

    law.go.kr 검색은 법령명 부분일치라 "민법"으로 검색하면 "난민법", "민법법인
    및 특수법인 등기규칙" 등도 함께 나온다. 이후 법령ID로 정확히 걸러낸다.
    """
    versions: list[LawVersion] = []
    page = 1
    while True:
        resp = session.get(
            f"{API_BASE}/lawSearch.do",
            params={"OC": OC, "target": "eflaw", "type": "XML", "query": query,
                    "display": 100, "page": page},
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        blocks = re.findall(r'<law id="\d+">(.*?)</law>', resp.text, re.S)
        if not blocks:
            break
        for block in blocks:
            lid = re.search(r"<법령ID>(\d+)</법령ID>", block).group(1)
            if lid != law_id:
                continue
            name = re.search(r"법령명한글><!\[CDATA\[(.*?)\]\]", block).group(1)
            mst = re.search(r"<법령일련번호>(\d+)</법령일련번호>", block).group(1)
            efyd = re.search(r"<시행일자>(\d+)</시행일자>", block).group(1)
            promulgation = re.search(r"<공포일자>(\d+)</공포일자>", block).group(1)
            revision = re.search(r"<제개정구분명>(.*?)</제개정구분명>", block).group(1)
            versions.append(LawVersion(lid, name, mst, efyd, promulgation, revision))
        total_cnt = int(re.search(r"<totalCnt>(\d+)</totalCnt>", resp.text).group(1))
        if page * 100 >= total_cnt:
            break
        page += 1
        time.sleep(0.2)

    # 같은 mst가 중복 조회될 수 있어 정리하고 시행일자 내림차순 정렬
    dedup = {v.mst: v for v in versions}
    return sorted(dedup.values(), key=lambda v: v.efyd, reverse=True)


# --------------------------------------------------
# 2. 조문 XML 본문 조회 + PDF 렌더링
# --------------------------------------------------
def fetch_law_detail(mst: str, session: requests.Session) -> ET.Element:
    """조문 XML 본문을 받는다.

    target=eflaw(시행일자별 연혁 조회)가 일부 mst에 대해 "페이지를 찾을 수
    없습니다" 500 오류를 반환하는 경우가 실측으로 확인되어(law.go.kr 쪽 데이터
    정합성 이슈로 추정), 이 경우 target=law로 같은 mst를 재시도한다. 두
    엔드포인트 모두 같은 법령일련번호 체계를 공유하며, target=law도 해당
    시행일자 버전의 본문을 정확히 반환함을 실측으로 확인했다.
    """
    last_exc: Exception | None = None
    for target in ("eflaw", "law"):
        try:
            resp = session.get(
                f"{API_BASE}/lawService.do",
                params={"OC": OC, "target": target, "MST": mst, "type": "XML"},
                timeout=TIMEOUT,
            )
            resp.raise_for_status()
            return ET.fromstring(resp.content)
        except requests.RequestException as e:
            last_exc = e
            continue
    raise last_exc


_CONTENT_INDENT = {"조문내용": "", "항내용": "1.5em", "호내용": "3em", "목내용": "4.5em"}


def _article_html(jomun: ET.Element) -> str:
    """조문단위 하나(조/항/호/목)를 들여쓰기가 반영된 HTML 문단들로 변환한다."""
    parts = []
    for el in jomun.iter():
        if el.tag in _CONTENT_INDENT and el.text and el.text.strip():
            indent = _CONTENT_INDENT[el.tag]
            text = el.text.strip().replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            style = f"margin-left:{indent};" if indent else "margin-top:0.8em;font-weight:bold;"
            parts.append(f'<p style="{style}">{text}</p>')
    return "".join(parts)


def render_law_pdf(root: ET.Element, dest_path: Path) -> None:
    info = root.find("기본정보")
    name = (info.findtext("법령명_한글") or "").strip()
    efyd = (info.findtext("시행일자") or "").strip()
    promulgation_date = (info.findtext("공포일자") or "").strip()
    promulgation_no = (info.findtext("공포번호") or "").strip()
    revision = (info.findtext("제개정구분") or "").strip()
    ministry = ""
    ministry_el = info.find("소관부처")
    if ministry_el is not None:
        ministry = (ministry_el.text or "").strip()

    body_html = "".join(_article_html(jomun) for jomun in root.iter("조문단위"))

    html = f"""<html><head><meta charset="utf-8"></head>
<body style="font-family: 'Noto Serif CJK KR', 'Noto Sans CJK KR', serif; font-size: 10.5pt; line-height: 1.6;">
<h1 style="font-size: 18pt;">{name}</h1>
<p style="color:#444;">
  시행 {efyd} · 공포 {promulgation_date} (제{promulgation_no}호, {revision}) · 소관부처: {ministry}
</p>
<hr>
{body_html}
</body></html>"""

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    weasyprint.HTML(string=html).write_pdf(str(dest_path))


# --------------------------------------------------
# 3. 파일명 정리 (다른 ingest 모듈과 동일한 정책)
# --------------------------------------------------
def sanitize_filename(name: str) -> str:
    name = name.replace("/", "_").replace("\\", "_")
    name = "".join(ch for ch in name if ch.isprintable()).strip() or "unnamed.pdf"
    encoded = name.encode("utf-8")
    if len(encoded) > 250:
        name = encoded[:250].decode("utf-8", errors="ignore")
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


def build_target_list(limit: int, session: requests.Session) -> list[LawVersion]:
    """민법/형법/개인정보 보호법의 연혁 전체를 모으고, limit에 못 미치면
    개인정보 보호법 시행령의 최신 버전으로 부족분을 채운다."""
    targets: list[LawVersion] = []
    for law_id, query, _ in PRIMARY_TARGETS:
        versions = fetch_law_history(law_id, query, session)
        print(f"  {query}: 역대 개정판 {len(versions)}건 확보")
        targets.extend(versions)

    if len(targets) < limit:
        pad_id, pad_query = PADDING_TARGET
        need = limit - len(targets)
        pad_versions = fetch_law_history(pad_id, pad_query, session)
        print(f"  (부족분 보충) {pad_query}: 최신 {min(need, len(pad_versions))}건 추가")
        targets.extend(pad_versions[:need])

    return targets[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=TOP_N, help="다운로드할 총 건수(기본 100)")
    parser.add_argument(
        "--skip-existing", type=lambda s: s.lower() != "false", default=True,
        help="이미 받은 버전(상태 파일 기준)은 건너뜀(기본 true)",
    )
    args = parser.parse_args()

    LAW_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()

    print("법령 연혁 목록 조회 중...")
    targets = build_target_list(args.limit, session)
    print(f"총 {len(targets)}건 확보\n")

    done = load_state() if args.skip_existing else set()
    report_rows = []
    ok_count = 0

    with open(REPORT_FILE, "w", encoding="utf-8", newline="") as report_f:
        writer = csv.DictWriter(
            report_f, fieldnames=["rank", "law_name", "efyd", "mst", "status", "reason"]
        )
        writer.writeheader()

        for rank, v in enumerate(targets, start=1):
            if v.mst in done:
                print(f"[{rank}/{len(targets)}] {v.law_name}({v.efyd}): 이미 처리됨, 건너뜀")
                continue

            try:
                root = fetch_law_detail(v.mst, session)
                file_name = sanitize_filename(f"{rank:03d}_{v.law_name}_{v.efyd}.pdf")
                dest_path = LAW_DIR / file_name
                render_law_pdf(root, dest_path)

                print(f"[{rank}/{len(targets)}] {v.law_name}({v.efyd}): 다운로드 완료 ({file_name})")
                entry = {"rank": rank, "law_name": v.law_name, "efyd": v.efyd, "mst": v.mst,
                          "status": "downloaded", "reason": file_name}
                ok_count += 1
                done.add(v.mst)
                save_state(done)
            except Exception as e:  # noqa: BLE001 - 한 건 실패가 전체 배치를 죽이지 않도록
                print(f"[{rank}/{len(targets)}] {v.law_name}({v.efyd}): 실패 - {e}")
                entry = {"rank": rank, "law_name": v.law_name, "efyd": v.efyd, "mst": v.mst,
                          "status": "failed", "reason": str(e)}

            report_rows.append(entry)
            writer.writerow(entry)
            report_f.flush()
            time.sleep(0.3)  # law.go.kr 서버에 대한 예의상 지연

    failed = [r for r in report_rows if r["status"] == "failed"]
    print(f"\n다운로드 결과: 성공 {ok_count}/{len(targets)}건, 실패 {len(failed)}건")
    if failed:
        print("실패 목록:")
        for r in failed:
            print(f"  - [{r['rank']}] {r['law_name']}({r['efyd']}): {r['reason']}")
    print(f"상세 결과: {REPORT_FILE}")


if __name__ == "__main__":
    main()
