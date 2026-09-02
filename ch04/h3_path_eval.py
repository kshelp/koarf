"""
h3_path_eval.py
================
논문 4.1절 H3("검색 경로 전체 비교", RQ3) 검증 구현체.

기존 think_mode_eval.py(Table 6)는 질의 1건을 think_on/think_off 각각 3회
반복 측정하는 설계였다 — 측정 안정성은 확인할 수 있지만 "다수 질의에 대한
일반화"는 애초에 설계상 불가능했다(4.4절이 스스로 이 한계를 지적한 바 있음).
이 모듈은 golden set의 서로 다른 질의 10건(answer_eval_testset.json과 동일한
일반 도메인 질의, 참조 정답 보유)을 각각 1회씩 think_on/think_off 양쪽으로
측정하는 대응표본(paired) 설계로 바꾸어, 질의 간 반복측정으로 통계적 검정
(Wilcoxon 부호순위 검정)이 가능하도록 한다.

측정 항목:
    - 종단(E2E) 응답 시간: 저지연 경로(think_off) vs 정밀 경로(think_on) —
      H3의 핵심 주장("저지연 경로가 정밀 경로보다 평균 응답시간이 유의하게
      낮을 것")을 대응표본 Wilcoxon 검정으로 확인한다.
    - 정답 여부: 4.1절이 명시한 대로 "동일한 조건에서 최종 응답 품질을 함께
      측정하여 두 검색 경로 간 차이를 탐색적으로 분석"한다(McNemar 검정 병행).

결과는 질의 단위로 즉시 CSV에 append하여 중간에 중단되어도 이미 처리한
질의의 결과는 보존한다.

실행 방법: ch03, ch04 어디서 실행해도 무관하며 Ollama와 Qdrant가 떠 있어야 한다.
    python3 h3_path_eval.py
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from typing import List

sys.path.insert(0, os.path.dirname(__file__))

from think_mode_eval import run_think_on, run_think_off, ThinkModeRun  # noqa: E402

CH03_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ch03"))
OUT_DIR = os.path.join(os.path.dirname(__file__), "results")
CSV_FIELDS = [
    "item_id", "mode", "query_rewrite_sec", "search_rerank_sec",
    "llm_gen_sec", "total_sec", "correct",
]

# answer_eval_testset.json과 동일한 일반 도메인 질의 10건 + 정답 판정용 짧은 참조 문자열.
# (참조 문자열은 answer_eval_testset.json의 reference에서 검색 가능한 핵심 수치/구절을 뽑은 것)
QUERIES = [
    {"item_id": "a1", "query": "서울시의 탄소중립 목표 연도와 2030년까지의 온실가스 감축률은?", "reference_substring": "2050"},
    {"item_id": "a2", "query": "서울의 1인 가구 수는 2020년 기준 몇 만 가구인가?", "reference_substring": "139.1만"},
    {"item_id": "a3", "query": "서울의 2019년 지역내총생산(GRDP)은 얼마인가?", "reference_substring": "411.4조"},
    {"item_id": "a4", "query": "What is the gross domestic product of the New York City metropolitan economy?", "reference_substring": "1.9 trillion"},
    {"item_id": "a5", "query": "성경 창세기 1장 1절의 내용은?", "reference_substring": "천지를 창조"},
    {"item_id": "a6", "query": "Transformer 논문에서 WMT 2014 영어-독일어 번역 과제의 BLEU 점수는 얼마인가?", "reference_substring": "28.4"},
    {"item_id": "a7", "query": "BERT는 GLUE 벤치마크 점수를 몇 %까지 끌어올렸는가?", "reference_substring": "80.5"},
    {"item_id": "a8", "query": "LoRA는 GPT-3 175B를 완전 미세조정할 때와 비교해 학습 가능한 파라미터 수를 몇 배 줄이는가?", "reference_substring": "10,000"},
    {"item_id": "a9", "query": "바둑의 평균 분기 계수(branching factor)는 체스와 비교해 대략 얼마나 되는가?", "reference_substring": "250"},
    {"item_id": "a10", "query": "By what percentage does the dense passage retriever outperform the Lucene-BM25 system in top-20 passage retrieval accuracy?", "reference_substring": "9%"},
    # 특화 도메인(Allganize: 금융·법률·의료·공공·커머스) 20건 — 각 도메인 4건.
    # 참조 문자열은 allganize_rag_ko 컬렉션의 실제 청크 내용에서 직접 확인한 사실.
    {"item_id": "f1", "query": "삼성전자는 언제, 어떤 이름으로 처음 설립되었는가?", "reference_substring": "1969년"},
    {"item_id": "f3", "query": "SK스퀘어의 설립일자는 언제인가?", "reference_substring": "2021년 11월"},
    {"item_id": "f5", "query": "현대자동차의 설립일자는 언제인가?", "reference_substring": "1967년 12월"},
    {"item_id": "f9", "query": "SK스퀘어의 본사 주소는 어디인가?", "reference_substring": "을지로"},
    {"item_id": "l1", "query": "민법상 사람이 성년에 이르는 나이는 몇 세인가?", "reference_substring": "19세"},
    {"item_id": "l3", "query": "개인정보 보호법에서 '고정형 영상정보처리기기'는 어떤 장치를 의미하는가?", "reference_substring": "고정형"},
    {"item_id": "l4", "query": "개인정보 보호법 시행령에서 폐쇄회로 텔레비전(CCTV)은 어떤 장치로 정의되는가?", "reference_substring": "카메라"},
    {"item_id": "l6", "query": "민법상 부재자가 재산관리인을 정하지 않은 경우 누가 재산관리에 필요한 처분을 명하는가?", "reference_substring": "법원"},
    {"item_id": "m3", "query": "ASP(항생제 적정사용 관리) 시범사업에서 의사 인력 충족 기준은 무엇인가?", "reference_substring": "0.5FTE"},
    {"item_id": "m4", "query": "ASP 책임 의사가 감염전문의가 아닌 타과 전문의인 경우 무엇을 이수해야 하는가?", "reference_substring": "전문교육"},
    {"item_id": "m6", "query": "제1급 법정감염병 중 중동호흡기증후군(MERS)의 최종 확진 기관은 어디인가?", "reference_substring": "본청"},
    {"item_id": "m9", "query": "수인성·식품매개감염병 관리지침에서 집단급식소는 몇 명 이상에게 음식물을 공급하는 시설로 정의되는가?", "reference_substring": "50명"},
    {"item_id": "p1", "query": "아동정책기본계획에 포함되어야 할 사항은 아동복지법 몇 조에 근거하는가?", "reference_substring": "제7조"},
    {"item_id": "p2", "query": "빈곤아동 지원 기본계획은 어느 법률에 근거해 수립되는가?", "reference_substring": "아동빈곤예방법"},
    {"item_id": "p4", "query": "산업안전보건법령상 특별교육 대상은 몇 종의 유해·위험 작업에 해당하는가?", "reference_substring": "39종"},
    {"item_id": "p9", "query": "'최신 AI 10대 핵심 기술 가이드북'이 정의하는 지식 증류(Knowledge Distillation)란 무엇인가?", "reference_substring": "작은 모델"},
    {"item_id": "c1", "query": "2021년 국내 이커머스 시장 규모는 얼마인가?", "reference_substring": "186조"},
    {"item_id": "c5", "query": "라이브커머스 이용자가 가장 많이 이용하는 쇼핑 플랫폼은 어디인가?", "reference_substring": "네이버"},
    {"item_id": "c6", "query": "라이브커머스에서 소비자가 가장 많이 구매하는 상품 유형은 무엇인가?", "reference_substring": "식품"},
    {"item_id": "c9", "query": "2021년 이커머스업종 디지털 광고비는 전년 대비 몇 % 증가했는가?", "reference_substring": "35%"},
]


def run(csv_path: str) -> None:
    if CH03_DIR not in sys.path:
        sys.path.insert(0, CH03_DIR)
    try:
        import rag_retriever
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "ch03.rag_retriever를 불러오지 못했습니다. Ollama(`ollama serve`)와 "
            "Qdrant(ch02/docker-compose.yml)가 실행 중인지 확인하세요."
        ) from exc

    write_header = not os.path.exists(csv_path)
    done = set()
    if not write_header:
        with open(csv_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                done.add((row["item_id"], row["mode"]))

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()

        for item in QUERIES:
            item_id, query, ref = item["item_id"], item["query"], item["reference_substring"]

            if (item_id, "think_off") not in done:
                t0 = time.time()
                r = run_think_off(rag_retriever, query, ref)
                elapsed = time.time() - t0
                row = {
                    "item_id": item_id, "mode": "think_off",
                    "query_rewrite_sec": r.query_rewrite_sec, "search_rerank_sec": round(r.search_rerank_sec, 2),
                    "llm_gen_sec": round(r.llm_gen_sec, 2), "total_sec": round(r.total_sec, 2), "correct": r.correct,
                }
                writer.writerow(row)
                f.flush()
                print(f"[{item_id}|think_off] wall={elapsed:.1f}s total={row['total_sec']}s correct={r.correct}", flush=True)

            if (item_id, "think_on") not in done:
                t0 = time.time()
                r = run_think_on(rag_retriever, query, ref)
                elapsed = time.time() - t0
                row = {
                    "item_id": item_id, "mode": "think_on",
                    "query_rewrite_sec": round(r.query_rewrite_sec, 2), "search_rerank_sec": round(r.search_rerank_sec, 2),
                    "llm_gen_sec": round(r.llm_gen_sec, 2), "total_sec": round(r.total_sec, 2), "correct": r.correct,
                }
                writer.writerow(row)
                f.flush()
                print(f"[{item_id}|think_on] wall={elapsed:.1f}s total={row['total_sec']}s correct={r.correct}", flush=True)


def analyze(csv_path: str) -> None:
    import pandas as pd
    from scipy.stats import wilcoxon, binomtest

    df = pd.read_csv(csv_path)

    print("\n=== 경로별 평균 (Table 6 확장판) ===")
    print(df.groupby("mode")[["query_rewrite_sec", "search_rerank_sec", "llm_gen_sec", "total_sec"]].mean().round(2))
    print("\n정답률:")
    print(df.groupby("mode")["correct"].mean())

    wide_total = df.pivot(index="item_id", columns="mode", values="total_sec")
    stat, p_latency = wilcoxon(wide_total["think_off"], wide_total["think_on"], alternative="less")
    print(f"\n=== H3 검정 (Wilcoxon signed-rank, total_sec think_off < think_on, one-sided) ===")
    print(f"think_off 평균={wide_total['think_off'].mean():.1f}s, think_on 평균={wide_total['think_on'].mean():.1f}s, "
          f"n={len(wide_total)}, statistic={stat}, p={p_latency}")

    wide_correct = df.pivot(index="item_id", columns="mode", values="correct")
    b = int(((~wide_correct["think_off"]) & wide_correct["think_on"]).sum())
    c = int((wide_correct["think_off"] & (~wide_correct["think_on"])).sum())
    n_disc = b + c
    p_correct = 1.0 if n_disc == 0 else binomtest(min(b, c), n_disc, 0.5, alternative="two-sided").pvalue
    print(f"\n=== 탐색적 정답률 비교 (McNemar, think_off vs think_on) ===")
    print(f"b(off오답->on정답)={b}, c(off정답->on오답)={c}, n_discordant={n_disc}, p={p_correct}")

    summary = {
        "mean_total_sec": df.groupby("mode")["total_sec"].mean().round(2).to_dict(),
        "accuracy": df.groupby("mode")["correct"].mean().round(4).to_dict(),
        "wilcoxon_latency": {"n": len(wide_total), "statistic": float(stat), "p_value": float(p_latency)},
        "mcnemar_correctness": {"b": b, "c": c, "n_discordant": n_disc, "p_value": p_correct},
    }
    stamp = time.strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(OUT_DIR, f"h3_path_eval_{stamp}_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n요약 저장: {OUT_DIR}/h3_path_eval_{stamp}_summary.json")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, "h3_path_eval_raw.csv")
    run(csv_path)
    analyze(csv_path)
