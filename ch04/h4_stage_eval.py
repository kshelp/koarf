"""
h4_stage_eval.py
=================
논문 4.1절 H4("경로 내부 단계별 지연시간 비교", RQ4) 검증 구현체.

기존 retrieval_eval.measure_hybrid_search_cost()는 golden set 70건 중 10건
표본에 대해서만 실행되었고(4.3절 <Table 5>), 결과를 메모리에만 쌓아 중간에
끊기면 진행 상황이 사라지는 구조였다. 이 모듈은 같은 계측 로직
(rag_retriever.search_with_queries의 stats 인자)을 golden set 70건 전체로
확장하고, 질의 단위로 즉시 CSV에 append하여 중단되어도 이미 처리한 질의의
결과는 보존한다. 아울러 하이브리드 검색 단계와 Cross-Encoder 재순위화 단계의
소요 시간을 같은 질의에 대한 대응표본으로 보고, Wilcoxon 부호순위 검정으로
H4("재순위화 단계의 평균 처리시간이 하이브리드 검색 단계보다 유의하게 클
것")를 통계적으로 검증한다.

실행 방법: ch03, ch04 어디서 실행해도 무관하며 Ollama와 Qdrant가 떠 있어야 한다.
    python3 h4_stage_eval.py
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from typing import List

sys.path.insert(0, os.path.dirname(__file__))

from retrieval_eval import load_golden_set_from_json  # noqa: E402

CH03_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ch03"))
OUT_DIR = os.path.join(os.path.dirname(__file__), "results")
CSV_FIELDS = [
    "query_id", "domain", "raw_candidate_count", "deduped_candidate_count",
    "prefiltered_candidate_count", "search_elapsed_sec", "rerank_elapsed_sec", "total_elapsed_sec",
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

    golden_set = load_golden_set_from_json(os.path.join(os.path.dirname(__file__), "golden_set.json"))

    write_header = not os.path.exists(csv_path)
    done = set()
    if not write_header:
        with open(csv_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                done.add(row["query_id"])

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()

        for gq in golden_set.values():
            if gq.query_id in done:
                continue
            stats: dict = {}
            t0 = time.time()
            rag_retriever.search_with_queries(
                gq.query_text, [gq.query_text], fetch_k=rag_retriever.HYBRID_FETCH_K, stats=stats,
            )
            wall = time.time() - t0
            row = {
                "query_id": gq.query_id,
                "domain": gq.domain,
                "raw_candidate_count": stats["raw_candidate_count"],
                "deduped_candidate_count": stats["deduped_candidate_count"],
                "prefiltered_candidate_count": stats["prefiltered_candidate_count"],
                "search_elapsed_sec": round(stats["search_elapsed_sec"], 3),
                "rerank_elapsed_sec": round(stats["rerank_elapsed_sec"], 3),
                "total_elapsed_sec": round(stats["search_elapsed_sec"] + stats["rerank_elapsed_sec"], 3),
            }
            writer.writerow(row)
            f.flush()
            print(
                f"[{gq.query_id}|{gq.domain}] wall={wall:.1f}s "
                f"search={row['search_elapsed_sec']}s rerank={row['rerank_elapsed_sec']}s "
                f"candidates(raw/prefiltered)={row['raw_candidate_count']}/{row['prefiltered_candidate_count']}",
                flush=True,
            )


def analyze(csv_path: str) -> None:
    import pandas as pd
    from scipy.stats import wilcoxon

    df = pd.read_csv(csv_path)

    print("\n=== 전체 집계 (n={}) ===".format(len(df)))
    print(df[["raw_candidate_count", "prefiltered_candidate_count",
              "search_elapsed_sec", "rerank_elapsed_sec", "total_elapsed_sec"]].mean().round(3))

    print("\n=== 도메인별 집계 ===")
    print(df.groupby("domain")[["search_elapsed_sec", "rerank_elapsed_sec", "total_elapsed_sec"]].mean().round(2))

    stat, p = wilcoxon(df["rerank_elapsed_sec"], df["search_elapsed_sec"], alternative="greater")
    print(f"\n=== H4 검정 (Wilcoxon signed-rank, rerank_elapsed_sec > search_elapsed_sec, one-sided) ===")
    print(f"search 평균={df['search_elapsed_sec'].mean():.2f}s, rerank 평균={df['rerank_elapsed_sec'].mean():.2f}s, "
          f"n={len(df)}, statistic={stat}, p={p}")

    summary = {
        "n": len(df),
        "mean_search_elapsed_sec": round(df["search_elapsed_sec"].mean(), 3),
        "mean_rerank_elapsed_sec": round(df["rerank_elapsed_sec"].mean(), 3),
        "mean_total_elapsed_sec": round(df["total_elapsed_sec"].mean(), 3),
        "mean_raw_candidate_count": round(df["raw_candidate_count"].mean(), 2),
        "mean_prefiltered_candidate_count": round(df["prefiltered_candidate_count"].mean(), 2),
        "wilcoxon": {"statistic": float(stat), "p_value": float(p)},
    }
    stamp = time.strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(OUT_DIR, f"h4_stage_eval_{stamp}_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n요약 저장: {OUT_DIR}/h4_stage_eval_{stamp}_summary.json")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, "h4_stage_eval_raw.csv")
    run(csv_path)
    analyze(csv_path)
