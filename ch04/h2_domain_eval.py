"""
h2_domain_eval.py
==================
논문 4.1절 H2("하이브리드 검색의 성능 및 도메인별 효과", RQ2) 검증 구현체.

retrieval_eval.py가 산출하는 전체 golden set(70건) 집계 결과(<Table 3>)는 이미
확보되어 있으나, H2 후반부("검색 방식과 도메인 유형 간 상호작용효과 ... 특화
도메인에서 크게 나타날 것")를 검증하려면 golden_set.json의 domain 필드
(general/wiki=일반, finance/law/medical/public/commerce=특화)로 결과를 나누어
봐야 한다. retrieval_eval.load_golden_set()이 domain 필드를 무시하고 있었던 것이
이 분석이 지금까지 비어 있던 직접적인 이유였다(retrieval_eval.py에 domain 필드를
추가로 반영함).

각 질의는 vector_only/hybrid+rerank 검색에 한 번씩만 쿼리하며(재순위화 비용이
질의당 수십 초 소요되므로), 결과를 질의 단위로 즉시 CSV에 append해 중간에
중단되어도 이미 처리한 질의의 결과는 보존한다.

실행 방법: ch03, ch04 어디서 실행해도 무관하며 Ollama와 Qdrant가 떠 있어야 한다.
    python3 h2_domain_eval.py
"""

from __future__ import annotations

import csv
import os
import sys
import time
from typing import Dict, List

sys.path.insert(0, os.path.dirname(__file__))

from retrieval_eval import (  # noqa: E402
    load_golden_set_from_json,
    load_ch03_retrievers,
    recall_at_k,
    ndcg_at_k,
)

GENERAL_DOMAINS = {"general", "wiki"}
SPECIALIZED_DOMAINS = {"finance", "law", "medical", "public", "commerce"}

K_VALUES = (1, 3, 5)
OUT_DIR = os.path.join(os.path.dirname(__file__), "results")
CSV_FIELDS = [
    "query_id", "domain", "domain_group", "method",
    "recall@1", "recall@3", "recall@5", "ndcg@1", "ndcg@3", "ndcg@5",
]


def domain_group(domain: str) -> str:
    if domain in GENERAL_DOMAINS:
        return "일반"
    if domain in SPECIALIZED_DOMAINS:
        return "특화"
    return "기타"


def run(csv_path: str) -> None:
    golden_set = load_golden_set_from_json(os.path.join(os.path.dirname(__file__), "golden_set.json"))
    retrievers = load_ch03_retrievers(k_values=K_VALUES)
    max_k = max(K_VALUES)

    write_header = not os.path.exists(csv_path)
    done_ids = set()
    if not write_header:
        with open(csv_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                done_ids.add((row["query_id"], row["method"]))

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()

        for gq in golden_set.values():
            dg = domain_group(gq.domain)
            for method_name, retriever_fn in retrievers.items():
                if (gq.query_id, method_name) in done_ids:
                    continue
                t0 = time.time()
                retrieved = retriever_fn(gq.query_text, max_k)
                elapsed = time.time() - t0
                row = {
                    "query_id": gq.query_id,
                    "domain": gq.domain,
                    "domain_group": dg,
                    "method": method_name,
                }
                for k in K_VALUES:
                    row[f"recall@{k}"] = recall_at_k(retrieved, gq.relevant_docs, k)
                    row[f"ndcg@{k}"] = ndcg_at_k(retrieved, gq.relevant_docs, k)
                writer.writerow(row)
                f.flush()
                print(
                    f"[{gq.query_id}|{gq.domain}|{method_name}] "
                    f"({elapsed:.1f}s) recall@1={row['recall@1']:.2f} ndcg@3={row['ndcg@3']:.3f}",
                    flush=True,
                )


def analyze(csv_path: str) -> None:
    import pandas as pd
    from scipy.stats import binomtest, mannwhitneyu

    df = pd.read_csv(csv_path)

    print("\n=== 전체 집계 (Table 3 검증용) ===")
    print(df.groupby("method")[["recall@1", "recall@3", "recall@5", "ndcg@1", "ndcg@3", "ndcg@5"]].mean().round(4))

    print("\n=== 도메인 그룹별 집계 (일반 vs 특화) ===")
    print(df.groupby(["method", "domain_group"])[["recall@1", "recall@3", "ndcg@1", "ndcg@3"]].mean().round(4))

    print("\n=== 세부 도메인별 집계 ===")
    print(df.groupby(["method", "domain"])[["recall@1", "recall@3", "ndcg@1", "ndcg@3"]].mean().round(4))

    # H2 전반부(주효과): recall@1 hit(정답이 top-1에 하나라도 있는지) 기준 vector_only vs hybrid+rerank McNemar 검정
    wide_hit = df.assign(hit1=df["recall@1"] > 0).pivot(index="query_id", columns="method", values="hit1")
    b = int(((~wide_hit["vector_only"]) & wide_hit["hybrid+rerank"]).sum())
    c = int((wide_hit["vector_only"] & (~wide_hit["hybrid+rerank"])).sum())
    n_discordant = b + c
    p_main = 1.0 if n_discordant == 0 else binomtest(min(b, c), n_discordant, 0.5, alternative="two-sided").pvalue
    print(f"\n=== H2 전반부 주효과 검정 (McNemar, recall@1 hit, vector_only vs hybrid+rerank) ===")
    print(f"b(vector실패->hybrid성공)={b}, c(vector성공->hybrid실패)={c}, n_discordant={n_discordant}, p={p_main}")

    # H2 후반부(상호작용효과): (hybrid - vector) ndcg@3 개선폭을 도메인 그룹별로 비교(Mann-Whitney U)
    wide_ndcg3 = df.pivot(index="query_id", columns="method", values="ndcg@3")
    delta = (wide_ndcg3["hybrid+rerank"] - wide_ndcg3["vector_only"]).rename("delta_ndcg3")
    dg_map = df.drop_duplicates("query_id").set_index("query_id")["domain_group"]
    delta_df = delta.to_frame().join(dg_map)

    general_delta = delta_df.loc[delta_df["domain_group"] == "일반", "delta_ndcg3"]
    specialized_delta = delta_df.loc[delta_df["domain_group"] == "특화", "delta_ndcg3"]
    u_stat, p_interaction = mannwhitneyu(specialized_delta, general_delta, alternative="two-sided")

    print(f"\n=== H2 후반부 상호작용효과 검정 (Mann-Whitney U, NDCG@3 개선폭 특화 vs 일반) ===")
    print(f"일반 도메인 평균 개선폭={general_delta.mean():.4f} (n={len(general_delta)}), "
          f"특화 도메인 평균 개선폭={specialized_delta.mean():.4f} (n={len(specialized_delta)}), "
          f"U={u_stat}, p={p_interaction}")

    summary = {
        "overall": df.groupby("method")[["recall@1", "recall@3", "recall@5", "ndcg@1", "ndcg@3", "ndcg@5"]].mean().round(4).to_dict(),
        "by_domain_group": df.groupby(["method", "domain_group"])[["recall@1", "recall@3", "ndcg@1", "ndcg@3"]].mean().round(4).reset_index().to_dict(orient="records"),
        "by_domain": df.groupby(["method", "domain"])[["recall@1", "recall@3", "ndcg@1", "ndcg@3"]].mean().round(4).reset_index().to_dict(orient="records"),
        "mcnemar_main_effect": {"b": b, "c": c, "n_discordant": n_discordant, "p_value": p_main},
        "mannwhitney_interaction": {
            "general_mean_delta_ndcg3": float(general_delta.mean()),
            "specialized_mean_delta_ndcg3": float(specialized_delta.mean()),
            "n_general": int(len(general_delta)),
            "n_specialized": int(len(specialized_delta)),
            "u_statistic": float(u_stat),
            "p_value": float(p_interaction),
        },
    }
    import json
    stamp = time.strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(OUT_DIR, f"h2_domain_eval_{stamp}_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n요약 저장: {OUT_DIR}/h2_domain_eval_{stamp}_summary.json")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, "h2_domain_eval_raw.csv")
    run(csv_path)
    analyze(csv_path)
