"""
query_rewrite_eval.py
======================
논문 1.2절 RQ1("대화 맥락을 반영한 질의 재작성을 로컬 소형 LLM으로 수행할 때,
인용문 보존과 출력 형식 제어를 위한 설계 장치가 질의 재작성의 안정성과 후속
검색 결과에 어떤 영향을 미치는가")을 검증하는 평가 구현체.

ch03/rag_retriever.py의 augment_query()는 answer_eval.py·think_mode_eval.py
양쪽 모두에서 대화 이력을 빈 리스트([])로 고정 호출하기 때문에("대화 맥락 반영"
자체가 실행되지 않음), RQ1은 지금까지 정량적으로 검증된 적이 없었다. 이 모듈은
그 공백을 메운다.

측정 항목 네 가지:
    1. 출력 형식 제어(안정성) - clean_augmented_query()의 레이블 제거
       안전장치가 실제로 얼마나 자주 개입하는지(raw != cleaned), 그리고
       개입 후에도 형식 위반이 남는지.
    2. 인용문 보존 - 질의에 큰따옴표 인용문이 있을 때, 재작성된 질의에
       원문 그대로 남아 있는지.
    3. 맥락 해소(대화 맥락 반영) - 대명사·생략된 주어가 있는 후속 질문에서,
       재작성된 질의가 이전 대화의 지시 대상(핵심 키워드)을 명시적으로
       포함하는지.
    4. 후속 검색 결과에 미치는 영향 - 같은 벡터 검색(재작성 없음 vs 재작성)
       으로 golden 문서를 top-k 안에 찾아내는지 비교. 재작성 외 변수(검색
       방법)를 통제해, 재작성 자체의 효과만 분리해서 본다.

실행 방법: ch03, ch04 어디서 실행해도 무관하며 Ollama와 Qdrant(ch02/docker-compose.yml)가
떠 있어야 한다.
    python3 query_rewrite_eval.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from typing import List

import pandas as pd

CH03_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ch03"))


# ----------------------------------------------------------------------
# 1. 테스트셋 로딩
# ----------------------------------------------------------------------

def load_testset(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------------------
# 2. 질의 재작성 실행 (raw/cleaned 둘 다 확보 - LLM 호출은 1회만)
# ----------------------------------------------------------------------

def augment_query_with_raw(rag_retriever, messages, query: str) -> tuple[str, str]:
    """augment_query()와 동일한 내부 로직을 그대로 밟되, 안전장치 적용 전(raw)과
    후(cleaned) 출력을 모두 반환한다(augment_query() 자체는 cleaned만 반환)."""
    history = rag_retriever.format_history(messages[:-1])
    raw = rag_retriever.query_augmentation_chain.invoke({"history": history, "query": query})
    cleaned = rag_retriever.clean_augmented_query(raw)
    return raw, cleaned


def build_messages(history_msgs, HumanMessage, AIMessage, query: str):
    messages = []
    for turn in history_msgs:
        if turn["role"] == "user":
            messages.append(HumanMessage(content=turn["content"]))
        else:
            messages.append(AIMessage(content=turn["content"]))
    messages.append(HumanMessage(content=query))
    return messages


# ----------------------------------------------------------------------
# 3. 후속 검색 영향 측정 (재작성 여부만 다르고 검색 방법은 동일하게 통제)
# ----------------------------------------------------------------------

def doc_id_for(doc) -> str:
    source = os.path.basename(doc.metadata.get("source", "unknown"))
    page = doc.metadata.get("page", 0)
    return f"{source}#p{page + 1}"


def vector_search_doc_ids(rag_retriever, query_text: str, k: int) -> List[str]:
    """pdf+wiki 두 코퍼스에서 벡터 유사도 상위 k개 문서ID를 반환한다(재순위화 없음).
    재작성 유무 외 변수를 통제하기 위해 하이브리드·재순위화는 쓰지 않는다."""
    found = rag_retriever.pdf_vectorstore.similarity_search_with_score(query_text, k=k)
    found += rag_retriever.wiki_vectorstore.similarity_search_with_score(query_text, k=k)
    found.sort(key=lambda pair: pair[1], reverse=True)
    return [doc_id_for(doc) for doc, _ in found[:k]]


def hit_at_k(retrieved_ids: List[str], relevant_docs: dict) -> bool:
    return any(doc_id in relevant_docs for doc_id in retrieved_ids)


# ----------------------------------------------------------------------
# 4. 항목별 평가
# ----------------------------------------------------------------------

@dataclass
class QueryRewriteResult:
    item_id: str
    raw_output: str
    augmented_query: str
    safety_net_triggered: bool     # raw != cleaned (안전장치가 실제로 개입했는지)
    format_violation_after: bool   # 안전장치 적용 후에도 위반이 남아있는지
    requires_context: bool
    context_resolved: bool | None      # requires_context=False면 None
    has_quote: bool
    quote_preserved: bool | None       # has_quote=False면 None
    raw_hit_at3: bool              # 재작성 없이 원 질문 그대로 검색했을 때 정답 문서 hit 여부
    augmented_hit_at3: bool        # 재작성된 질의로 검색했을 때 정답 문서 hit 여부


def evaluate_item(rag_retriever, HumanMessage, AIMessage, item: dict, k: int = 3) -> QueryRewriteResult:
    messages = build_messages(item["history"], HumanMessage, AIMessage, item["query"])
    raw, cleaned = augment_query_with_raw(rag_retriever, messages, item["query"])

    # clean_augmented_query()가 실제로 텍스트를 바꿨다면 안전장치가 개입한 것이다.
    safety_net_triggered = raw.strip() != cleaned
    # 안전장치 적용 후에도 여전히 레이블/괄호 패턴이 남아있는지(재적용해 변화가 있으면 위반).
    format_violation_after = rag_retriever.clean_augmented_query(cleaned) != cleaned

    context_resolved = None
    if item["requires_context"]:
        context_resolved = any(kw in cleaned for kw in item["expected_keywords"])

    quote = item.get("quote_to_preserve")
    has_quote = quote is not None
    quote_preserved = (quote in cleaned) if has_quote else None

    raw_ids = vector_search_doc_ids(rag_retriever, item["query"], k=k)
    augmented_ids = vector_search_doc_ids(rag_retriever, cleaned, k=k)

    return QueryRewriteResult(
        item_id=item["item_id"],
        raw_output=raw,
        augmented_query=cleaned,
        safety_net_triggered=safety_net_triggered,
        format_violation_after=format_violation_after,
        requires_context=item["requires_context"],
        context_resolved=context_resolved,
        has_quote=has_quote,
        quote_preserved=quote_preserved,
        raw_hit_at3=hit_at_k(raw_ids, item["relevant_docs"]),
        augmented_hit_at3=hit_at_k(augmented_ids, item["relevant_docs"]),
    )


# ----------------------------------------------------------------------
# 5. 종합
# ----------------------------------------------------------------------

def summarize(results: List[QueryRewriteResult]) -> dict:
    n = len(results)
    context_cases = [r for r in results if r.requires_context]
    quote_cases = [r for r in results if r.has_quote]

    return {
        "n": n,
        "안전장치_개입률(raw!=cleaned)": sum(r.safety_net_triggered for r in results) / n,
        "안전장치_적용후_형식위반률": sum(r.format_violation_after for r in results) / n,
        "맥락_해소_성공률": (
            sum(r.context_resolved for r in context_cases) / len(context_cases)
            if context_cases else None
        ),
        "인용문_보존률": (
            sum(r.quote_preserved for r in quote_cases) / len(quote_cases)
            if quote_cases else None
        ),
        "검색_hit률_재작성전(raw_query)": sum(r.raw_hit_at3 for r in results) / n,
        "검색_hit률_재작성후(augmented_query)": sum(r.augmented_hit_at3 for r in results) / n,
    }


def mcnemar_test(results: List[QueryRewriteResult]) -> dict:
    """raw_hit_at3 vs augmented_hit_at3(대응표본, 이진값)에 대한 McNemar 검정.
    표본이 작으므로(n<25 통상 권장 기준) 카이제곱 근사 대신 정확이항검정을 쓴다."""
    from scipy.stats import binomtest

    # b: raw 실패->augmented 성공(재작성이 유리), c: raw 성공->augmented 실패(재작성이 불리)
    b = sum(1 for r in results if (not r.raw_hit_at3) and r.augmented_hit_at3)
    c = sum(1 for r in results if r.raw_hit_at3 and (not r.augmented_hit_at3))
    n_discordant = b + c
    if n_discordant == 0:
        p_value = 1.0
    else:
        p_value = binomtest(min(b, c), n_discordant, 0.5, alternative="two-sided").pvalue
    return {"b(raw실패->aug성공)": b, "c(raw성공->aug실패)": c, "n_discordant": n_discordant, "p_value": p_value}


def results_to_dataframe(results: List[QueryRewriteResult]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "item_id": r.item_id,
            "requires_context": r.requires_context,
            "context_resolved": r.context_resolved,
            "has_quote": r.has_quote,
            "quote_preserved": r.quote_preserved,
            "safety_net_triggered": r.safety_net_triggered,
            "format_violation_after": r.format_violation_after,
            "raw_hit@3": r.raw_hit_at3,
            "augmented_hit@3": r.augmented_hit_at3,
            "augmented_query": r.augmented_query[:60],
        }
        for r in results
    ])


# ----------------------------------------------------------------------
# 6. 실행부
# ----------------------------------------------------------------------

if __name__ == "__main__":
    if CH03_DIR not in sys.path:
        sys.path.insert(0, CH03_DIR)

    try:
        import rag_retriever  # ch03/rag_retriever.py
        from langchain_core.messages import HumanMessage, AIMessage
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "ch03.rag_retriever를 불러오지 못했습니다. Ollama(`ollama serve`)와 "
            "Qdrant(ch02/docker-compose.yml)가 실행 중인지 확인하세요."
        ) from exc

    testset_path = os.path.join(os.path.dirname(__file__), "query_rewrite_testset.json")
    testset = load_testset(testset_path)

    results = []
    for item in testset:
        print(f"[{item['item_id']}] 재작성 중... (query={item['query'][:40]!r})")
        r = evaluate_item(rag_retriever, HumanMessage, AIMessage, item)
        print(f"    raw      : {r.raw_output!r}")
        print(f"    cleaned  : {r.augmented_query!r}")
        print(f"    안전장치 개입={r.safety_net_triggered}, 형식위반(사후)={r.format_violation_after}, "
              f"맥락해소={r.context_resolved}, 인용보존={r.quote_preserved}, "
              f"hit(raw->aug)={r.raw_hit_at3}->{r.augmented_hit_at3}")
        results.append(r)

    print("\n=== 항목별 결과 ===")
    print(results_to_dataframe(results).to_string(index=False))

    print("\n=== 종합 (RQ1 대응) ===")
    for k, v in summarize(results).items():
        print(f"{k}: {v}")

    print("\n=== McNemar 검정 (raw_hit@3 vs augmented_hit@3, 대응표본) ===")
    for k, v in mcnemar_test(results).items():
        print(f"{k}: {v}")

    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    results_to_dataframe(results).to_csv(os.path.join(out_dir, f"query_rewrite_eval_{stamp}.csv"), index=False)
    with open(os.path.join(out_dir, f"query_rewrite_eval_{stamp}_summary.json"), "w", encoding="utf-8") as f:
        json.dump({**summarize(results), **mcnemar_test(results)}, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {out_dir}/query_rewrite_eval_{stamp}.csv (+_summary.json)")
