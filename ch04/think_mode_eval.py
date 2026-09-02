"""
think_mode_eval.py
===================
논문 4장(실험 및 결과) 4.3절 "적응형 경로 비교('/think')" 평가 지표 구현체.
이 모듈이 산출하는 값이 <Table 4>(H3: 경로 비교, RQ3)와 <Table 5>(H4: 정밀
경로 내부 단계별 지연시간, RQ4)를 동일한 실행에서 함께 산출하는 실제 출처다.

rag_run.py가 실제로 구현한 두 경로를 그대로 재현해 종단(E2E) 응답 시간을
단계별(질의 재작성/하이브리드 검색/Cross-Encoder 재순위화/LLM 생성)로 나눠
측정하고, 두 경로 모두 golden set 참조 정답을 반환하는지 비교한다.
하이브리드 검색과 재순위화 시간을 별도 필드로 남기는 이유는, search_with_queries가
이미 두 단계를 stats에 개별 기록하기 때문에(H4용으로 별도 질의 세트를 다시 돌릴
필요 없이) think_on 실행 자체에서 H4 비교도 함께 성립하기 때문이다.

- "think_on"  : augment_query(질의 재작성) -> search_with_queries(하이브리드
                검색 + Cross-Encoder 재순위화, fetch_k=HYBRID_FETCH_K,
                adaptive_top_n=True) -> LLM 생성
- "think_off" : search_vector_top1(벡터 top-1, 질의 재작성·BM25·재순위화 생략)
                -> LLM 생성

동일 질의를 반복 측정하는 이유는 4.2절이 "지연시간·처리량은 동일 질의 3회
반복 측정 평균으로 보고한다"고 명시하기 때문이다.

실행 방법: ch03, ch04 어디서 실행해도 무관하며 Ollama와 Qdrant(ch02/docker-compose.yml)가
떠 있어야 한다.
    python3 think_mode_eval.py
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from typing import List

import pandas as pd

CH03_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ch03"))

# 4.3절 <Table 4>가 실제로 측정에 사용한 golden set 질의(q2/a2)와 정답.
DEFAULT_QUERY = "서울의 1인 가구 수는 2020년 기준 몇 만 가구인가?"
DEFAULT_REFERENCE_SUBSTRING = "139.1만"


def _build_context(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def _generate_answer(rag_retriever, question: str, docs) -> tuple[str, float]:
    """answer_eval.generate_answer와 동일한 프롬프트 형식으로 LLM 생성 시간을 잰다."""
    context = _build_context(docs)
    prompt = (
        "다음 컨텍스트만 근거로 질문에 간결하게 답하세요.\n\n"
        f"컨텍스트:\n{context}\n\n질문: {question}"
    )
    start = time.perf_counter()
    answer = rag_retriever.llm.invoke(prompt).content
    elapsed = time.perf_counter() - start
    return answer, elapsed


@dataclass
class ThinkModeRun:
    mode: str  # "think_on" | "think_off"
    run_index: int
    query_rewrite_sec: float | None  # think_off는 재작성을 생략하므로 None
    hybrid_search_sec: float  # think_off는 하이브리드가 아닌 벡터 top-1 검색 시간
    rerank_sec: float  # think_off는 재순위화를 생략하므로 0.0
    llm_gen_sec: float
    answer: str
    correct: bool

    @property
    def total_sec(self) -> float:
        return (self.query_rewrite_sec or 0.0) + self.hybrid_search_sec + self.rerank_sec + self.llm_gen_sec


def run_think_on(rag_retriever, query: str, reference_substring: str) -> ThinkModeRun:
    """rag_run.py의 "/think" 있는 경로: 질의 재작성 -> 하이브리드 검색 -> 재순위화 -> LLM 생성.

    search_with_queries가 stats에 하이브리드 검색(search_elapsed_sec)과
    Cross-Encoder 재순위화(rerank_elapsed_sec)를 이미 별도로 기록하므로,
    H3(경로 비교)와 H4(단계별 지연시간 비교)를 동일한 실행에서 함께 측정한다.
    """
    rewrite_start = time.perf_counter()
    augmented_query = rag_retriever.augment_query([], query)
    rewrite_elapsed = time.perf_counter() - rewrite_start

    stats: dict = {}
    docs = rag_retriever.search_with_queries(
        augmented_query,
        [augmented_query],
        fetch_k=rag_retriever.HYBRID_FETCH_K,
        adaptive_top_n=True,
        stats=stats,
    )
    hybrid_search_elapsed = stats.get("search_elapsed_sec", 0.0)
    rerank_elapsed = stats.get("rerank_elapsed_sec", 0.0)

    answer, gen_elapsed = _generate_answer(rag_retriever, query, docs)

    return ThinkModeRun(
        mode="think_on",
        run_index=-1,
        query_rewrite_sec=rewrite_elapsed,
        hybrid_search_sec=hybrid_search_elapsed,
        rerank_sec=rerank_elapsed,
        llm_gen_sec=gen_elapsed,
        answer=answer,
        correct=reference_substring in answer,
    )


def run_think_off(rag_retriever, query: str, reference_substring: str) -> ThinkModeRun:
    """rag_run.py의 "/think" 없는 경로: 벡터 top-1 -> LLM 생성 (질의 재작성·하이브리드·재순위화 생략)."""
    search_start = time.perf_counter()
    docs = rag_retriever.search_vector_top1(query)
    search_elapsed = time.perf_counter() - search_start

    answer, gen_elapsed = _generate_answer(rag_retriever, query, docs)

    return ThinkModeRun(
        mode="think_off",
        run_index=-1,
        query_rewrite_sec=None,
        hybrid_search_sec=search_elapsed,
        rerank_sec=0.0,
        llm_gen_sec=gen_elapsed,
        answer=answer,
        correct=reference_substring in answer,
    )


def measure_think_mode_latency(
    rag_retriever,
    query: str = DEFAULT_QUERY,
    reference_substring: str = DEFAULT_REFERENCE_SUBSTRING,
    repeats: int = 3,
) -> List[ThinkModeRun]:
    """동일 질의를 "/think" on/off 각각 repeats회 반복 측정한다(4.2절 반복 측정 규칙)."""
    results: List[ThinkModeRun] = []
    for i in range(repeats):
        run = run_think_off(rag_retriever, query, reference_substring)
        run.run_index = i
        results.append(run)
    for i in range(repeats):
        run = run_think_on(rag_retriever, query, reference_substring)
        run.run_index = i
        results.append(run)
    return results


def summarize_think_mode_latency(results: List[ThinkModeRun]) -> pd.DataFrame:
    """<Table 4>와 같은 형태(경로별 평균 총 소요시간·단계별 소요시간·정답 여부)로 요약한다."""
    df = pd.DataFrame(
        [
            {
                "mode": r.mode,
                "run_index": r.run_index,
                "query_rewrite_sec": r.query_rewrite_sec,
                "hybrid_search_sec": round(r.hybrid_search_sec, 2),
                "rerank_sec": round(r.rerank_sec, 2),
                "llm_gen_sec": round(r.llm_gen_sec, 2),
                "total_sec": round(r.total_sec, 2),
                "correct": r.correct,
            }
            for r in results
        ]
    )
    summary = (
        df.groupby("mode")[["query_rewrite_sec", "hybrid_search_sec", "rerank_sec", "llm_gen_sec", "total_sec"]]
        .mean()
        .round(2)
    )
    summary["all_correct"] = df.groupby("mode")["correct"].all()
    return summary.reset_index()


if __name__ == "__main__":
    if CH03_DIR not in sys.path:
        sys.path.insert(0, CH03_DIR)

    try:
        import rag_retriever  # ch03/rag_retriever.py
    except Exception as exc:  # noqa: BLE001 - 원인이 다양해서 메시지로 안내
        raise RuntimeError(
            "ch03.rag_retriever를 불러오지 못했습니다. Ollama(`ollama serve`)와 "
            "Qdrant(ch02/docker-compose.yml)가 실행 중인지 확인하세요."
        ) from exc

    print(f"질의: {DEFAULT_QUERY}")
    print(f"참조 정답 포함 여부 판정 기준: {DEFAULT_REFERENCE_SUBSTRING!r} 부분 문자열 포함\n")

    run_results = measure_think_mode_latency(rag_retriever)

    print("=== 실행별 상세 결과 ===")
    for r in run_results:
        print(
            f"[{r.mode} #{r.run_index}] total={r.total_sec:.1f}s "
            f"(rewrite={r.query_rewrite_sec}, hybrid_search={r.hybrid_search_sec:.1f}s, "
            f"rerank={r.rerank_sec:.1f}s, gen={r.llm_gen_sec:.1f}s) correct={r.correct} answer={r.answer[:80]!r}"
        )

    print("\n=== 요약 (4.3절 <Table 4> 대응) ===")
    print(summarize_think_mode_latency(run_results).to_string(index=False))
