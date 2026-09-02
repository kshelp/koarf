"""
retrieval_eval.py
==================
논문 4장(실험 및 결과) 4.2절 "검색 성능" 평가 지표 구현체. ch03에서 구현한 RAG
검색 파이프라인(Qdrant 벡터 검색 + Cross-Encoder 재순위화)의 성능을
Recall@k, NDCG@k로 평가하고, 3.3절이 설계한 하이브리드 검색·재순위화 비용
상한을 실측으로 검증한다.

- Recall@k : 정답 문서를 상위 k개 안에 얼마나 잘 포함시켰는지 (0~1, 놓친 게 있는지)
- NDCG@k   : 정답 문서가 "얼마나 상위 순위"에 있는지까지 반영 (0~1, 순위가 얼마나 좋은지)

"벡터 검색만" vs "벡터 검색 + 재순위화"를 같은 golden set으로 비교해
4.3절 <Table 1> Pilot retrieval performance results를 산출하고(아래 3절),
하이브리드 검색·재순위화 소요 시간을 측정해 4.3절 <Table 3> Pilot efficiency
results를 산출한다(아래 7절).

실행 방법: ch03, ch04 어느 디렉터리에서 실행해도 무관하며(rag_retriever
임포트 경로를 CH03_DIR로 절대경로화해두었음), Ollama와 Qdrant(ch02/docker-compose.yml)가
떠 있어야 합니다.
    python3 retrieval_eval.py
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence

import pandas as pd


# ----------------------------------------------------------------------
# 1. Golden Set(정답 데이터) 자료구조
# ----------------------------------------------------------------------

@dataclass
class GoldenQuery:
    """쿼리 하나에 대한 정답 정보를 담는 클래스.

    relevant_docs: {문서ID: 관련성 등급} 형태의 딕셔너리.
        - 이진 평가만 할 거라면 등급을 전부 1로 통일해도 됨.
        - 등급을 세분화하면(예: 핵심정답=2, 참고가능=1) NDCG가 더 정교해짐.
    """
    query_id: str
    query_text: str
    relevant_docs: Dict[str, int]  # {doc_id: relevance_grade (1 이상)}
    domain: str = "general"  # golden_set.json의 domain 필드(4.1절 H2 도메인 그룹 분석용)


GoldenSet = Dict[str, GoldenQuery]


def load_golden_set(data: List[dict]) -> GoldenSet:
    """딕셔너리 리스트를 GoldenSet으로 변환.

    data 예시:
    [
        {
            "query_id": "q1",
            "query_text": "한국어 형태소 분석이란?",
            "relevant_docs": {"doc_23": 2, "doc_47": 1}
        },
        ...
    ]
    """
    golden_set: GoldenSet = {}
    for item in data:
        golden_set[item["query_id"]] = GoldenQuery(
            query_id=item["query_id"],
            query_text=item["query_text"],
            relevant_docs=item["relevant_docs"],
            domain=item.get("domain", "general"),
        )
    return golden_set


def load_golden_set_from_json(path: str) -> GoldenSet:
    """load_golden_set()과 같은 형식의 JSON 파일(예: golden_set.json)을 읽어온다."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return load_golden_set(data)


# ----------------------------------------------------------------------
# 2. 핵심 지표 계산 함수 (순수 함수 - 어떤 vectorstore와도 무관하게 동작)
# ----------------------------------------------------------------------

def _dedupe_preserve_rank(retrieved_ids: Sequence[str]) -> List[str]:
    """같은 doc_id가 여러 순위에 중복으로 나오면 처음 나온 순위만 남긴다.

    ch03처럼 doc_id를 "파일명#페이지" 단위로 성기게 잡으면, 같은 페이지에서 시작하는
    서로 다른 청크 두 개가 검색 결과에 같이 뽑혀 동일 doc_id로 두 번 잡힐 수 있다.
    중복을 그대로 두면 recall/NDCG가 1.0을 넘어버리므로(정답을 두 번 맞춘 것으로 계산),
    스코어링 전에 반드시 중복을 제거해야 한다.
    """
    seen = set()
    deduped = []
    for doc_id in retrieved_ids:
        if doc_id not in seen:
            seen.add(doc_id)
            deduped.append(doc_id)
    return deduped


def recall_at_k(retrieved_ids: Sequence[str], relevant_docs: Dict[str, int], k: int) -> float:
    """Recall@k = (상위 k개 중 정답 개수) / (전체 정답 개수)"""
    if not relevant_docs:
        return 0.0
    top_k = _dedupe_preserve_rank(retrieved_ids)[:k]
    hit_count = sum(1 for doc_id in top_k if doc_id in relevant_docs)
    return hit_count / len(relevant_docs)


def dcg_at_k(retrieved_ids: Sequence[str], relevant_docs: Dict[str, int], k: int) -> float:
    """DCG@k = Σ ( 관련성점수 / log2(순위+1) )
    순위가 뒤로 갈수록(1위 → 5위 → 10위) 로그 함수로 가치를 할인한다.
    """
    dcg = 0.0
    for rank, doc_id in enumerate(_dedupe_preserve_rank(retrieved_ids)[:k], start=1):
        relevance = relevant_docs.get(doc_id, 0)
        if relevance > 0:
            dcg += relevance / math.log2(rank + 1)
    return dcg


def ndcg_at_k(retrieved_ids: Sequence[str], relevant_docs: Dict[str, int], k: int) -> float:
    """NDCG@k = DCG@k / IDCG@k (이상적으로 정렬했을 때 대비 실제 성능 비율)"""
    dcg = dcg_at_k(retrieved_ids, relevant_docs, k)

    # IDCG: 정답 문서를 관련성 높은 순으로 완벽히 정렬했을 때의 DCG (이론상 최댓값)
    ideal_order = sorted(relevant_docs.values(), reverse=True)[:k]
    idcg = sum(rel / math.log2(rank + 1) for rank, rel in enumerate(ideal_order, start=1))

    if idcg == 0:
        return 0.0
    return dcg / idcg


# ----------------------------------------------------------------------
# 3. 검색기(Retriever) 평가 실행부
# ----------------------------------------------------------------------

# retriever_fn(query_text, k) -> 상위 k개 문서ID 리스트 (순위 순서를 반드시 유지해야 함)
RetrieverFn = Callable[[str, int], List[str]]


@dataclass
class QueryResult:
    query_id: str
    k: int
    recall: float
    ndcg: float


def evaluate_retriever(
    retriever_fn: RetrieverFn,
    golden_set: GoldenSet,
    k_values: Sequence[int] = (1, 3, 5, 10),
) -> List[QueryResult]:
    """하나의 검색기(retriever)를 golden set 전체 쿼리에 대해 평가.

    retriever_fn: 쿼리 텍스트와 k를 받아서 상위 k개 문서ID 리스트를 반환하는 함수.
                  (예: LangChain vectorstore.similarity_search를 감싼 wrapper)
    """
    results: List[QueryResult] = []
    max_k = max(k_values)

    for gq in golden_set.values():
        # k가 여러 개여도 검색은 가장 큰 k로 한 번만 수행 (중복 검색 방지)
        retrieved_ids = retriever_fn(gq.query_text, max_k)

        for k in k_values:
            r = recall_at_k(retrieved_ids, gq.relevant_docs, k)
            n = ndcg_at_k(retrieved_ids, gq.relevant_docs, k)
            results.append(QueryResult(query_id=gq.query_id, k=k, recall=r, ndcg=n))

    return results


def summarize_results(results: List[QueryResult]) -> pd.DataFrame:
    """쿼리별 결과를 k값 기준으로 평균 내어 요약 표로 변환."""
    df = pd.DataFrame([r.__dict__ for r in results])
    summary = df.groupby("k")[["recall", "ndcg"]].mean().reset_index()
    summary.columns = ["k", "Recall@k", "NDCG@k"]
    return summary


# ----------------------------------------------------------------------
# 4. 여러 검색기(임베딩 모델) 비교
# ----------------------------------------------------------------------

def compare_retrievers(
    retrievers: Dict[str, RetrieverFn],
    golden_set: GoldenSet,
    k_values: Sequence[int] = (1, 3, 5, 10),
) -> pd.DataFrame:
    """여러 검색기(예: bge-m3 vs bona/bge-m3-korean)를 같은 golden set으로 비교.

    반환값: 모델명 x k값 조합으로 Recall@k, NDCG@k를 정리한 표(DataFrame)
    """
    all_rows = []
    for model_name, retriever_fn in retrievers.items():
        results = evaluate_retriever(retriever_fn, golden_set, k_values)
        summary = summarize_results(results)
        summary.insert(0, "model", model_name)
        all_rows.append(summary)

    comparison_df = pd.concat(all_rows, ignore_index=True)
    return comparison_df.round(4)


def pivot_comparison(comparison_df: pd.DataFrame) -> pd.DataFrame:
    """모델을 행(row), k값을 열(column)로 하는 보기 좋은 피벗 표로 변환.

    예:
                     Recall@1  Recall@3  Recall@5  NDCG@1  NDCG@3  NDCG@5
    bge-m3               0.40      0.70      0.80    0.40    0.55    0.60
    bge-m3-korean        0.50      0.80      0.90    0.50    0.65    0.70
    """
    pivot = comparison_df.pivot(index="model", columns="k", values=["Recall@k", "NDCG@k"])
    # 컬럼명을 "Recall@1", "NDCG@3" 같은 형태로 평탄화
    pivot.columns = [f"{metric.split('@')[0]}@{k}" for metric, k in pivot.columns]
    return pivot.reset_index()


# ----------------------------------------------------------------------
# 5. LangChain VectorStore 연동 헬퍼 (선택 사항)
# ----------------------------------------------------------------------

def make_langchain_retriever_fn(
    vectorstore,
    doc_id_key: str = "doc_id",
    doc_id_fn: Callable[[object], str] | None = None,
) -> RetrieverFn:
    """LangChain VectorStore(FAISS, Chroma, PGVector 등)를 RetrieverFn 형태로 감싸주는 함수.

    doc_id_fn을 주지 않으면 metadata[doc_id_key]를 그대로 문서ID로 쓴다. 이 경우
    벡터스토어에 문서를 넣을 때 metadata에 doc_id_key(기본값 "doc_id")를 미리
    넣어둬야 한다. 반대로 doc_id_fn(doc) -> str 을 주면 metadata를 조합해서
    문서ID를 그때그때 만들 수 있다 (ch03처럼 source/page만 있는 경우 등).

    사용 예:
        retriever_fn = make_langchain_retriever_fn(my_faiss_store)
        retrievers = {"bge-m3": retriever_fn, ...}
    """
    def _retriever_fn(query_text: str, k: int) -> List[str]:
        docs = vectorstore.similarity_search(query_text, k=k)
        if doc_id_fn is not None:
            return [doc_id_fn(doc) for doc in docs]
        return [doc.metadata.get(doc_id_key, "") for doc in docs]

    return _retriever_fn


# ----------------------------------------------------------------------
# 6. ch03 RAG 파이프라인 연동
# ----------------------------------------------------------------------
# ch03/rag_retriever.py는 doc_id를 따로 부여하지 않고 metadata에 source(PDF 경로),
# page(0부터 시작)만 넣어둔다. 그래서 doc_id를 "파일명#p사람이 읽는 페이지번호"
# 형태로 그때그때 만들어서 쓴다. rag_run.py의 build_source_link()와 동일한 규칙이라
# golden_set.json에 적어둔 doc_id를 화면에 뜬 출처 링크와 그대로 대조할 수 있다.

CH03_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ch03"))


def doc_id_for(doc) -> str:
    source = os.path.basename(doc.metadata.get("source", "unknown"))
    page = doc.metadata.get("page", 0)
    return f"{source}#p{page + 1}"


def load_ch03_retrievers(k_values: Sequence[int]) -> Dict[str, RetrieverFn]:
    """ch03의 실제 vectorstore/reranker를 불러와 두 가지 RetrieverFn을 만든다.

    - "vector_only"    : Cross-Encoder 없이 벡터 검색 결과를 그대로 사용
    - "hybrid+rerank"  : rag_run.py의 "/think" 경로가 실제로 쓰는 하이브리드 검색
                         (search_with_queries, 벡터 3코퍼스+BM25 3코퍼스,
                         fetch_k=HYBRID_FETCH_K) + Cross-Encoder 재순위화

    Ollama/Qdrant가 떠 있지 않거나 ch03 모듈 임포트에 실패하면 RuntimeError를 낸다.
    """
    if CH03_DIR not in sys.path:
        sys.path.insert(0, CH03_DIR)

    try:
        import rag_retriever  # ch03/rag_retriever.py
    except Exception as exc:  # noqa: BLE001 - 원인이 다양해서 메시지로 안내
        raise RuntimeError(
            "ch03.rag_retriever를 불러오지 못했습니다. Ollama(`ollama serve`)와 "
            "Qdrant(ch02/docker-compose.yml)가 실행 중인지 확인하세요."
        ) from exc

    max_k = max(k_values)

    def vector_only_fn(query_text: str, k: int) -> List[str]:
        # rag_retriever는 PDF/위키/allganize 컬렉션을 pdf_vectorstore/wiki_vectorstore/
        # allganize_vectorstore로 분리해 보관하므로(단일 vectorstore 없음),
        # search_vector_top1과 동일하게 세 곳에서 k개씩 가져와
        # 벡터 유사도(Qdrant 코사인 유사도, 높을수록 유사) 기준으로 합쳐 정렬한다.
        found = rag_retriever.pdf_vectorstore.similarity_search_with_score(query_text, k=k)
        found += rag_retriever.wiki_vectorstore.similarity_search_with_score(query_text, k=k)
        found += rag_retriever.allganize_vectorstore.similarity_search_with_score(query_text, k=k)
        found.sort(key=lambda pair: pair[1], reverse=True)
        return [doc_id_for(doc) for doc, _ in found[:k]]

    def hybrid_reranked_fn(query_text: str, k: int) -> List[str]:
        # fetch_k는 k와 무관하게 프로덕션과 동일한 HYBRID_FETCH_K로 고정한다.
        # (k별로 fetch_k를 늘리면 프로덕션에서는 절대 발생하지 않는 후보 폭보다
        # 후한 조건을 준 것이 되어, "실제 경로 그대로 재현"이라는 취지에 어긋난다.)
        docs = rag_retriever.search_with_queries(
            query_text, [query_text], fetch_k=rag_retriever.HYBRID_FETCH_K, top_n=k
        )
        return [doc_id_for(doc) for doc in docs]

    return {
        "vector_only": vector_only_fn,
        "hybrid+rerank": hybrid_reranked_fn,
    }


# ----------------------------------------------------------------------
# 7. 하이브리드 검색 비용 계측 (4.3절 <Table 3> Pilot efficiency results)
#    3.3절의 설계 주장 "경로당 후보 수(HYBRID_FETCH_K=5)를 제한하여 재순위화
#    대상 자체를 관리"(6경로 x 5 = 이론적 상한 30개, 중복 제거 전)를 검증하기
#    위해, rag_run.py의 "/think" 경로가 실제로 호출하는 것과 동일한 방식으로
#    search_with_queries()를 호출한다: 재작성 질의 하나만 담은 목록,
#    fetch_k=HYBRID_FETCH_K. stats 인자가 기록하는 검색·재순위화 소요 시간이
#    <Table 3>의 실측치 출처다(3.3절). 질의 재작성(augment_query) 자체는
#    golden set이 단발 질의라 대상에서 제외하며, 그 비용은 4.3절 <Table 4>
#    ("/think" on/off 비교, think_mode_eval.py)에서 별도로 측정한다.
# ----------------------------------------------------------------------

@dataclass
class HybridSearchCost:
    query_id: str
    raw_candidate_count: int         # 중복 제거 전, 6개 경로(PDF/Wiki/allganize 벡터+BM25) 후보 합
    deduped_candidate_count: int     # 중복 제거 후 후보 수
    prefiltered_candidate_count: int  # RRF 1차 필터링 후 실제 Cross-Encoder 재순위화 대상 수
    search_elapsed_sec: float     # 하이브리드 검색(벡터+BM25) 소요 시간
    rerank_elapsed_sec: float     # Cross-Encoder 재순위화 소요 시간

    @property
    def total_elapsed_sec(self) -> float:
        return self.search_elapsed_sec + self.rerank_elapsed_sec


def measure_hybrid_search_cost(golden_set: GoldenSet) -> List[HybridSearchCost]:
    """golden set의 각 질의에 대해 rag_run.py "/think" 경로와 동일한 단일 질의
    하이브리드 검색(ch03.rag_retriever.search_with_queries)을 호출해 후보 수·
    소요 시간을 기록한다.
    """
    if CH03_DIR not in sys.path:
        sys.path.insert(0, CH03_DIR)

    try:
        import rag_retriever
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "ch03.rag_retriever를 불러오지 못했습니다. Ollama(`ollama serve`)와 "
            "Qdrant(ch02/docker-compose.yml)가 실행 중인지 확인하세요."
        ) from exc

    results: List[HybridSearchCost] = []
    for gq in golden_set.values():
        stats: dict = {}
        rag_retriever.search_with_queries(
            gq.query_text,
            [gq.query_text],
            fetch_k=rag_retriever.HYBRID_FETCH_K,
            stats=stats,
        )

        results.append(
            HybridSearchCost(
                query_id=gq.query_id,
                raw_candidate_count=stats["raw_candidate_count"],
                deduped_candidate_count=stats["deduped_candidate_count"],
                prefiltered_candidate_count=stats["prefiltered_candidate_count"],
                search_elapsed_sec=stats["search_elapsed_sec"],
                rerank_elapsed_sec=stats["rerank_elapsed_sec"],
            )
        )

    return results


def summarize_hybrid_search_cost(
    results: List[HybridSearchCost],
) -> pd.DataFrame:
    """쿼리별 계측 결과를 <Table 3>과 같은 형태(검색/재순위화/합계 시간, 후보 수)로 정리한다."""
    df = pd.DataFrame(
        [
            {
                "query_id": r.query_id,
                "raw_candidate_count": r.raw_candidate_count,
                "deduped_candidate_count": r.deduped_candidate_count,
                "prefiltered_candidate_count": r.prefiltered_candidate_count,
                "search_elapsed_sec": round(r.search_elapsed_sec, 2),
                "rerank_elapsed_sec": round(r.rerank_elapsed_sec, 2),
                "total_elapsed_sec": round(r.total_elapsed_sec, 2),
            }
            for r in results
        ]
    )
    mean_row = {
        "query_id": "MEAN",
        "raw_candidate_count": df["raw_candidate_count"].mean(),
        "deduped_candidate_count": df["deduped_candidate_count"].mean(),
        "prefiltered_candidate_count": df["prefiltered_candidate_count"].mean(),
        "search_elapsed_sec": round(df["search_elapsed_sec"].mean(), 2),
        "rerank_elapsed_sec": round(df["rerank_elapsed_sec"].mean(), 2),
        "total_elapsed_sec": round(df["total_elapsed_sec"].mean(), 2),
    }
    return pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)


def check_candidate_upper_bound(
    results: List[HybridSearchCost],
    hybrid_fetch_k: int,
) -> None:
    """3.3절이 말하는 이론적 상한(6개 경로 x HYBRID_FETCH_K)을 실제로 넘지 않는지 확인한다.

    경로 6개 = PDF·위키·allganize 벡터 검색 3개 + BM25(PDF·위키·allganize, 각각 최대 fetch_k) 3개.
    """
    theoretical_max = 6 * hybrid_fetch_k

    for r in results:
        assert r.raw_candidate_count <= theoretical_max, (
            f"{r.query_id}: 재순위화 대상 후보 {r.raw_candidate_count}개가 이론적 상한 "
            f"{theoretical_max}개(경로 6개 x fetch_k {hybrid_fetch_k})를 초과했습니다."
        )

    print(
        f"[통과] 모든 질의의 후보 수가 이론적 상한({theoretical_max}개 = "
        f"경로 6개 x fetch_k {hybrid_fetch_k}) 이내로 유지되었습니다."
    )


# ----------------------------------------------------------------------
# 8. 실행부
#    ch04/golden_set.json: 실제 ingest된 PDF(data/2040_seoul_plan.pdf 등)에서
#    직접 확인한 문장으로 만든 golden set. 쿼리를 더 늘리고 싶으면 같은 형식
#    ({"query_id", "query_text", "relevant_docs": {doc_id: 등급}})으로 추가하면 된다.
# ----------------------------------------------------------------------

if __name__ == "__main__":
    golden_set_path = os.path.join(os.path.dirname(__file__), "golden_set.json")
    golden_set = load_golden_set_from_json(golden_set_path)

    retrievers = load_ch03_retrievers(k_values=(1, 3, 5))

    comparison = compare_retrievers(retrievers, golden_set, k_values=(1, 3, 5))
    print("=== 긴 형식(long format) 결과 ===")
    print(comparison)

    print("\n=== 표 형식(pivot) 결과 - 4.3절 <Table 1>에 대응 ===")
    print(pivot_comparison(comparison))

    print("\n=== 하이브리드 검색 비용 계측 (3.3절 상한 검증, 4.3절 <Table 3> 대응) ===")
    sys.path.insert(0, CH03_DIR)
    import rag_retriever  # noqa: E402  (HYBRID_FETCH_K 상수 참조용)

    hybrid_cost_results = measure_hybrid_search_cost(golden_set)
    print(summarize_hybrid_search_cost(hybrid_cost_results).to_string(index=False))
    check_candidate_upper_bound(
        hybrid_cost_results,
        hybrid_fetch_k=rag_retriever.HYBRID_FETCH_K,
    )