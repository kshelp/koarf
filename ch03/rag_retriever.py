# ------------------------------------
# 실행 방법: 
# 추가 문서를 벡터 DB에 적재한 뒤, 아래 명령으로 BM25 캐시를 수동 재생성한다.
# cd ch03
# python3 rag_retriever.py --rebuild-bm25 pdf        # PDF 컬렉션만
# python3 rag_retriever.py --rebuild-bm25 wiki       # 위키백과 컬렉션만
# python3 rag_retriever.py --rebuild-bm25 allganize  # allganize 평가용 컬렉션만
# python3 rag_retriever.py --rebuild-bm25 all        # 셋 다
# ------------------------------------
import os
import pickle
import re
import time
from pathlib import Path

import torch
from konlpy.tag import Okt
from rank_bm25 import BM25Okapi

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import CrossEncoder

from qdrant_utils import fetch_all_documents, get_vectorstore

# --------------------------------------------------
# 1. 임베딩 모델 (문장을 숫자 벡터로 변환)
# --------------------------------------------------
embedding = OllamaEmbeddings(
    model="kure-v1",  # ollama에 pull된 임베딩 모델명 (다른 예: bge-m3, nomic-embed-text)
)

# --------------------------------------------------
# 2. 언어 모델 (질문 다듬기, 답변 생성에 사용)
# --------------------------------------------------
llm = ChatOllama(
    model="exaone3.5:2.4b",  # timHan/llama3.2korean3B4QKM 등으로 교체 가능
    temperature=0,
)

# --------------------------------------------------
# 3. 벡터 저장소 (Qdrant) 연결
# --------------------------------------------------
# 코퍼스별로 컬렉션을 분리해두었다(PDF 문서 vs 위키백과 일반 도메인 코퍼스).
# Qdrant 컬렉션 하나는 코퍼스 하나만 담당하므로, 두 컬렉션을 각각 연결한 뒤
# 검색 시(search_with_queries) 후보를 합쳐서 재순위화한다.
COLLECTION_NAME = "pdf"
WIKI_COLLECTION_NAME = "korean_wikipedia"  # ch03/ingest_wiki.py가 적재하는 컬렉션
ALLGANIZE_COLLECTION_NAME = "allganize_rag_ko"  # ch03/ingest_allganize.py가 적재하는 평가용 벤치마크 컬렉션

pdf_vectorstore = get_vectorstore(COLLECTION_NAME, embedding)
wiki_vectorstore = get_vectorstore(WIKI_COLLECTION_NAME, embedding)
allganize_vectorstore = get_vectorstore(ALLGANIZE_COLLECTION_NAME, embedding)

# --------------------------------------------------
# 4. 검색 파라미터: 최종 전달 문서 수
# --------------------------------------------------
# 로컬 LLM 서버의 메모리가 8GB로 넉넉하지 않아, LLM에는 문서를 조금만 전달해야 한다.
# 그래서 벡터 검색으로는 후보를 넉넉히 가져온 뒤, Cross-Encoder로 재순위화해서
# 실제로 LLM에 넘길 문서 수를 줄인다.
RERANK_TOP_N = 1  # 재순위화 후 최종적으로 LLM에 전달할 문서 수(search_with_queries 기본값)

HYBRID_FETCH_K = 5  # 질의는 하나(재작성된 질문)뿐이지만 경로가 6개(PDF·위키·allganize 벡터 3경로 + PDF·위키·allganize BM25 3경로)

# Cross-Encoder 재순위화는 후보 수(N)에 거의 선형으로 시간이 늘어난다(실측: 후보당 약 3.5초).
# 벡터 검색(코사인 유사도)과 BM25(빈도 기반 점수)는 점수 스케일이 달라 직접 비교할 수 없으므로,
# RRF(Reciprocal Rank Fusion, 채널별 순위만 이용)로 4채널(벡터 3코퍼스 + BM25 통합 1채널) 결과를
# 합친 뒤 상위 HYBRID_PREFILTER_K개만 남기고 나서 Cross-Encoder에 넘긴다.
HYBRID_PREFILTER_K = 15  # RRF 1차 필터링 후 Cross-Encoder에 넘길 후보 수
RRF_K = 60  # RRF 공식의 평활 상수(원 논문 Cormack et al. 2009의 관행값)

SEARCH_VECTOR_TOP_1 = 1  # 저지연 경로(search_vector_top1)에서 코퍼스별로 가져올 벡터 검색 상위 문서 수

# --------------------------------------------------
# 5. BM25 키워드 검색 인덱스: 벡터 검색을 보완하는 하이브리드 검색용 희소 인덱스
# --------------------------------------------------
# 임베딩 기반 벡터 검색은 의미적으로 비슷한 문장은 잘 찾아내지만, "139.1만 가구"처럼
# 정확한 수치·고유명사가 일치해야 하는 질의에서는 오히려 키워드 매칭보다 약할 수
# 있다. BM25는 형태소 단위 토큰의 등장 빈도로 관련도를 계산하므로 이런 경우를 잘
# 잡아내며, 8번 Multi-Query 검색에서 벡터 검색 결과와 합쳐 하이브리드로 사용한다.
_BM25_CACHE_DIR = Path(__file__).parent / ".bm25_cache"
# KoNLPy 기본 힙(1024MB)으로는 위키백과 컬렉션(3만 건 이상)을 한 JVM 세션에서
# 전량 토큰화할 때 네이티브 SIGSEGV로 죽는 사례가 실측으로 확인되어(15분 49초
# 경과 후 크래시) 힙을 2배로 늘렸다. 이 초기화 비용은 BM25 캐시가 있으면 매
# 실행마다 드는 것이 아니라 컬렉션 문서 수가 바뀌어 캐시가 무효화될 때만 든다.
_okt = Okt(max_heap_size=2048)


def _bm25_tokenize(text: str) -> list[str]:
    """한국어는 형태소 단위, 그 외(영어 등)는 공백 기준 토큰으로 분석한다."""
    if re.search(r"[가-힣]", text):
        return _okt.morphs(text)
    return text.split()


def _bm25_cache_path(collection_name: str) -> Path:
    return _BM25_CACHE_DIR / f"{collection_name}.pkl"


def _load_bm25_cache(collection_name: str) -> tuple[BM25Okapi | None, list[Document]]:
    """디스크에 저장된 BM25 캐시만 읽는다. DB 조회나 형태소 분석은 전혀 하지
    않으므로 앱 기동 시 호출해도 즉시 끝난다.

    캐시가 없으면 (None, [])을 반환하며, 이 경우 bm25_search()는 해당
    컬렉션을 조용히 건너뛴다(하이브리드 검색이 벡터 검색만으로 동작).
    수만 건 규모 컬렉션을 형태소 분석하는 비용이 크고(과거 실측 기준 JVM
    힙 부족으로 네이티브 크래시까지 발생한 적이 있어) 서버 기동마다 자동으로
    다시 만들지 않는다 — 캐시 생성/갱신은 아래 rebuild_bm25_cache()를
    `python3 rag_retriever.py --rebuild-bm25 <pdf|wiki|all>`로 수동 실행해야 한다.
    """
    cache_path = _bm25_cache_path(collection_name)
    if not cache_path.exists():
        print(f"[rag_retriever] BM25 캐시 없음: {collection_name} "
              f"(python3 rag_retriever.py --rebuild-bm25 {collection_name} 로 생성)")
        return None, []
    with open(cache_path, "rb") as f:
        _cached_count, cached_bm25, cached_docs = pickle.load(f)
    return cached_bm25, cached_docs


def rebuild_bm25_cache(collection_name: str) -> tuple[BM25Okapi | None, list[Document]]:
    """DB에서 컬렉션 전체를 읽어 형태소 분석 후 BM25 인덱스를 새로 만들고
    캐시에 저장한다. 수만 건 규모에서는 수 분~십수 분이 걸리는 무거운
    작업이므로, 서버 기동 경로가 아니라 이 함수(또는 CLI)를 통해 필요할
    때만 수동으로 실행한다."""
    docs = fetch_all_documents(collection_name)
    _BM25_CACHE_DIR.mkdir(exist_ok=True)
    tokenized = [_bm25_tokenize(doc.page_content) for doc in docs]
    bm25 = BM25Okapi(tokenized) if tokenized else None
    with open(_bm25_cache_path(collection_name), "wb") as f:
        pickle.dump((len(docs), bm25, docs), f)
    return bm25, docs


_pdf_bm25, _pdf_bm25_docs = _load_bm25_cache(COLLECTION_NAME)
_wiki_bm25, _wiki_bm25_docs = _load_bm25_cache(WIKI_COLLECTION_NAME)
_allganize_bm25, _allganize_bm25_docs = _load_bm25_cache(ALLGANIZE_COLLECTION_NAME)


def bm25_search(query: str, k: int) -> list[Document]:
    """PDF·위키백과·allganize 세 코퍼스에서 BM25 점수 상위 k개씩 검색해 합쳐서 반환한다."""
    tokens = _bm25_tokenize(query)
    results = []
    for bm25, docs in (
        (_pdf_bm25, _pdf_bm25_docs),
        (_wiki_bm25, _wiki_bm25_docs),
        (_allganize_bm25, _allganize_bm25_docs),
    ):
        if bm25 is None or not docs:
            continue
        scores = bm25.get_scores(tokens)
        top_idx = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)[:k]
        results.extend(docs[i] for i in top_idx if scores[i] > 0)
    return results


# --------------------------------------------------
# 6. Reranker (Cross-Encoder): 검색된 후보 문서를 질문과의 관련도로 재순위화
# --------------------------------------------------
# 4코어 전체를 쓰도록 늘려봤으나(os.cpu_count()), 이 서버는 임베딩 서빙용
# ollama llama-server가 상시 약 2코어(179~193% CPU)를 점유하고 있어, 4스레드로
# 늘리면 이미 바쁜 코어와 경합해 컨텍스트 스위칭 오버헤드만 늘고 재순위화가
# 오히려 느려짐을 실측으로 확인했다(51.1s -> 55.4s). 실제 여유 코어 수에 맞춰
# 기본값 2를 그대로 유지한다.
torch.set_num_threads(2)

reranker = CrossEncoder(
    "BAAI/bge-reranker-v2-m3",  # 다른 예: cross-encoder/ms-marco-MiniLM-L-6-v2
    max_length=512,
)


def search_vector_top1(query: str) -> list[Document]:
    """질의 재작성·BM25·Cross-Encoder 재순위화 없이, 원 질문 그대로 세 코퍼스에
    벡터 유사도 검색만 수행해 가장 유사한 문서 1건만 반환한다.

    /think 없이 들어온 질문에 대한 저지연 경로용으로, PDF·위키백과·allganize
    각각의 상위 SEARCH_VECTOR_TOP_1건씩만 후보로 가져온 뒤 벡터 유사도(Qdrant
    코사인 유사도, 높을수록 유사)로 비교해 전체 1위만 남긴다.

    세 벡터스토어가 같은 임베딩 모델을 공유하므로, 질의 임베딩을 한 번만
    계산해(embed_query) 재사용한다(similarity_search_with_score_by_vector).
    코퍼스별로 매번 similarity_search_with_score(query, ...)를 부르면 동일한
    질의를 Ollama에 세 번 중복으로 임베딩시키게 된다.
    """
    query_vector = embedding.embed_query(query)
    found = pdf_vectorstore.similarity_search_with_score_by_vector(query_vector, k=SEARCH_VECTOR_TOP_1)
    found += wiki_vectorstore.similarity_search_with_score_by_vector(query_vector, k=SEARCH_VECTOR_TOP_1)
    found += allganize_vectorstore.similarity_search_with_score_by_vector(query_vector, k=SEARCH_VECTOR_TOP_1)
    if not found:
        return []
    found.sort(key=lambda pair: pair[1], reverse=True)  # 코사인 유사도가 높을수록 유사하므로 내림차순
    return [found[0][0]]


# --------------------------------------------------
# 7. Query Augmentation Chain: 대화 맥락을 반영해 질문을 명료하게 다듬는 체인
# --------------------------------------------------
# 질의 재작성과 Multi-Query 확장(8번)은 각각 별도의 단일 목적 프롬프트로
# 처리한다. 인용문이 포함된 질문에서 로컬 소형 모델에게 "재작성과 확장을
# 동시에 하라"는 복합 지시를 내리면 지시를 온전히 따르지 못하고 질문에
# 직접 답하려 드는 경향이 있어, 프롬프트를 분리해야 형식이 안정적으로
# 지켜진다.
#
# 이전 대화를 실제 채팅 메시지(MessagesPlaceholder)로 그대로 넣으면, 모델이
# "대화를 이어가는 챗봇" 역할로 착각해 질문에 직접 답을 해버리는 경향이 있다.
# 그래서 이전 대화는 평문 텍스트([이전 대화] 블록)로만 보여주는 참고 자료로 넘기고,
# 재작성 대상인 현재 질문은 [사용자 질문]으로 명확히 구분해 하나의 human 턴에 담는다.
query_augmentation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "너는 RAG 시스템의 질문 재작성기(Query Rewriter)이다. 너는 사용자와 대화하는 챗봇이 아니라, "
            "[이전 대화]를 참고 자료로만 사용해 [사용자 질문] 하나를 재작성하는 도구다. "
            "질문에 답하거나 대화를 이어가지 말고, 아래 규칙을 반드시 지켜라.\n\n"
            "1. 사용자 질문의 의도를 파악하여 명료한 한 문장의 질문으로 변환하라.\n"
            "2. 대명사(이/저/그것 등)와 생략된 주어·목적어를 [이전 대화]를 참고해 명확한 명사로 표현하라.\n"
            "3. [사용자 질문]에 큰따옴표(\"...\")로 직접 인용된 문구가 있으면, 그 인용문은 "
            "요약하거나 '이 구절', '해당 문구' 같은 대명사로 바꾸지 말고 원문 그대로 재작성된 "
            "질문에 포함시켜라. 인용문 자체가 검색에 필요한 핵심 단서이기 때문이다.\n"
            "4. 절대 질문에 직접 답변하거나 답변 예시를 만들지 마라.\n"
            "5. '답변 예시:', '변환된 질문:' 같은 어떠한 레이블이나 설명, 괄호도 붙이지 말고, "
            "재작성된 질문 문장 하나만 출력하라.\n"
            "6. 예시 1)\n[이전 대화]\n(없음)\n[사용자 질문]\n대한민국 공휴일이 궁금해\n"
            "출력: 대한민국의 공휴일 목록은 무엇인가?\n\n"
            "7. 예시 2)\n[이전 대화]\n사용자: 대한민국 공휴일이 궁금해\n어시스턴트: 대한민국의 공휴일은 신정, 설날...\n"
            "[사용자 질문]\n그거 대체휴일도 있어?\n출력: 대한민국 공휴일의 대체휴일 제도는 무엇인가?\n\n"
            "8. 예시 3)\n[이전 대화]\n(없음)\n[사용자 질문]\n"
            "\"하나님이 세상을 이처럼 사랑하사 독생자를 주셨으니\"라는 말씀이 성경에 어디에 있어?\n"
            "출력: \"하나님이 세상을 이처럼 사랑하사 독생자를 주셨으니\"라는 구절은 성경 어디에 나오는가?",
        ),
        (
            "human",
            "[이전 대화]\n{history}\n\n[사용자 질문]\n{query}",
        ),
    ]
)

query_augmentation_chain = query_augmentation_prompt | llm | StrOutputParser()

# LLM이 규칙을 어기고 "변환된 질문:" 같은 라벨이나 괄호를 덧붙이는 경우에 대비한
# 안전장치. 프롬프트만으로 완전히 통제되지 않는 로컬 소형 모델의 출력을 후처리한다.
_LABEL_PREFIX_RE = re.compile(
    r"^[\s(（]*(변환된\s*질문|재작성된\s*질문|rewritten\s*query|출력)\s*[:：]\s*",
    re.IGNORECASE,
)


def clean_augmented_query(text: str) -> str:
    text = text.strip()
    text = _LABEL_PREFIX_RE.sub("", text).strip()
    if text.startswith("(") and text.endswith(")"):
        text = text[1:-1].strip()
    return text


def format_history(messages: list[BaseMessage], max_turns: int = 6) -> str:
    """대화 이력(BaseMessage 목록)을 프롬프트용 평문으로 변환한다. 이력이 없으면 "(없음)"을 반환한다."""
    turns = []
    for msg in messages[-max_turns:]:
        if isinstance(msg, HumanMessage):
            turns.append(f"사용자: {msg.content}")
        elif isinstance(msg, AIMessage):
            turns.append(f"어시스턴트: {msg.content}")
    return "\n".join(turns) if turns else "(없음)"


def augment_query(messages: list[BaseMessage], query: str) -> str:
    """대화 맥락을 반영해 현재 질문을 명료한 한 문장으로 재작성한다.

    messages는 현재 질문(query)이 마지막 항목으로 이미 포함된 전체 대화 이력이어야 한다
    (재작성 대상과 중복되지 않도록 마지막 항목은 제외하고 이전 대화만 프롬프트에 넣는다).
    """
    history = format_history(messages[:-1])
    raw = query_augmentation_chain.invoke({"history": history, "query": query})
    return clean_augmented_query(raw)


# --------------------------------------------------
# 8. 하이브리드 검색: 벡터 검색과 BM25 키워드 검색을 각각 독립적으로 수행한 뒤,
#    결과를 합치고 중복을 제거해 재순위화한다. 실제 챗봇 흐름(rag_run.py)은
#    재작성된 질문(7번 augment_query) 하나만 담은 목록으로
#    search_with_queries()를 호출한다.
# --------------------------------------------------


# --------------------------------------------------
# 8-1. 질의 유형 자동 분류 — Cross-Encoder 재순위화 이후,
#      질의 유형에 따라 최종 전달 문서 수(top_n)를 세 단계로 조절한다.
#      "/think" 경로(search_with_queries)에서만 사용되며, route_query와
#      마찬가지로 로컬 LLM을 호출하지 않는 규칙 기반 패턴 매칭으로 구현해
#      이미 지연시간이 큰 "/think" 경로에 추가 LLM 호출 비용을 얹지
#      않는다(3.1절 경량성 원칙).
# --------------------------------------------------
QUERY_TYPE_FACTUAL = 1        # 사실 확인형: 단일 사실(수치·고유명사 등) 확인
QUERY_TYPE_EXPLANATORY = 2    # 설명형: 개념·원리·과정에 대한 설명 요구
QUERY_TYPE_COMPARATIVE = 3    # 비교·나열형: 여러 대상의 비교·차이·목록 요구

# 우선순위: 비교·나열형 > 설명형 > 사실 확인형(기본값). 한 질문이 여러
# 패턴에 동시에 걸리면(예: "OO와 OO의 차이를 설명해줘") 더 많은 근거
# 문서가 필요한 쪽(비교·나열형)을 우선한다.
_COMPARATIVE_PATTERNS = re.compile(
    r"(차이점?|비교|대비|장단점|장점.{0,3}단점|공통점|종류|유형|각각|나열|목록|"
    r"몇\s*가지|중\s*어느|중\s*무엇)",
)
_EXPLANATORY_PATTERNS = re.compile(
    r"(설명|알려\s*줘|알려\s*주세요|무엇인가|무엇이야|무엇입니까|어떻게|어떤\s*방식|"
    r"원리|과정|이유|왜)",
)

# 질의 유형별 top_n. 사실 확인형은 근거 문서 1건으로 충분한 경우가 많고,
# 비교·나열형은 여러 항목의 근거를 함께 제시해야 하므로 더 많은 문서가
# 필요하다는 직관을 반영한다.
QUERY_TYPE_TOP_N = {
    QUERY_TYPE_FACTUAL: 1,
    QUERY_TYPE_EXPLANATORY: 2,
    QUERY_TYPE_COMPARATIVE: 3,
}


def classify_query_type(query: str) -> int:
    """질문을 사실 확인형(1)/설명형(2)/비교·나열형(3)으로 분류한다."""
    if _COMPARATIVE_PATTERNS.search(query):
        return QUERY_TYPE_COMPARATIVE
    if _EXPLANATORY_PATTERNS.search(query):
        return QUERY_TYPE_EXPLANATORY
    return QUERY_TYPE_FACTUAL


def search_with_queries(
    rerank_query: str,
    queries: list[str],
    fetch_k: int,
    top_n: int = RERANK_TOP_N,
    stats: dict | None = None,
    adaptive_top_n: bool = False,
):
    """여러 검색 질의로 벡터 검색(세 코퍼스) + BM25 키워드 검색을 각각 수행해 합치고,
    중복 문서를 제거한다. RRF(Reciprocal Rank Fusion)로 1차 필터링해 상위
    HYBRID_PREFILTER_K개만 남긴 뒤, rerank_query와의 관련도로 Cross-Encoder
    재순위화를 수행해 최종 top_n개만 반환한다.

    adaptive_top_n=True이면 Cross-Encoder 재순위화 이후 rerank_query의 질의
    유형을 자동 분류해 top_n을 재결정한다(인자로 받은 top_n은
    이 경우 무시된다).
    """
    seen_keys = set()  # 중복 문서를 걸러내기 위해 이미 채택한 문서의 키를 담는 집합
    candidates = []  # (key, doc) 튜플로 중복 제거를 거친 후보 문서들을 담을 리스트
    rrf_scores: dict[tuple, float] = {}  # 문서 키 -> RRF 누적 점수
    raw_candidate_count = 0  # 중복 제거 전, 모든 질의에 걸쳐 찾은 문서 수 합계(통계용)
    search_start = time.perf_counter()  # 검색 단계 시작 시각 기록(소요 시간 측정용)
    for q in queries:  # 원 질의 + 확장된 질의들을 하나씩 돌면서 검색
        # 세 벡터스토어가 같은 임베딩 모델을 공유하므로 질의 임베딩을 한 번만
        # 계산해 재사용한다. similarity_search(q, ...)를 코퍼스마다 부르면
        # 동일한 질의를 Ollama에 세 번 중복으로 임베딩시키게 된다.
        query_vector = embedding.embed_query(q)
        # 벡터 3경로 + BM25(3코퍼스 통합) 채널. 채널 안에서는 이미 관련도순으로
        # 정렬돼 있으므로, 리스트 내 위치(rank)를 그대로 RRF 순위로 쓴다.
        channels = (
            pdf_vectorstore.similarity_search_by_vector(query_vector, k=fetch_k),
            wiki_vectorstore.similarity_search_by_vector(query_vector, k=fetch_k),
            allganize_vectorstore.similarity_search_by_vector(query_vector, k=fetch_k),
            bm25_search(q, k=fetch_k),
        )
        for channel in channels:  # 채널(벡터 3개 + BM25 1개) 각각에 대해
            raw_candidate_count += len(channel)  # 이번 채널에서 찾은 문서 수를 총합에 더함
            for rank, doc in enumerate(channel, start=1):  # 채널 내 순위(1부터 시작)와 함께 순회
                key = (
                    doc.metadata.get("source"),  # 문서 출처(파일명 등)
                    doc.metadata.get("page"),  # 문서 내 페이지 번호
                    doc.page_content,  # 실제 본문 텍스트
                )  # 세 값을 묶어 "같은 문서인지" 판별하는 키로 사용
                # RRF: 벡터 유사도와 BM25 점수는 스케일이 달라 직접 비교할 수 없으므로,
                # 채널별 순위(rank)만으로 점수를 매겨 합산한다(점수가 클수록 관련도 높음).
                rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
                if key in seen_keys:  # 이미 후보에 들어간 문서와 키가 같다면
                    continue  # 중복이므로 건너뛰고 다음 문서로(점수는 이미 위에서 누적함)
                seen_keys.add(key)  # 처음 보는 문서이므로 키를 기록해두고
                candidates.append((key, doc))  # 후보 리스트에 추가
    search_elapsed = time.perf_counter() - search_start  # 검색 단계에 걸린 총 시간

    if not candidates:  # 모든 질의를 검색했는데도 후보가 하나도 없다면
        if stats is not None:  # 통계를 기록할 딕셔너리가 전달된 경우
            stats.update(
                query_count=len(queries),  # 사용한 질의 개수
                raw_candidate_count=raw_candidate_count,  # 중복 제거 전 문서 총 개수
                deduped_candidate_count=0,  # 후보가 없으므로 0
                prefiltered_candidate_count=0,  # RRF 1차 필터링 후보도 0
                search_elapsed_sec=search_elapsed,  # 검색에 걸린 시간
                rerank_elapsed_sec=0.0,  # 재순위화를 하지 않았으므로 0
                query_type=None,  # 질의 유형 분류도 하지 않았으므로 None
                resolved_top_n=0,  # 반환할 문서가 없으므로 0
            )
        return []  # 빈 리스트를 결과로 반환하고 함수 종료

    # RRF 점수로 1차 정렬 후 상위 HYBRID_PREFILTER_K개만 남긴다. Cross-Encoder는
    # 후보 수(N)에 거의 선형으로 시간이 늘어나므로, 정확도에 크게 기여하지 않을
    # 하위권 후보를 미리 걸러내 재순위화 비용을 줄이는 게 목적이다.
    candidates.sort(key=lambda pair: rrf_scores[pair[0]], reverse=True)
    prefiltered = [doc for _, doc in candidates[: max(HYBRID_PREFILTER_K, top_n)]]

    # -------------------------------
    # Cross-Encoder 재순위화 단계: 시간복잡도가 매우 증가함.
    rerank_start = time.perf_counter()  # 재순위화 단계 시작 시각 기록
    pairs = [(rerank_query, doc.page_content) for doc in prefiltered]  # (기준 질의, 문서 본문) 쌍을 만듦
    scores = reranker.predict(pairs)  # Cross-Encoder 모델로 각 쌍의 관련도 점수를 계산
    reranked = sorted(zip(scores, prefiltered), key=lambda pair: pair[0], reverse=True)  # 점수가 높은 순으로 (점수, 문서) 정렬
    rerank_elapsed = time.perf_counter() - rerank_start  # 재순위화에 걸린 시간

    # 재순위화가 끝난 뒤에야 질의 유형에 따라 top_n을 재결정한다.
    query_type = None  # 기본값: 적응형 top_n을 쓰지 않으면 유형 분류 안 함
    if adaptive_top_n:  # 호출자가 적응형 top_n을 요청했다면
        query_type = classify_query_type(rerank_query)  # 기준 질의가 사실확인형/설명형/비교형 중 무엇인지 분류
        top_n = QUERY_TYPE_TOP_N[query_type]  # 분류 결과에 맞는 top_n으로 교체(인자로 받은 top_n은 무시됨)

    if stats is not None:  # 통계를 기록할 딕셔너리가 전달된 경우
        stats.update(
            query_count=len(queries),  # 사용한 질의 개수
            raw_candidate_count=raw_candidate_count,  # 중복 제거 전 문서 총 개수
            deduped_candidate_count=len(candidates),  # 중복 제거 후 후보 문서 개수
            prefiltered_candidate_count=len(prefiltered),  # RRF 1차 필터링 후 Cross-Encoder에 넘긴 후보 수
            search_elapsed_sec=search_elapsed,  # 검색 단계 소요 시간
            rerank_elapsed_sec=rerank_elapsed,  # 재순위화 단계 소요 시간
            query_type=query_type,  # 최종적으로 분류된 질의 유형(미사용 시 None)
            resolved_top_n=top_n,  # 최종적으로 사용된 top_n 값
        )

    return [doc for _, doc in reranked[:top_n]]  # 재순위화된 문서 중 상위 top_n개만 골라 반환


# --------------------------------------------------
# 9. Router: 질문이 문서 검색(RAG)이 필요한지, 바로 LLM에게 넘겨도 되는지 판단
# (인사말이나 날씨/시간 관련 질문만 direct로 보내고, 그 외에는 모두 rag로 보낸다)
# --------------------------------------------------

# 문장 맨 앞에 나올 때만 인사/잡담으로 인정
_GREETING_PATTERNS = re.compile(
    r"^\s*(안녕|hi|hello|고마워|감사(?:(?![가-힣])|(?=합|드|해|히))|잘가|바이(?![가-힣])|반가워|넌\s*뭐|넌\s*누구|누구세요)",
    re.IGNORECASE,
)

# 문장 어디에 있든 검색 없이 바로 답할 수 있는 주제
_WEATHER_TIME_PATTERNS = re.compile(
    r"(날씨|기온|미세먼지|몇\s*시|날짜|요일)",
    re.IGNORECASE,
)


def route_query(query: str) -> str:
    """질문을 분석해 'direct' 또는 'rag' 중 하나를 반환한다.

    인사말이나 날씨/시간 관련 질문만 'direct'로 보내고, 그 외 모든 질문은 'rag'로 보낸다.
    """
    stripped = query.strip()
    if _GREETING_PATTERNS.match(stripped):
        return "direct"
    if _WEATHER_TIME_PATTERNS.search(stripped):
        return "direct"
    return "rag"


# --------------------------------------------------
# 10. 동작 확인용 테스트 코드 (필요할 때만 주석을 풀어서 사용)
# --------------------------------------------------
# query = "서울의 환경 정책을 설명해줘."
# docs = search_with_queries(query, [query], fetch_k=HYBRID_FETCH_K, adaptive_top_n=True)

# for i, doc in enumerate(docs, start=1):
#     print(f"--- 검색 결과 {i} ---")
#     print(doc.page_content)

# context = "\n\n".join(doc.page_content for doc in docs)

# answer_prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", "사용자의 질문에 대해 아래 context에 기반하여 답변하라.:\n\n{context}"),
#         ("human", "{query}"),
#     ]
# )

# answer_chain = answer_prompt | llm | StrOutputParser()

# answer = answer_chain.invoke({"context": context, "query": query})

# print("--- LLM 답변 ---")
# print(answer)


# --------------------------------------------------
# 11. BM25 캐시 수동 재생성 CLI
#     python3 rag_retriever.py --rebuild-bm25 pdf|wiki|allganize|all
# --------------------------------------------------
if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(description="BM25 캐시를 수동으로 재생성한다.")
    parser.add_argument(
        "--rebuild-bm25",
        choices=["pdf", "wiki", "allganize", "all"],
        required=True,
        help="재생성할 컬렉션 (pdf=pdf, wiki=korean_wikipedia, allganize=allganize_rag_ko, all=전부)",
    )
    args = parser.parse_args()

    targets = []
    if args.rebuild_bm25 in ("pdf", "all"):
        targets.append(("pdf", COLLECTION_NAME))
    if args.rebuild_bm25 in ("wiki", "all"):
        targets.append(("wiki", WIKI_COLLECTION_NAME))
    if args.rebuild_bm25 in ("allganize", "all"):
        targets.append(("allganize", ALLGANIZE_COLLECTION_NAME))

    for label, collection_name in targets:
        print(f"[{label}] {collection_name} BM25 캐시 재생성 시작...")
        t0 = time.time()
        bm25, docs = rebuild_bm25_cache(collection_name)
        print(f"[{label}] 완료: 문서 {len(docs)}건, {time.time() - t0:.1f}초 소요 "
              f"-> {_bm25_cache_path(collection_name)}")
