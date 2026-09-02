# 실행 방법: streamlit run rag_run.py
import os
import re
from urllib.parse import quote

import streamlit as st

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

import rag_retriever  # 문서 검색, 질문 다듬기, 라우팅 판단을 담당하는 모듈

# --------------------------------------------------
# 1. LLM (Ollama)
# --------------------------------------------------
llm = ChatOllama(
    model="exaone3.5:2.4b",  # timHan/llama3.2korean3B4QKM 등으로 교체 가능
    temperature=0,
)


def approx_token_count(text: str) -> int:
    """대략적인 토큰 수를 추정한다.

    llm.get_num_tokens()는 모델 전용 토크나이저가 없으면 Hugging Face Hub에서
    GPT-2 토크나이저를 내려받아 대신 쓰는데, 이 과정에서 부정확하다는 경고와
    인증되지 않은 요청 경고가 함께 뜬다. 여기서는 표시용 대략치만 필요하므로
    네트워크 요청 없이 글자 수 기반으로 간단히 추정한다.
    """
    return max(1, len(text) // 2)


# ch04/static은 ch04/data를 가리키는 심볼릭 링크이고, .streamlit/config.toml에서
# enableStaticServing을 켜두었기 때문에 이 폴더 안의 파일은 Streamlit이
# "app/static/<경로>" URL로 직접 서비스해준다. file:// 링크는 브라우저가
# http로 열린 페이지에서의 이동을 막아버려 클릭해도 반응이 없으므로 쓰지 않는다.
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))


def build_source_link(doc) -> tuple[str, str] | None:
    """문서의 metadata(source, page)로 원본 PDF를 여는 링크를 만든다.

    source가 DATA_DIR(=static 폴더의 실제 위치) 밖에 있으면 Streamlit이 서비스할 수
    없으므로 None을 반환한다. page는 0부터 시작하므로 사람이 읽는 페이지 번호로
    보이도록 1을 더한다.
    """
    
    source = doc.metadata.get("source")
    if not source:
        return None

    abs_path = os.path.abspath(source)
    rel_path = os.path.relpath(abs_path, DATA_DIR)
    if rel_path.startswith(".."):
        return None

    label = os.path.basename(abs_path)
    url = f"app/static/{quote(rel_path.replace(os.sep, '/'))}"

    page = doc.metadata.get("page")
    if page is not None:
        label += f" (p.{page + 1})"
        url += f"#page={page + 1}"

    return label, url


# 시스템 프롬프트로 답변 본문에 출처를 쓰지 말라고 지시해도, 로컬 소형 모델이
# "참고 출처", "출처", "참고문헌" 같은 제목을 답변 끝에 덧붙이는 경우가 있다.
# 이 라벨은 (1) "출처" 한 줄만 있는 제목 형태로도, (2) "**출처**: 파일명 p.N"처럼
# 라벨 뒤에 실제 출처 내용까지 같은 줄에 적는 형태로도 나타나므로 둘 다 잡아낸다.
# 마크다운 제목/강조 기호를 뗀 줄이 라벨로 시작하면(뒤에 아무것도 없거나, 콜론과
# 함께 내용이 이어지거나 모두) 출처 섹션의 시작으로 보고, 그 지점부터는 화면에
# 표시하지 않는다(안전장치, 프롬프트만으로 완전히 통제되지 않음).
_SOURCE_HEADING_RE = re.compile(
    r"^[\s#>*_\-]*(참고\s*출처|출처|참고\s*문헌|참고\s*자료|references?|sources?)"
    r"[\s#*_\-]*(?:[:：].*|)$",
    re.IGNORECASE,
)

# 위 라벨이 줄 맨 앞이 아니라 "...보여줍니다. 출처: **출처** (구체적인 파일명이나
# 페이지 번호가 주어지지 않았으므로...)"처럼 문장 중간에 끼어드는 경우도 실제로
# 관찰됐다. _SOURCE_HEADING_RE는 줄 시작(^)에 앵커링돼 있어 이런 경우를 놓친다.
# 콜론이 라벨 바로 뒤에 붙을 때만 매칭시켜, "이 자료의 출처는 ..."처럼 조사가
# 붙는 정상적인 문장 속 용법과는 구분한다.
_SOURCE_INLINE_RE = re.compile(
    r"[#*_\-]*(참고\s*출처|출처|참고\s*문헌|참고\s*자료|references?|sources?)[#*_\-]*[:：]",
    re.IGNORECASE,
)


def _strip_source_from_line(line: str) -> tuple[str, bool]:
    """한 줄에서 출처 라벨을 찾아 잘라낸다.

    반환값: (화면에 표시해도 안전한 앞부분, 이 지점에서 스트림 전체를 끊어야
    하는지). 라벨이 줄 전체(제목 형태)든 문장 중간(인라인 형태)이든, 일단
    라벨이 나타나면 그 이후는 출처 관련 사족일 가능성이 높다고 보고 스트림을
    끊는다.
    """
    if _SOURCE_HEADING_RE.match(line):
        return "", True
    m = _SOURCE_INLINE_RE.search(line)
    if m:
        return line[:m.start()].rstrip(), True
    return line, False


def strip_source_section(stream):
    """LLM 스트림에서 출처/참고 표기가 나타나면(줄 전체든 문장 중간이든) 그 지점에서 잘라낸다."""
    buffer = ""
    for chunk in stream:
        buffer += chunk.content
        *complete_lines, buffer = buffer.split("\n")
        for line in complete_lines:
            kept, should_stop = _strip_source_from_line(line)
            if kept:
                yield kept + "\n"
            if should_stop:
                return
    if buffer:
        kept, _ = _strip_source_from_line(buffer)
        if kept:
            yield kept


# 질의 재작성(Augmented Query)·하이브리드 검색·재순위화 모듈은 파일럿 실측 기준
# 질의당 약 63초가 소요된다(제4장 <Table 3>). Router가 "rag"로 판단한 질문 중
# 첫 줄에 "/think"가 있으면 기존과 동일하게 이 세 모듈을 모두 거치고, 없으면
# 이 세 모듈을 건너뛰고 원 질문으로 vectorstore를 직접 검색해 가장 유사한
# 문서 1건만 LLM에 전달한다(rag_retriever.search_vector_top1).
_THINK_PREFIX_RE = re.compile(r"^\s*/think\b[ \t]*", re.IGNORECASE)


def strip_think_prefix(text: str) -> tuple[bool, str]:
    """질의 첫 줄에서 '/think' 접두어를 찾아 제거한다.

    "/think 질문" 처럼 같은 줄에 이어 쓰거나, "/think"만 있는 줄 다음 줄에
    질문을 쓰는 두 가지 입력 형태를 모두 지원한다.
    """
    first_line, _, rest = text.partition("\n")
    m = _THINK_PREFIX_RE.match(first_line)
    if not m:
        return False, text
    remainder = first_line[m.end():]
    cleaned = f"{remainder}\n{rest}" if remainder and rest else (remainder or rest)
    return True, cleaned.strip()


# --------------------------------------------------
# 2. Streamlit 화면 설정
# --------------------------------------------------
st.set_page_config(page_title="RAG Chat", page_icon="💬")
st.title("💬 RAG Chat")

# --------------------------------------------------
# 3. 세션(대화 기록) 초기화
# --------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [AIMessage(content="안녕하세요. 무엇을 도와드릴까요?")]

# --------------------------------------------------
# 4. 이전 대화 출력
# --------------------------------------------------
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)

# --------------------------------------------------
# 5. 사용자 입력
# --------------------------------------------------
if raw_prompt := st.chat_input(
    "질문을 입력하세요. (검색이 필요한 질문의 첫 줄에 /think를 붙이면 질의 재작성·"
    "하이브리드 검색·재순위화를 모두 거쳐 답변, 없으면 벡터 검색 top-1 문서만으로 빠르게 답변)"
):
    think_mode, prompt = strip_think_prefix(raw_prompt)

    st.chat_message("user").write(prompt)
    st.session_state.messages.append(HumanMessage(content=prompt))

    with st.spinner("자료 검색 중..."):

        # ----------------------------------------------
        # 6. Router: 문서 검색(RAG)이 필요한 질문인지, 바로 LLM에게 넘겨도 되는
        #    단순 질문인지 판단 (규칙 기반 + 애매하면 LLM 판단, rag_retriever.route_query 내부에서 처리)
        #    /think 여부와 무관하게 라우팅은 항상 먼저 수행한다.
        # ----------------------------------------------
        route = rag_retriever.route_query(prompt)  # "direct" 또는 "rag"

        st.sidebar.markdown("### Route")
        st.sidebar.write(route)

        sources: list[tuple[str, str]] = []  # (라벨, file:// 링크) 목록, rag일 때만 채워짐

        if route == "direct":
            # 단순 질문: 문서 검색 없이 대화 기록만 그대로 LLM에게 전달
            messages = list(st.session_state.messages)

        elif not think_mode:
            # RAG로 판단됐지만 /think가 없는 경우: 질의 재작성·하이브리드 검색·
            # 재순위화 모듈을 모두 건너뛰고, 원 질문 그대로 vectorstore에 벡터
            # 유사도 검색만 수행해 가장 유사한 문서 1건만 LLM에 전달한다(저지연 경로).
            docs = rag_retriever.search_vector_top1(prompt)

            st.sidebar.markdown("### 저지연 경로")
            st.sidebar.write("vector_top1 (질의 재작성·하이브리드 검색·재순위화 생략)")

        else:
            # RAG로 판단됐고 /think가 있는 경우: 기존 전체 파이프라인(질의 재작성
            # → 하이브리드 검색 → 재순위화)을 그대로 수행한다.
            # ----------------------------------------------
            # 7. 질의 재작성: 대화 맥락을 반영해 질문을 명료하게 다듬음
            # ----------------------------------------------
            augmented_query = rag_retriever.augment_query(
                st.session_state.messages, prompt
            )

            st.sidebar.markdown("### 정밀 경로")
            st.sidebar.markdown("### Augmented Query")
            st.sidebar.write(augmented_query)

            # ----------------------------------------------
            # 8. 하이브리드 검색: 재작성된 질문 하나로 벡터 검색과 BM25 키워드 검색을 각각
            #    수행하고, 결과를 합쳐 중복을 제거한 뒤 Cross-Encoder로 재순위화.
            #    재순위화 이후에는 질의 유형(사실 확인형/설명형/
            #    비교·나열형)을 자동 분류해 최종 top_n을 결정한다(adaptive_top_n=True).
            # ----------------------------------------------
            search_stats: dict = {}
            docs = rag_retriever.search_with_queries(
                augmented_query,
                [augmented_query],
                fetch_k=rag_retriever.HYBRID_FETCH_K,
                adaptive_top_n=True,
                stats=search_stats,
            )

            query_type_label = {
                rag_retriever.QUERY_TYPE_FACTUAL: "사실 확인형",
                rag_retriever.QUERY_TYPE_EXPLANATORY: "설명형",
                rag_retriever.QUERY_TYPE_COMPARATIVE: "비교·나열형",
            }.get(search_stats.get("query_type"), "알 수 없음")
            st.sidebar.markdown("### Query Type")
            st.sidebar.write(
                f"{query_type_label} -> top_n={search_stats.get('resolved_top_n')}"
            )

        if route != "direct":
            # 문서 앞에 출처(파일명, 페이지)와 대략적인 토큰 수를 표기하고, 문서 사이는 빈 줄로 구분
            context = "\n\n".join(
                f"*** 출처: {os.path.basename(doc.metadata.get('source', '알 수 없음'))}"
                f" p.{doc.metadata.get('page', 0) + 1}, tokens≈{approx_token_count(doc.page_content)}"
                f"\n{doc.page_content}"
                for doc in docs
            )

            # 문서별 file:// 링크를 만들어 사이드바와 답변 아래에서 클릭해 열 수 있게 함
            seen_links = set()
            for doc in docs:
                link = build_source_link(doc)
                if link and link not in seen_links:
                    seen_links.add(link)
                    sources.append(link)

            st.sidebar.markdown("### Retrieved Documents")
            st.sidebar.write(context)
            if sources:
                st.sidebar.markdown("### References")
                for label, url in sources:
                    st.sidebar.markdown(f"- [{label}]({url})")

            # ----------------------------------------------
            # 9. Prompt: AI에게 실제로 보낼 메시지 구성
            # ----------------------------------------------
            messages = [SystemMessage(content=f"""
                        당신은 RAG 전문가입니다.

                        아래 Context만 사용하여 답변하세요.

                        Context
                        --------
                        {context}

                        규칙
                        1. Context에 있는 내용만 이용한다.
                        2. 모르면 모른다고 답한다.
                        3. 추측하지 않는다.
                        4. 한국어로 답변한다.
                        5. 참고 내용이 중복되면 Context의 내용을 우선하여 답한다.
                        6. 출처(파일명, 페이지, 링크)는 화면에 별도로 표시하므로, 답변 본문
                           끝에 "출처", "참고 출처", "참고문헌", "참고 자료" 같은 제목의
                           섹션을 만들지 않는다. 어떤 이름으로도 출처나 링크를 답변에
                           언급하거나 나열하지 않는다. 이는 문장 끝에 별도 섹션으로 붙이는
                           경우뿐 아니라, 문장 중간에 "출처: ..."처럼 끼워 넣거나 파일명·
                           페이지 정보가 없다는 점을 언급하는 것도 포함한다 — "출처"라는
                           단어 자체를 답변 어디에도 쓰지 않는다.
                    """)]
            messages.extend(
                st.session_state.messages
            )  # 이전 대화 기록을 이어 붙여 맥락 유지

        stream = llm.stream(messages)

    # ----------------------------------------------
    # 10. Streaming: 답변을 실시간으로 받아 화면에 출력 (direct/rag 공통)
    # ----------------------------------------------
    with st.spinner("답변 생성 중..."):
        assistant_box = st.chat_message("assistant")
        answer = assistant_box.write_stream(strip_source_section(stream))
        if sources:
            links = "  \n".join(f"- [{label}]({url})" for label, url in sources)
            assistant_box.markdown(f"**출처**  \n{links}")

    st.session_state.messages.append(AIMessage(content=answer))
