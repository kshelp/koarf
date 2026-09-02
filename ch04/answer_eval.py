"""
answer_eval.py
===============
논문 4장(실험 및 결과) 4.2절 "답변 성능" 평가 지표 구현체. 이 모듈이 산출하는
값이 4.3절 <Table 2> Pilot answer-quality results의 실제 출처다.

참조 기반(reference-based) 지표 - 정답(reference)이 있어야 계산 가능:
    - EM   : 정규화된 정답과 응답 문자열의 완전 일치 여부
    - F1   : 정답-응답 간 형태소(morpheme) 단위 토큰 중첩 F1
    - ROUGE-L : 정답-응답 간 형태소 단위 최장공통부분수열(LCS) 기반 F1
    - BERTScore(proxy) : 정답-응답 문장 임베딩 코사인 유사도
      (4.2절에서 정의한 "참조 기반(F1, ROUGE-L, BERTScore)" 지표 중
      BERTScore를 토큰 단위 최대유사도 대신 문장 임베딩 코사인 유사도로
      단순화한 근사치다.)

참조 독립(reference-free) 지표 - 정답 없이 (질문, 컨텍스트, 응답)만으로 계산:
    - Faithfulness     : 응답의 주장(claim)들이 컨텍스트로 뒷받침되는 비율
    - Answer Relevancy : 응답에서 역생성한 가상 질의와 원 질의 간 임베딩 유사도

ch03(rag_retriever.py)의 실제 임베딩/LLM을 그대로 재사용하도록 설계했다.
실행 방법: ch03, ch04 어디서 실행해도 무관하며 Ollama가 떠 있어야 한다.
    python3 answer_eval.py
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from typing import Callable, List, Sequence

import numpy as np
import pandas as pd
from konlpy.tag import Okt

EmbedFn = Callable[[str], Sequence[float]]

# ----------------------------------------------------------------------
# 1. 텍스트 정규화 & 형태소 분석
# ----------------------------------------------------------------------

_okt = Okt()
_PUNCT_RE = re.compile(r"[^\w\s]")
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """EM 비교용 정규화: 소문자화, 문장부호 제거, 공백 정규화."""
    text = text.lower().strip()
    text = _PUNCT_RE.sub("", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def morphs(text: str) -> List[str]:
    """한국어는 형태소 단위, 그 외(영어 등)는 공백 기준 토큰으로 분석한다.

    조사·어미의 활용형이 다양한 한국어 특성상 어절 단위 비교는 표층형
    차이만으로 과소평가되므로, F1/ROUGE-L 모두 형태소 단위로 계산한다.
    """
    if re.search(r"[가-힣]", text):
        return _okt.morphs(text)
    return text.split()


# ----------------------------------------------------------------------
# 2. 참조 기반 지표
# ----------------------------------------------------------------------

def exact_match(prediction: str, reference: str) -> float:
    """EM = 정규화된 정답과 응답 문자열이 완전히 일치하면 1.0, 아니면 0.0"""
    return float(normalize_text(prediction) == normalize_text(reference))


def f1_score(prediction: str, reference: str) -> float:
    """F1 = 정답-응답 간 형태소 단위 토큰 중첩 기준 정밀도/재현율의 조화평균"""
    pred_tokens = morphs(normalize_text(prediction))
    ref_tokens = morphs(normalize_text(reference))
    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)

    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def _lcs_len(a: Sequence[str], b: Sequence[str]) -> int:
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


def rouge_l(prediction: str, reference: str) -> float:
    """ROUGE-L = 정답-응답 형태소 시퀀스의 최장공통부분수열(LCS) 기반 F1"""
    pred_tokens = morphs(normalize_text(prediction))
    ref_tokens = morphs(normalize_text(reference))
    if not pred_tokens or not ref_tokens:
        return 0.0

    lcs = _lcs_len(pred_tokens, ref_tokens)
    if lcs == 0:
        return 0.0

    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    a_arr, b_arr = np.array(a), np.array(b)
    denom = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
    if denom == 0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / denom)


def bertscore_proxy(prediction: str, reference: str, embed_fn: EmbedFn) -> float:
    """BERTScore(proxy) = 정답-응답 문장 임베딩 코사인 유사도"""
    return cosine_similarity(embed_fn(prediction), embed_fn(reference))


# ----------------------------------------------------------------------
# 3. 참조 독립 지표 (LLM 판정자 기반)
# ----------------------------------------------------------------------

def _extract_json_array(text: str) -> str:
    """LLM 출력에서 ```json 코드펜스 등을 걷어내고 JSON 배열 부분만 뽑아낸다."""
    text = text.strip()
    text = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1:
        raise ValueError(f"응답에서 JSON 배열을 찾지 못했습니다: {text!r}")
    return text[start : end + 1]


_FAITHFULNESS_PROMPT = """당신은 RAG 시스템의 답변을 채점하는 평가자입니다.
아래 컨텍스트와, 그 컨텍스트를 근거로 생성된 답변이 주어집니다.
답변을 사실적 주장(claim) 단위로 나누고, 각 주장이 컨텍스트 내용만으로
뒷받침되는지("supported": true/false) 판단하세요.

컨텍스트:
{context}

답변:
{answer}

다른 설명 없이 아래 형식의 JSON 배열만 출력하세요:
[{{"claim": "주장 내용", "supported": true}}, {{"claim": "주장 내용", "supported": false}}]
"""

_HYPOTHETICAL_QUESTIONS_PROMPT = """다음은 어떤 질문에 대한 답변입니다. 이 답변만 보고,
이 답변이 나올 법한 질문을 서로 다른 표현으로 {n}개 만드세요.

답변:
{answer}

각 줄에 질문 하나씩, 번호나 부가 설명 없이 질문 문장만 출력하세요.
"""


@dataclass
class AnswerJudge:
    """Faithfulness / Answer Relevancy를 계산하는 LLM 판정자.

    judge_llm은 평가 대상 백본 모델(EXAONE 3.5 2.4B, 4.2절)과의 이해충돌을
    피하기 위해 답변 생성에 직접 사용되지 않는 별도 계열의 모델을 넘겨야
    한다(아래 JUDGE_MODEL 참고).
    """

    judge_llm: object  # .invoke(prompt) -> AIMessage(.content) 인터페이스를 따르는 LLM
    embed_fn: EmbedFn

    def faithfulness(self, answer: str, context: str) -> float:
        prompt = _FAITHFULNESS_PROMPT.format(context=context, answer=answer)
        raw = self.judge_llm.invoke(prompt).content
        try:
            claims = json.loads(_extract_json_array(raw))
        except (ValueError, json.JSONDecodeError):
            return 0.0
        if not claims:
            return 0.0
        supported = sum(1 for c in claims if c.get("supported"))
        return supported / len(claims)

    def answer_relevancy(self, answer: str, question: str, n: int = 3) -> float:
        prompt = _HYPOTHETICAL_QUESTIONS_PROMPT.format(answer=answer, n=n)
        raw = self.judge_llm.invoke(prompt).content
        hypothetical_qs = [line.strip("- ").strip() for line in raw.splitlines() if line.strip()]
        if not hypothetical_qs:
            return 0.0

        question_vec = self.embed_fn(question)
        sims = [cosine_similarity(question_vec, self.embed_fn(hq)) for hq in hypothetical_qs]
        return float(np.mean(sims))


# ----------------------------------------------------------------------
# 4. 종합 평가
# ----------------------------------------------------------------------

@dataclass
class AnswerEvalResult:
    item_id: str
    f1: float
    rouge_l: float
    bertscore: float
    faithfulness: float
    answer_relevancy: float


def evaluate_answer(
    item_id: str,
    question: str,
    answer: str,
    context: str,
    reference: str,
    judge: AnswerJudge,
) -> AnswerEvalResult:
    # EM은 자유 서술형 응답에서는 항상 0에 가까워 변별력이 없어 이 평가에서는 제외한다.
    # (exact_match()는 위 2번 섹션에 그대로 남겨두었으니 필요하면 다시 넣어 쓰면 된다.)
    return AnswerEvalResult(
        item_id=item_id,
        f1=f1_score(answer, reference),
        rouge_l=rouge_l(answer, reference),
        bertscore=bertscore_proxy(answer, reference, judge.embed_fn),
        faithfulness=judge.faithfulness(answer, context),
        answer_relevancy=judge.answer_relevancy(answer, question),
    )


def summarize_results(results: List[AnswerEvalResult]) -> pd.DataFrame:
    """항목별 결과와 함께, 지표별 평균/표준편차를 담은 요약 행을 반환한다."""
    df = pd.DataFrame([r.__dict__ for r in results])
    metric_cols = ["f1", "rouge_l", "bertscore", "faithfulness", "answer_relevancy"]
    summary = {"item_id": "MEAN±STD"}
    for col in metric_cols:
        summary[col] = f"{df[col].mean():.4f}±{df[col].std():.4f}"
    return pd.concat([df.round(4), pd.DataFrame([summary])], ignore_index=True)


# ----------------------------------------------------------------------
# 5. ch03 RAG 파이프라인 연동 + 실행부
# ----------------------------------------------------------------------
# 백본 모델(exaone3.5:2.4b, ch03 rag_retriever.llm)이 실제로 생성한 답변을
# 채점 대상으로 삼고, 판정자(judge)로는 이해충돌을 피하기 위해 다른 계열의
# 모델(timHan/llama3.2korean3B4QKM)을 사용한다. 둘 다 로컬 Ollama 모델이며,
# 4.2절이 명시한 "동일 백본(EXAONE 3.5 2.4B, Ollama)에서 하드웨어·템플릿·
# 디코딩 파라미터(temperature=0)를 통제"하는 조건과 일치한다.

CH03_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ch03"))
JUDGE_MODEL = "timHan/llama3.2korean3B4QKM"


def load_testset(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def generate_answer(rag_retriever, question: str) -> tuple[str, str]:
    """rag_run.py의 "/think" 경로(augment_query -> 하이브리드 검색+재순위화,
    fetch_k=HYBRID_FETCH_K, adaptive_top_n)를 그대로 태워 (컨텍스트, 답변)을 만든다.
    testset이 대화 이력 없는 단발 질의라 augment_query에는 빈 이력을 넘긴다
    (think_mode_eval.run_think_on과 동일한 방식).
    """
    augmented_query = rag_retriever.augment_query([], question)
    docs = rag_retriever.search_with_queries(
        augmented_query,
        [augmented_query],
        fetch_k=rag_retriever.HYBRID_FETCH_K,
        adaptive_top_n=True,
    )
    context = "\n\n".join(doc.page_content for doc in docs)
    prompt = (
        "다음 컨텍스트만 근거로 질문에 간결하게 답하세요.\n\n"
        f"컨텍스트:\n{context}\n\n질문: {question}"
    )
    answer = rag_retriever.llm.invoke(prompt).content
    return context, answer


if __name__ == "__main__":
    if CH03_DIR not in sys.path:
        sys.path.insert(0, CH03_DIR)

    try:
        import rag_retriever  # ch03/rag_retriever.py
        from langchain_ollama import ChatOllama
    except Exception as exc:  # noqa: BLE001 - 원인이 다양해서 메시지로 안내
        raise RuntimeError(
            "ch03.rag_retriever를 불러오지 못했습니다. Ollama(`ollama serve`)와 "
            "Qdrant(ch02/docker-compose.yml)가 실행 중인지 확인하세요."
        ) from exc

    judge = AnswerJudge(
        judge_llm=ChatOllama(model=JUDGE_MODEL, temperature=0),
        embed_fn=rag_retriever.embedding.embed_query,
    )

    testset_path = os.path.join(os.path.dirname(__file__), "answer_eval_testset.json")
    testset = load_testset(testset_path)

    results = []
    for item in testset:
        print(f"[{item['item_id']}] 답변 생성 중... ({item['question'][:30]}...)")
        context, answer = generate_answer(rag_retriever, item["question"])
        print(f"[{item['item_id']}] 답변: {answer}")

        result = evaluate_answer(
            item_id=item["item_id"],
            question=item["question"],
            answer=answer,
            context=context,
            reference=item["reference"],
            judge=judge,
        )
        results.append(result)
        print(f"[{item['item_id']}] {result}\n")

    print("=== 종합 결과 ===")
    print(summarize_results(results).to_string(index=False))
