"""적응형 RAG 컨텍스트 압축 프레임워크.

박사학위논문 제4장(적응형 RAG 컨텍스트 압축 프레임워크 설계)의 4.1~4.4절을 구현한
패키지이다. 로컬 LLM 서빙 환경 연동(4.5절, Ollama/GGUF 어댑터 등)은 이 패키지의
범위에 포함하지 않는다.

핵심 모듈 구성:
    - schemas            : 모듈 간 인터페이스 계약 (4.1.3, 4.4.4)
    - morphology          : 형태소 기반 전처리, 비정형 토큰 처리 (4.2.1, 4.2.6)
    - embeddings          : QuerySim/문서 중복도 산정을 위한 경량 문장 임베딩 (4.2.2)
    - scoring             : 토큰 중요도 기반 컨텍스트 스코어링 모듈 (4.2)
    - query_classifier    : 질의 유형(사실확인/추론) 분류기 (4.3.6)
    - compression         : 적응형 압축률 결정 알고리즘 (4.3)
    - pipeline            : 검색-압축 파이프라인 통합 구조 (4.4)
    - logging_utils       : 로깅 및 관측성 지표 (4.4.4, 4.4.8)
"""

from .schemas import (
    CompressionPlan,
    Decision,
    PRESERVE_SCORE,
    QueryLog,
    QueryType,
    RationaleEntry,
    ScoredChunk,
    Sentence,
    TokenScore,
)
from .morphology import MorphAnalyzer, Morpheme, POSCategory, POS_WEIGHT, RuleBasedMorphAnalyzer
from .embeddings import CharNgramEmbedder, RobertaEmbedder, SentenceEmbedder, default_embedder
from .scoring import DOC_TYPE_PROFILES, RawChunk, ScorerWeights, TokenImportanceScorer
from .query_classifier import QueryFeatures, QueryTypeClassifier
from .compression import AdaptiveCompressor, CompressionHyperparameters
from .pipeline import (
    CompressionPipeline,
    ContextAssembler,
    Retriever,
    SessionRedundancyCache,
    SimplePromptBuilder,
)
from .logging_utils import JsonlQueryLogger, compute_metrics

__all__ = [
    "CompressionPlan", "Decision", "PRESERVE_SCORE", "QueryLog", "QueryType",
    "RationaleEntry", "ScoredChunk", "Sentence", "TokenScore",
    "MorphAnalyzer", "Morpheme", "POSCategory", "POS_WEIGHT", "RuleBasedMorphAnalyzer",
    "CharNgramEmbedder", "RobertaEmbedder", "SentenceEmbedder", "default_embedder",
    "DOC_TYPE_PROFILES", "RawChunk", "ScorerWeights", "TokenImportanceScorer",
    "QueryFeatures", "QueryTypeClassifier",
    "AdaptiveCompressor", "CompressionHyperparameters",
    "CompressionPipeline", "ContextAssembler", "Retriever", "SessionRedundancyCache", "SimplePromptBuilder",
    "JsonlQueryLogger", "compute_metrics",
]
