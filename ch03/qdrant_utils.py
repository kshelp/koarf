"""
qdrant_utils.py
================
ch03의 ingest/검색 스크립트들이 공통으로 쓰는 Qdrant 연결 헬퍼.
Qdrant는 ch02/docker-compose.yml로 localhost:6333에 띄워져 있다고 가정한다.
"""

from __future__ import annotations

from typing import Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

QDRANT_URL = "http://localhost:6333"

_SCROLL_PAGE_SIZE = 256


def get_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL)


def get_vectorstore(collection_name: str, embedding: Embeddings) -> QdrantVectorStore:
    """컬렉션이 이미 있으면 그대로 연결하고, 없으면 embedding 차원에 맞춰 새로 만든다
    (임베딩 모델을 한 번 호출해 벡터 크기를 알아내는 것까지 construct_instance가
    처리한다)."""
    return QdrantVectorStore.construct_instance(
        embedding=embedding,
        collection_name=collection_name,
        client_options={"url": QDRANT_URL},
    )


def fetch_all_documents(collection_name: str) -> list[Document]:
    """컬렉션의 모든 문서를 본문·메타데이터 그대로 가져온다.
    PGVector 시절 langchain_pg_embedding을 통째로 SELECT하던 것과 동등하며,
    BM25 캐시 재생성(rag_retriever.rebuild_bm25_cache)이 사용한다."""
    client = get_client()
    if not client.collection_exists(collection_name):
        return []

    docs = []
    offset = None
    while True:
        records, offset = client.scroll(
            collection_name=collection_name,
            limit=_SCROLL_PAGE_SIZE,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for record in records:
            payload = record.payload or {}
            docs.append(Document(
                page_content=payload.get("page_content", ""),
                metadata=payload.get("metadata") or {},
            ))
        if offset is None:
            break
    return docs


def find_source_by_filename(collection_name: str, file_name: str) -> Optional[str]:
    """metadata.source가 이 파일명으로 끝나는 청크가 있으면 그 source 문자열을
    반환하고 없으면 None을 반환한다(PGVector의 `LIKE '%/파일명'` 매칭과 동등).
    Qdrant는 서버 사이드 suffix 매치를 지원하지 않아 payload만 scroll로
    순회하며 파이썬에서 비교한다(ingest_pdf.py의 중복 적재 확인용)."""
    client = get_client()
    if not client.collection_exists(collection_name):
        return None

    suffix = f"/{file_name}"
    offset = None
    while True:
        records, offset = client.scroll(
            collection_name=collection_name,
            limit=_SCROLL_PAGE_SIZE,
            offset=offset,
            with_payload=["metadata"],
            with_vectors=False,
        )
        for record in records:
            source = (record.payload or {}).get("metadata", {}).get("source")
            if source and source.endswith(suffix):
                return source
        if offset is None:
            break
    return None


def update_source_by_filename(collection_name: str, file_name: str, new_source: str) -> int:
    """find_source_by_filename과 같은 기준으로 매칭되는 모든 포인트의
    metadata.source를 new_source로 일괄 갱신한다. 갱신된 포인트 수를 반환한다
    (ingest_pdf.py가 이미 적재된 문서를 새 part_XXXX 폴더로 옮길 때 사용)."""
    client = get_client()
    if not client.collection_exists(collection_name):
        return 0

    suffix = f"/{file_name}"
    matched_ids = []
    offset = None
    while True:
        records, offset = client.scroll(
            collection_name=collection_name,
            limit=_SCROLL_PAGE_SIZE,
            offset=offset,
            with_payload=["metadata"],
            with_vectors=False,
        )
        for record in records:
            source = (record.payload or {}).get("metadata", {}).get("source")
            if source and source.endswith(suffix):
                matched_ids.append(record.id)
        if offset is None:
            break

    if matched_ids:
        client.set_payload(
            collection_name=collection_name,
            payload={"source": new_source},
            points=matched_ids,
            key="metadata",
        )
    return len(matched_ids)
