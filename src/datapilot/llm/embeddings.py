"""
Embedding 封装层
使用 SiliconFlow API 进行向量嵌入
"""

import httpx
from typing import Optional
from ..config.settings import get_settings


class EmbeddingClient:
    """Embedding 客户端"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        settings = get_settings()
        self.api_key = api_key or settings.embedding_api_key
        self.base_url = base_url or settings.embedding_base_url
        self.model = model or settings.embedding_model
        self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=30.0,
            )
        return self._client

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """
        获取文本的向量嵌入

        Args:
            texts: 文本列表

        Returns:
            向量列表
        """
        response = await self.client.post(
            "/embeddings",
            json={
                "model": self.model,
                "input": texts,
                "encoding_format": "float",
            },
        )
        response.raise_for_status()
        data = response.json()

        # 按 index 排序确保顺序正确
        embeddings = sorted(data["data"], key=lambda x: x["index"])
        return [e["embedding"] for e in embeddings]

    async def embed_single(self, text: str) -> list[float]:
        """获取单个文本的向量嵌入"""
        embeddings = await self.embed([text])
        return embeddings[0]

    async def close(self):
        """关闭客户端"""
        if self._client:
            await self._client.aclose()
            self._client = None


class RerankClient:
    """Rerank 重排序客户端"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        settings = get_settings()
        self.api_key = api_key or settings.rerank_api_key
        self.base_url = base_url or settings.rerank_base_url
        self.model = model or settings.rerank_model
        self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=30.0,
            )
        return self._client

    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: Optional[int] = None,
    ) -> list[dict]:
        """
        对文档进行重排序

        Args:
            query: 查询文本
            documents: 文档列表
            top_n: 返回前 N 个结果

        Returns:
            排序后的结果列表，包含 index, score, document
        """
        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
        }
        if top_n:
            payload["top_n"] = top_n

        response = await self.client.post("/rerank", json=payload)
        response.raise_for_status()
        data = response.json()

        results = []
        for item in data.get("results", []):
            results.append({
                "index": item["index"],
                "score": item["relevance_score"],
                "document": documents[item["index"]],
            })

        return results

    async def close(self):
        """关闭客户端"""
        if self._client:
            await self._client.aclose()
            self._client = None


# 全局单例
_embedding_client: Optional[EmbeddingClient] = None
_rerank_client: Optional[RerankClient] = None


def get_embedding_client() -> EmbeddingClient:
    """获取 Embedding 客户端单例"""
    global _embedding_client
    if _embedding_client is None:
        _embedding_client = EmbeddingClient()
    return _embedding_client


def get_rerank_client() -> RerankClient:
    """获取 Rerank 客户端单例"""
    global _rerank_client
    if _rerank_client is None:
        _rerank_client = RerankClient()
    return _rerank_client


__all__ = [
    "EmbeddingClient",
    "RerankClient",
    "get_embedding_client",
    "get_rerank_client",
]
