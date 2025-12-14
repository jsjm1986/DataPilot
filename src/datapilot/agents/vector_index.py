# -*- coding: utf-8 -*-
"""
Vector Schema Index - 向量索引加速 Schema 检索

用于大 Schema 场景下的快速表检索:
1. 使用 OpenAI embedding API 生成表描述向量
2. 余弦相似度搜索相关表
3. 可选持久化索引
"""

import json
import hashlib
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

from ..llm.embeddings import get_embedding_client


@dataclass
class TableEmbedding:
    """表向量"""
    name: str
    description: str
    embedding: list[float] = field(default_factory=list)


class VectorSchemaIndex:
    """
    Schema 向量索引

    使用 embedding 向量进行快速表检索，
    适用于大 Schema (>50 表) 场景
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        embedding_model: str = "text-embedding-3-small",
    ):
        """
        初始化向量索引

        Args:
            cache_dir: 缓存目录 (用于持久化索引)
            embedding_model: embedding 模型名称
        """
        self.embedding_client = get_embedding_client()
        self.embedding_model = embedding_model
        self.cache_dir = Path(cache_dir) if cache_dir else None

        # 内存索引
        self.table_embeddings: dict[str, TableEmbedding] = {}
        self._index_hash: Optional[str] = None

    def _compute_hash(self, tables: list[dict]) -> str:
        """计算表列表的哈希值"""
        content = json.dumps(
            sorted([t.get("name", "") for t in tables]),
            sort_keys=True
        )
        return hashlib.md5(content.encode()).hexdigest()

    def _get_cache_path(self, index_hash: str) -> Optional[Path]:
        """获取缓存文件路径"""
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            return self.cache_dir / f"schema_index_{index_hash}.json"
        return None

    def _load_from_cache(self, index_hash: str) -> bool:
        """从缓存加载索引"""
        cache_path = self._get_cache_path(index_hash)
        if cache_path and cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                self.table_embeddings = {
                    name: TableEmbedding(
                        name=name,
                        description=item["description"],
                        embedding=item["embedding"],
                    )
                    for name, item in data.get("embeddings", {}).items()
                }
                self._index_hash = index_hash
                return True
            except Exception as e:
                print(f"Failed to load cache: {e}")
        return False

    def _save_to_cache(self, index_hash: str):
        """保存索引到缓存"""
        cache_path = self._get_cache_path(index_hash)
        if cache_path:
            try:
                data = {
                    "hash": index_hash,
                    "model": self.embedding_model,
                    "embeddings": {
                        name: {
                            "description": te.description,
                            "embedding": te.embedding,
                        }
                        for name, te in self.table_embeddings.items()
                    },
                }
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False)
            except Exception as e:
                print(f"Failed to save cache: {e}")

    def _build_table_description(self, table: dict) -> str:
        """构建表描述文本"""
        name = table.get("name", "")
        comment = table.get("comment", "")
        columns = table.get("columns", [])

        # 构建描述
        parts = [f"Table: {name}"]
        if comment:
            parts.append(f"Description: {comment}")

        if columns:
            col_descs = []
            for col in columns[:20]:  # 限制列数
                col_name = col.get("name", "")
                col_type = col.get("type", "")
                col_comment = col.get("comment", "")

                col_desc = f"{col_name} ({col_type})"
                if col_comment:
                    col_desc += f" - {col_comment}"
                col_descs.append(col_desc)

            parts.append("Columns: " + ", ".join(col_descs))

        return "\n".join(parts)

    async def build_index(self, tables: list[dict], force_rebuild: bool = False):
        """
        构建向量索引

        Args:
            tables: 表信息列表
            force_rebuild: 是否强制重建
        """
        if not tables:
            return

        # 计算哈希
        index_hash = self._compute_hash(tables)

        # 检查是否需要重建
        if not force_rebuild and self._index_hash == index_hash:
            return  # 索引已是最新

        # 尝试从缓存加载
        if not force_rebuild and self._load_from_cache(index_hash):
            return

        # 构建新索引
        self.table_embeddings = {}

        # 批量生成 embedding
        descriptions = []
        table_names = []

        for table in tables:
            name = table.get("name", "")
            if not name:
                continue

            description = self._build_table_description(table)
            descriptions.append(description)
            table_names.append(name)

        # 调用 embedding API
        if descriptions:
            try:
                embeddings = await self.embedding_client.embed_batch(descriptions)

                for name, desc, emb in zip(table_names, descriptions, embeddings):
                    self.table_embeddings[name] = TableEmbedding(
                        name=name,
                        description=desc,
                        embedding=emb,
                    )

                self._index_hash = index_hash
                self._save_to_cache(index_hash)

            except Exception as e:
                print(f"Failed to build vector index: {e}")
                # 回退到无向量模式
                for name, desc in zip(table_names, descriptions):
                    self.table_embeddings[name] = TableEmbedding(
                        name=name,
                        description=desc,
                        embedding=[],
                    )

    async def search(
        self,
        query: str,
        top_k: int = 10,
        threshold: float = 0.3,
    ) -> list[dict]:
        """
        向量检索相关表

        Args:
            query: 查询文本
            top_k: 返回数量
            threshold: 相似度阈值

        Returns:
            相关表列表，按相似度排序
        """
        if not self.table_embeddings:
            return []

        # 检查是否有向量
        has_vectors = any(te.embedding for te in self.table_embeddings.values())

        if not has_vectors:
            # 无向量，使用关键词匹配
            return self._keyword_search(query, top_k)

        # 生成查询向量
        try:
            query_embedding = await self.embedding_client.embed(query)
        except Exception as e:
            print(f"Failed to embed query: {e}")
            return self._keyword_search(query, top_k)

        # 计算相似度
        results = []
        for name, te in self.table_embeddings.items():
            if not te.embedding:
                continue

            similarity = self._cosine_similarity(query_embedding, te.embedding)
            if similarity >= threshold:
                results.append({
                    "name": name,
                    "score": similarity,
                    "source": "vector",
                })

        # 排序并返回
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def _keyword_search(self, query: str, top_k: int) -> list[dict]:
        """关键词搜索 (后备方案)"""
        query_lower = query.lower()
        results = []

        for name, te in self.table_embeddings.items():
            score = 0
            name_lower = name.lower()
            desc_lower = te.description.lower()

            # 表名匹配
            if name_lower in query_lower:
                score += 10
            for word in query_lower.split():
                if len(word) > 2 and word in name_lower:
                    score += 5
                if len(word) > 2 and word in desc_lower:
                    score += 2

            if score > 0:
                results.append({
                    "name": name,
                    "score": score / 20,  # 归一化
                    "source": "keyword",
                })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    @staticmethod
    def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
        """计算余弦相似度"""
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def get_stats(self) -> dict:
        """获取索引统计信息"""
        total = len(self.table_embeddings)
        with_vectors = sum(1 for te in self.table_embeddings.values() if te.embedding)

        return {
            "total_tables": total,
            "tables_with_vectors": with_vectors,
            "index_hash": self._index_hash,
            "embedding_model": self.embedding_model,
        }


# 便捷函数
def create_vector_index(cache_dir: str = None) -> VectorSchemaIndex:
    """创建向量索引实例"""
    return VectorSchemaIndex(cache_dir=cache_dir)


__all__ = ["VectorSchemaIndex", "TableEmbedding", "create_vector_index"]
