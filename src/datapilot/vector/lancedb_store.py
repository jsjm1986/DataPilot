# -*- coding: utf-8 -*-
"""
LanceDB Vector Store

轻量级嵌入式向量数据库，支持:
1. 本地文件持久化 (跨平台，Windows 友好)
2. SQL 风格元数据过滤
3. 高效增量更新
4. 零服务架构

替代 Qdrant 作为默认向量存储方案。
"""

import os
import hashlib
from pathlib import Path
from typing import Any, Optional
from dataclasses import dataclass, field

import pyarrow as pa

try:
    import lancedb
    from lancedb.pydantic import LanceModel, Vector
    LANCEDB_AVAILABLE = True
except ImportError:
    LANCEDB_AVAILABLE = False
    lancedb = None
    LanceModel = None
    Vector = None

from ..config.settings import get_settings


# ============================================
# 数据模型定义
# ============================================

@dataclass
class VectorRecord:
    """向量记录"""
    id: str
    vector: list[float]
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """搜索结果"""
    id: str
    score: float
    text: str
    metadata: dict[str, Any]


# ============================================
# LanceDB Store
# ============================================

class LanceDBStore:
    """
    LanceDB 向量存储

    特点:
    - 纯嵌入式，无需外部服务
    - 本地文件持久化
    - 支持 SQL 风格过滤
    - Windows/Linux/Mac 全平台支持

    使用示例:
    ```python
    store = LanceDBStore(table_name="schema_index")
    store.upsert(ids=["t1"], vectors=[[0.1, 0.2, ...]], texts=["users表"])
    results = store.search(query_vector=[0.1, ...], limit=5)
    ```
    """

    def __init__(
        self,
        table_name: str = "default",
        db_path: Optional[str] = None,
        vector_dim: int = 1024,
        metric: str = "cosine",
    ):
        """
        初始化 LanceDB Store

        Args:
            table_name: 表名 (如 schema_index, value_index, query_cache)
            db_path: 数据库路径，默认 data/lancedb
            vector_dim: 向量维度，默认 1024 (text-embedding-3-small)
            metric: 距离度量，cosine/L2/dot
        """
        if not LANCEDB_AVAILABLE:
            raise ImportError(
                "LanceDB not installed. Run: pip install lancedb"
            )

        self.table_name = table_name
        self.vector_dim = vector_dim
        self.metric = metric

        # 确定数据库路径
        if db_path is None:
            settings = get_settings()
            db_path = getattr(settings, 'lancedb_path', 'data/lancedb')

        self.db_path = db_path
        Path(db_path).mkdir(parents=True, exist_ok=True)

        # 连接数据库
        self._db = lancedb.connect(db_path)
        self._table = None

        # 尝试打开现有表
        self._open_or_create_table()

    def _open_or_create_table(self):
        """打开或创建表"""
        try:
            existing_tables = self._db.table_names()
            if self.table_name in existing_tables:
                self._table = self._db.open_table(self.table_name)
            else:
                self._table = None
        except Exception:
            self._table = None

    def _ensure_table(self, sample_vector: list[float]) -> None:
        """确保表存在，如不存在则创建"""
        if self._table is not None:
            return

        # 创建空表的 schema
        schema = pa.schema([
            pa.field("id", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), len(sample_vector))),
            pa.field("text", pa.string()),
            pa.field("metadata", pa.string()),  # JSON 序列化
            pa.field("database", pa.string()),  # 用于过滤
            pa.field("table_name", pa.string()),  # 用于过滤
            pa.field("type", pa.string()),  # record type
        ])

        self._table = self._db.create_table(
            self.table_name,
            schema=schema,
            mode="overwrite",
        )

    def _generate_id(self, text: str, prefix: str = "") -> str:
        """生成唯一 ID"""
        content = f"{prefix}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    # ============================================
    # 核心方法
    # ============================================

    def upsert(
        self,
        ids: list[str],
        vectors: list[list[float]],
        texts: list[str],
        metadatas: Optional[list[dict[str, Any]]] = None,
        database: str = "default",
        record_type: str = "default",
    ) -> int:
        """
        插入或更新向量

        Args:
            ids: 记录 ID 列表
            vectors: 向量列表
            texts: 文本列表
            metadatas: 元数据列表
            database: 数据库标识 (用于过滤)
            record_type: 记录类型 (schema/value/query)

        Returns:
            插入/更新的记录数
        """
        import json

        if not vectors:
            return 0

        # 确保表存在
        self._ensure_table(vectors[0])

        # 准备数据
        records = []
        for i, (id_, vec, text) in enumerate(zip(ids, vectors, texts)):
            meta = metadatas[i] if metadatas else {}
            records.append({
                "id": id_,
                "vector": vec,
                "text": text,
                "metadata": json.dumps(meta, ensure_ascii=False),
                "database": meta.get("database", database),
                "table_name": meta.get("table_name", ""),
                "type": record_type,
            })

        # 删除已存在的记录 (实现 upsert)
        try:
            existing_ids = [r["id"] for r in records]
            self._table.delete(f"id IN {tuple(existing_ids)}" if len(existing_ids) > 1 else f"id = '{existing_ids[0]}'")
        except Exception:
            pass  # 表可能为空或 ID 不存在

        # 插入新记录
        self._table.add(records)

        return len(records)

    def search(
        self,
        query_vector: list[float],
        limit: int = 10,
        score_threshold: float = 0.0,
        filter_expr: Optional[str] = None,
        database: Optional[str] = None,
        record_type: Optional[str] = None,
    ) -> list[SearchResult]:
        """
        向量搜索

        Args:
            query_vector: 查询向量
            limit: 返回数量
            score_threshold: 最小分数阈值 (cosine: 0-1)
            filter_expr: SQL 过滤表达式 (如 "database = 'mydb'")
            database: 数据库过滤 (便捷参数)
            record_type: 记录类型过滤

        Returns:
            搜索结果列表
        """
        import json

        if self._table is None:
            return []

        # 构建查询
        query = self._table.search(query_vector)

        # 构建过滤条件
        filters = []
        if filter_expr:
            filters.append(filter_expr)
        if database:
            filters.append(f"database = '{database}'")
        if record_type:
            filters.append(f"type = '{record_type}'")

        if filters:
            query = query.where(" AND ".join(filters))

        # 执行搜索
        query = query.metric(self.metric).limit(limit)

        try:
            results = query.to_list()
        except Exception:
            return []

        # 转换结果
        search_results = []
        for r in results:
            # LanceDB 返回 _distance，cosine 距离越小越相似
            # 转换为相似度分数: score = 1 - distance (for cosine)
            distance = r.get("_distance", 0)
            if self.metric == "cosine":
                score = 1 - distance
            else:
                score = 1 / (1 + distance)  # L2

            if score < score_threshold:
                continue

            try:
                metadata = json.loads(r.get("metadata", "{}"))
            except:
                metadata = {}

            search_results.append(SearchResult(
                id=r.get("id", ""),
                score=score,
                text=r.get("text", ""),
                metadata=metadata,
            ))

        return search_results

    def delete(
        self,
        ids: Optional[list[str]] = None,
        filter_expr: Optional[str] = None,
        database: Optional[str] = None,
    ) -> int:
        """
        删除记录

        Args:
            ids: 要删除的 ID 列表
            filter_expr: SQL 过滤表达式
            database: 数据库过滤

        Returns:
            删除的记录数
        """
        if self._table is None:
            return 0

        # 构建删除条件
        conditions = []
        if ids:
            if len(ids) == 1:
                conditions.append(f"id = '{ids[0]}'")
            else:
                conditions.append(f"id IN {tuple(ids)}")
        if filter_expr:
            conditions.append(filter_expr)
        if database:
            conditions.append(f"database = '{database}'")

        if not conditions:
            return 0

        where_clause = " AND ".join(conditions)

        try:
            # 获取删除前的数量
            before_count = self._table.count_rows(where_clause)
            self._table.delete(where_clause)
            return before_count
        except Exception:
            return 0

    def count(
        self,
        filter_expr: Optional[str] = None,
        database: Optional[str] = None,
    ) -> int:
        """获取记录数"""
        if self._table is None:
            return 0

        try:
            if filter_expr:
                return self._table.count_rows(filter_expr)
            elif database:
                return self._table.count_rows(f"database = '{database}'")
            else:
                return self._table.count_rows()
        except Exception:
            return 0

    def clear(self, database: Optional[str] = None) -> None:
        """
        清空表

        Args:
            database: 如果指定，只清空该数据库的记录
        """
        if database:
            self.delete(database=database)
        else:
            # 删除整个表
            try:
                self._db.drop_table(self.table_name)
            except Exception:
                pass
            self._table = None

    def get_stats(self) -> dict[str, Any]:
        """获取统计信息"""
        if self._table is None:
            return {
                "table_name": self.table_name,
                "total_records": 0,
                "databases": [],
            }

        try:
            total = self._table.count_rows()

            # 获取数据库列表
            df = self._table.to_pandas()
            databases = df["database"].unique().tolist() if not df.empty else []

            return {
                "table_name": self.table_name,
                "total_records": total,
                "databases": databases,
                "db_path": self.db_path,
            }
        except Exception as e:
            return {
                "table_name": self.table_name,
                "error": str(e),
            }


# ============================================
# 便捷工厂函数
# ============================================

# 全局实例缓存
_stores: dict[str, LanceDBStore] = {}


def get_lancedb_store(
    table_name: str = "default",
    vector_dim: int = 1024,
) -> LanceDBStore:
    """
    获取 LanceDB Store 实例 (单例模式)

    Args:
        table_name: 表名
        vector_dim: 向量维度

    Returns:
        LanceDBStore 实例
    """
    global _stores

    key = f"{table_name}_{vector_dim}"
    if key not in _stores:
        _stores[key] = LanceDBStore(
            table_name=table_name,
            vector_dim=vector_dim,
        )

    return _stores[key]


def get_schema_store() -> LanceDBStore:
    """获取 Schema Index Store"""
    return get_lancedb_store("schema_index", vector_dim=1024)


def get_value_store() -> LanceDBStore:
    """获取 Value Index Store"""
    return get_lancedb_store("value_index", vector_dim=1024)


def get_cache_store() -> LanceDBStore:
    """获取 Query Cache Store"""
    return get_lancedb_store("query_cache", vector_dim=1024)


__all__ = [
    "LanceDBStore",
    "VectorRecord",
    "SearchResult",
    "get_lancedb_store",
    "get_schema_store",
    "get_value_store",
    "get_cache_store",
    "LANCEDB_AVAILABLE",
]
