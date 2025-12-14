# -*- coding: utf-8 -*-
"""
Value Vector Index (LanceDB 版)

索引高基数列值 (产品名、客户名等) 用于语义值映射
支持 LanceDB (默认) 和 Qdrant (可选) 双后端
"""

from typing import Optional, Any

from ..config.settings import get_settings
from ..llm.embeddings import get_embedding_client
from ..db.connector import get_db_manager

# 尝试导入 LanceDB，失败则回退到 Qdrant
try:
    from .lancedb_store import LanceDBStore, get_value_store, LANCEDB_AVAILABLE
except ImportError:
    LANCEDB_AVAILABLE = False
    LanceDBStore = None
    get_value_store = None

try:
    from .qdrant_client import QdrantStore
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    QdrantStore = None


class ValueIndex:
    """
    值向量索引

    索引高基数列值 (产品名、客户名等) 用于语义值映射。

    用途:
    - 将用户输入的模糊值映射到数据库中的精确值
    - 例如: "苹果手机" -> "iPhone 15 Pro Max"

    存储后端优先级:
    1. LanceDB (默认，轻量级，无需服务)
    2. Qdrant (可选，生产环境大规模场景)
    """

    TABLE_NAME = 'value_index'

    # 默认索引列 (高基数)
    DEFAULT_INDEX_COLUMNS = [
        ('products', 'name'),
        ('products', 'brand'),
        ('customers', 'name'),
        ('customers', 'city'),
        ('categories', 'name'),
    ]

    def __init__(
        self,
        database: str = 'default',
        use_qdrant: bool = False,
    ):
        """
        初始化 Value Index

        Args:
            database: 数据库标识
            use_qdrant: 是否强制使用 Qdrant
        """
        self.database = database
        self._embedding = get_embedding_client()

        # 选择存储后端
        settings = get_settings()
        force_qdrant = use_qdrant or getattr(settings, 'vector_backend', '') == 'qdrant'

        if force_qdrant and QDRANT_AVAILABLE:
            self._backend = 'qdrant'
            self._store = QdrantStore(
                collection_name='datapilot_values',
                vector_size=1024,
            )
        elif LANCEDB_AVAILABLE:
            self._backend = 'lancedb'
            self._store = LanceDBStore(
                table_name=self.TABLE_NAME,
                vector_dim=1024,
            )
        elif QDRANT_AVAILABLE:
            self._backend = 'qdrant'
            self._store = QdrantStore(
                collection_name='datapilot_values',
                vector_size=1024,
            )
        else:
            raise ImportError(
                "No vector backend available. Install lancedb or qdrant-client."
            )

    @property
    def backend(self) -> str:
        """获取当前后端名称"""
        return self._backend

    async def build_index(
        self,
        columns: Optional[list[tuple]] = None,
        batch_size: int = 20,
    ) -> int:
        """
        从数据库构建值索引

        Args:
            columns: 要索引的 (表, 列) 元组列表
            batch_size: 批量 embedding 大小

        Returns:
            索引的值数量
        """
        columns = columns or self.DEFAULT_INDEX_COLUMNS
        db_manager = get_db_manager()
        connector = db_manager.get(self.database)

        total_indexed = 0

        for table, column in columns:
            # 获取不重复的值
            sql = f'SELECT DISTINCT {column} FROM {table} WHERE {column} IS NOT NULL LIMIT 1000'
            try:
                rows = await connector.execute_query(sql)
            except Exception as e:
                print(f'Skip {table}.{column}: {e}')
                continue

            if not rows:
                continue

            # 准备数据
            ids = []
            texts = []
            metadatas = []

            for row in rows:
                value = row.get(column)
                if not value:
                    continue

                value_str = str(value)
                ids.append(f'value:{table}.{column}:{value_str[:50]}')
                texts.append(value_str)
                metadatas.append({
                    'table_name': table,
                    'column': column,
                    'value': value_str,
                    'database': self.database,
                })

            # 批量 embed 和 upsert
            for i in range(0, len(texts), batch_size):
                batch_ids = ids[i:i+batch_size]
                batch_texts = texts[i:i+batch_size]
                batch_metadatas = metadatas[i:i+batch_size]

                vectors = await self._embedding.embed(batch_texts)

                if self._backend == 'lancedb':
                    self._store.upsert(
                        ids=batch_ids,
                        vectors=vectors,
                        texts=batch_texts,
                        metadatas=batch_metadatas,
                        database=self.database,
                        record_type='value',
                    )
                else:
                    # Qdrant 接口
                    self._store.upsert(batch_ids, vectors, batch_metadatas)

                total_indexed += len(batch_ids)

        return total_indexed

    async def search_values(
        self,
        query: str,
        table: Optional[str] = None,
        column: Optional[str] = None,
        top_k: int = 5,
        score_threshold: float = 0.6,
    ) -> list[dict[str, Any]]:
        """
        搜索匹配的值

        Args:
            query: 搜索查询 (用户输入)
            table: 可选的表过滤
            column: 可选的列过滤
            top_k: 最大返回数
            score_threshold: 最小相似度

        Returns:
            匹配的值列表及元数据
        """
        query_vector = await self._embedding.embed_single(query)

        if self._backend == 'lancedb':
            filter_parts = [f"database = '{self.database}'"]
            if table:
                filter_parts.append(f"table_name = '{table}'")
            if column:
                # 需要在 metadata 中有 column 字段
                pass  # LanceDB 过滤在 metadata JSON 中较复杂，暂用全量过滤

            results = self._store.search(
                query_vector=query_vector,
                limit=top_k * 2,  # 多取一些，后面过滤
                score_threshold=score_threshold,
                filter_expr=" AND ".join(filter_parts) if filter_parts else None,
            )

            # 后过滤 column
            filtered = []
            for r in results:
                if column and r.metadata.get('column') != column:
                    continue
                filtered.append({
                    'value': r.metadata.get('value'),
                    'table': r.metadata.get('table_name'),
                    'column': r.metadata.get('column'),
                    'score': r.score,
                })
                if len(filtered) >= top_k:
                    break

            return filtered
        else:
            # Qdrant 接口
            filter_cond = {}
            if table:
                filter_cond['table'] = table
            if column:
                filter_cond['column'] = column

            results = self._store.search(
                query_vector=query_vector,
                limit=top_k,
                score_threshold=score_threshold,
                filter_conditions=filter_cond if filter_cond else None,
            )

            return [
                {
                    'value': r['payload'].get('value'),
                    'table': r['payload'].get('table_name') or r['payload'].get('table'),
                    'column': r['payload'].get('column'),
                    'score': r['score'],
                }
                for r in results
            ]

    async def map_entity(
        self,
        entity: str,
        table: Optional[str] = None,
        column: Optional[str] = None,
        top_k: int = 3,
    ) -> Optional[dict[str, Any]]:
        """
        将用户实体映射到数据库值

        Args:
            entity: 用户输入的实体
            table: 可选的表过滤
            column: 可选的列过滤
            top_k: 候选数量

        Returns:
            最佳匹配值或 None
        """
        results = await self.search_values(
            query=entity,
            table=table,
            column=column,
            top_k=top_k,
            score_threshold=0.7,
        )

        if not results:
            return None

        # 返回最佳匹配
        best = results[0]
        return {
            'input': entity,
            'db_value': best['value'],
            'table': best['table'],
            'column': best['column'],
            'score': best['score'],
            'alternatives': results[1:] if len(results) > 1 else [],
        }

    def clear(self) -> None:
        """清空索引"""
        if self._backend == 'lancedb':
            self._store.clear(database=self.database)
        else:
            self._store.clear()

    def count(self) -> int:
        """获取索引大小"""
        if self._backend == 'lancedb':
            return self._store.count(database=self.database)
        else:
            return self._store.count()

    def get_stats(self) -> dict[str, Any]:
        """获取统计信息"""
        return {
            'backend': self._backend,
            'database': self.database,
            'count': self.count(),
        }


# ============================================
# 全局实例
# ============================================

_indexes: dict[str, ValueIndex] = {}


def get_value_index(database: str = 'default') -> ValueIndex:
    """获取全局 Value Index 实例"""
    global _indexes
    if database not in _indexes:
        _indexes[database] = ValueIndex(database)
    return _indexes[database]


__all__ = ['ValueIndex', 'get_value_index']
