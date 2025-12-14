# -*- coding: utf-8 -*-
"""
Schema Vector Index (LanceDB 版)

索引数据库 Schema (表、列) 用于语义检索
支持 LanceDB (默认) 和 Qdrant (可选) 双后端
"""

from typing import Optional, Any

from ..config.settings import get_settings
from ..llm.embeddings import get_embedding_client
from ..db.connector import get_db_manager

# 尝试导入 LanceDB，失败则回退到 Qdrant
try:
    from .lancedb_store import LanceDBStore, get_schema_store, LANCEDB_AVAILABLE
except ImportError:
    LANCEDB_AVAILABLE = False
    LanceDBStore = None
    get_schema_store = None

try:
    from .qdrant_client import QdrantStore
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    QdrantStore = None


class SchemaIndex:
    """
    Schema 向量索引

    为数据库表和列信息建立向量索引，支持语义检索。

    存储后端优先级:
    1. LanceDB (默认，轻量级，无需服务)
    2. Qdrant (可选，生产环境大规模场景)
    """

    TABLE_NAME = 'schema_index'

    def __init__(
        self,
        database: str = 'default',
        use_qdrant: bool = False,
    ):
        """
        初始化 Schema Index

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
                collection_name='datapilot_schema',
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
                collection_name='datapilot_schema',
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

    async def build_index(self) -> int:
        """
        从数据库构建 Schema 索引

        Returns:
            索引的记录数
        """
        db_manager = get_db_manager()
        connector = db_manager.get(self.database)
        tables = await connector.get_tables()

        ids = []
        texts = []
        metadatas = []

        for table in tables:
            table_name = table['name']
            table_comment = table.get('comment', '')
            columns = table.get('columns', [])

            # 索引表
            table_text = f"Table: {table_name}"
            if table_comment:
                table_text += f" ({table_comment})"

            col_names = [c['name'] for c in columns]
            table_text += " Columns: " + ", ".join(col_names)

            ids.append(f"table:{table_name}")
            texts.append(table_text)
            metadatas.append({
                'type': 'table',
                'table_name': table_name,
                'comment': table_comment,
                'columns': col_names,
                'database': self.database,
            })

            # 索引每个列
            for col in columns:
                col_name = col['name']
                col_type = col.get('type', '')
                col_comment = col.get('comment', '')

                col_text = f"Column: {table_name}.{col_name} ({col_type})"
                if col_comment:
                    col_text += f" - {col_comment}"

                ids.append(f"column:{table_name}.{col_name}")
                texts.append(col_text)
                metadatas.append({
                    'type': 'column',
                    'table_name': table_name,
                    'column': col_name,
                    'col_type': col_type,
                    'comment': col_comment,
                    'database': self.database,
                })

        # 批量获取 embeddings
        batch_size = 20
        total_indexed = 0

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
                    record_type='schema',
                )
            else:
                # Qdrant 接口
                self._store.upsert(batch_ids, vectors, batch_metadatas)

            total_indexed += len(batch_ids)

        return total_indexed

    async def search_tables(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.5,
    ) -> list[dict[str, Any]]:
        """
        搜索相关表

        Args:
            query: 搜索查询
            top_k: 最大返回数
            score_threshold: 最小相似度

        Returns:
            匹配的表列表
        """
        query_vector = await self._embedding.embed_single(query)

        if self._backend == 'lancedb':
            results = self._store.search(
                query_vector=query_vector,
                limit=top_k * 2,
                score_threshold=score_threshold,
                filter_expr=f"type = 'table' AND database = '{self.database}'",
            )
            # 转换结果格式
            tables = []
            seen = set()
            for r in results:
                table_name = r.metadata.get('table_name')
                if table_name and table_name not in seen:
                    seen.add(table_name)
                    tables.append({
                        'name': table_name,
                        'score': r.score,
                        'comment': r.metadata.get('comment', ''),
                        'columns': r.metadata.get('columns', []),
                    })
                if len(tables) >= top_k:
                    break
            return tables
        else:
            # Qdrant 接口
            results = self._store.search(
                query_vector=query_vector,
                limit=top_k * 2,
                score_threshold=score_threshold,
                filter_conditions={'type': 'table'},
            )
            seen = set()
            tables = []
            for r in results:
                table = r['payload'].get('table_name') or r['payload'].get('table')
                if table and table not in seen:
                    seen.add(table)
                    tables.append({
                        'name': table,
                        'score': r['score'],
                        'comment': r['payload'].get('comment', ''),
                        'columns': r['payload'].get('columns', []),
                    })
                if len(tables) >= top_k:
                    break
            return tables

    async def search_columns(
        self,
        query: str,
        table: Optional[str] = None,
        top_k: int = 10,
        score_threshold: float = 0.5,
    ) -> list[dict[str, Any]]:
        """
        搜索相关列

        Args:
            query: 搜索查询
            table: 可选的表过滤
            top_k: 最大返回数
            score_threshold: 最小相似度

        Returns:
            匹配的列列表
        """
        query_vector = await self._embedding.embed_single(query)

        if self._backend == 'lancedb':
            filter_parts = [f"type = 'column'", f"database = '{self.database}'"]
            if table:
                filter_parts.append(f"table_name = '{table}'")

            results = self._store.search(
                query_vector=query_vector,
                limit=top_k,
                score_threshold=score_threshold,
                filter_expr=" AND ".join(filter_parts),
            )
            return [
                {
                    'table': r.metadata.get('table_name'),
                    'column': r.metadata.get('column'),
                    'type': r.metadata.get('col_type'),
                    'score': r.score,
                }
                for r in results
            ]
        else:
            # Qdrant 接口
            filter_cond = {'type': 'column'}
            if table:
                filter_cond['table'] = table

            results = self._store.search(
                query_vector=query_vector,
                limit=top_k,
                score_threshold=score_threshold,
                filter_conditions=filter_cond,
            )
            return [
                {
                    'table': r['payload'].get('table_name') or r['payload'].get('table'),
                    'column': r['payload'].get('column'),
                    'type': r['payload'].get('col_type'),
                    'score': r['score'],
                }
                for r in results
            ]

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

_indexes: dict[str, SchemaIndex] = {}


def get_schema_index(database: str = 'default') -> SchemaIndex:
    """获取全局 Schema Index 实例"""
    global _indexes
    if database not in _indexes:
        _indexes[database] = SchemaIndex(database)
    return _indexes[database]


__all__ = ['SchemaIndex', 'get_schema_index']
