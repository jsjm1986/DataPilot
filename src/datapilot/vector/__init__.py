# -*- coding: utf-8 -*-
"""
Vector Store Module

向量存储模块，支持多后端:
1. LanceDB (推荐，轻量级，无需服务)
2. Qdrant (可选，生产环境大规模场景)
"""

# LanceDB (优先)
try:
    from .lancedb_store import (
        LanceDBStore,
        VectorRecord,
        SearchResult,
        get_lancedb_store,
        get_schema_store,
        get_value_store,
        get_cache_store,
        LANCEDB_AVAILABLE,
    )
except ImportError:
    LANCEDB_AVAILABLE = False
    LanceDBStore = None
    VectorRecord = None
    SearchResult = None
    get_lancedb_store = None
    get_schema_store = None
    get_value_store = None
    get_cache_store = None

# Qdrant (备选)
try:
    from .qdrant_client import QdrantStore, get_qdrant_store
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    QdrantStore = None
    get_qdrant_store = None

# 索引类 (自动选择后端)
from .schema_index import SchemaIndex, get_schema_index
from .value_index import ValueIndex, get_value_index

__all__ = [
    # LanceDB
    "LanceDBStore",
    "VectorRecord",
    "SearchResult",
    "get_lancedb_store",
    "get_schema_store",
    "get_value_store",
    "get_cache_store",
    "LANCEDB_AVAILABLE",
    # Qdrant
    "QdrantStore",
    "get_qdrant_store",
    "QDRANT_AVAILABLE",
    # 索引
    "SchemaIndex",
    "get_schema_index",
    "ValueIndex",
    "get_value_index",
]
