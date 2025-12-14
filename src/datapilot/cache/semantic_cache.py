# -*- coding: utf-8 -*-
"""
Semantic Cache (语义缓存) - LanceDB 增强版

基于向量相似度的查询缓存，支持：
1. 精确匹配 (哈希)
2. 语义匹配 (Embedding + 余弦相似度)
3. LanceDB 向量存储 (新增，持久化)
4. Redis 分布式存储 (可选)
5. 内存存储 (默认)

存储后端优先级:
1. LanceDB (推荐，持久化 + 向量搜索)
2. 内存 + Redis (兼容模式)

README 要求:
- 相似度阈值 >= 0.85
- TTL 10-30 分钟
- 语义相似的查询直接返回缓存结果
"""

import json
import hashlib
import time
import math
from typing import Any, Optional
from dataclasses import dataclass, asdict

from ..config.settings import get_settings

# 尝试导入 LanceDB
try:
    from ..vector.lancedb_store import LanceDBStore, get_cache_store, LANCEDB_AVAILABLE
except ImportError:
    LANCEDB_AVAILABLE = False
    LanceDBStore = None
    get_cache_store = None


@dataclass
class CacheEntry:
    """缓存条目"""
    query: str                          # 原始查询
    query_hash: str                     # 查询哈希
    embedding: list[float]              # 查询向量
    sql: str                            # 生成的 SQL
    result: list[dict]                  # 查询结果
    row_count: int                      # 结果行数
    database: str                       # 数据库名称
    created_at: float                   # 创建时间
    ttl_seconds: int                    # 过期时间
    hit_count: int = 0                  # 命中次数
    last_hit_at: float = 0.0            # 最后命中时间


@dataclass
class CacheHit:
    """缓存命中结果"""
    query: str                          # 原始查询
    sql: str                            # SQL
    data: list[dict]                    # 结果数据
    row_count: int                      # 行数
    cached: bool = True                 # 是否来自缓存
    similarity: float = 1.0             # 相似度 (1.0 = 精确匹配)
    hit_count: int = 0                  # 命中次数
    cache_age_seconds: float = 0.0      # 缓存年龄


class SemanticCache:
    """
    语义缓存

    支持两种匹配模式：
    1. 精确匹配：基于查询哈希的快速匹配
    2. 语义匹配：基于 Embedding 向量的相似度匹配

    存储后端：
    1. LanceDB (推荐，向量搜索 + 持久化)
    2. 内存存储 (默认回退)
    3. Redis 分布式存储 (可选，用于哈希精确匹配)
    """

    def __init__(
        self,
        similarity_threshold: float = 0.85,
        ttl_minutes: int = 30,
        max_entries: int = 1000,
        use_redis: bool = False,
        use_lancedb: bool = True,
    ):
        """
        初始化语义缓存

        Args:
            similarity_threshold: 语义相似度阈值 (0.0-1.0)
            ttl_minutes: 缓存过期时间 (分钟)
            max_entries: 最大缓存条目数
            use_redis: 是否使用 Redis (用于精确匹配)
            use_lancedb: 是否使用 LanceDB (用于语义匹配)
        """
        settings = get_settings()
        self.similarity_threshold = similarity_threshold or settings.cache_similarity_threshold
        self.ttl_seconds = (ttl_minutes or settings.cache_ttl_minutes) * 60
        self.max_entries = max_entries
        self.use_redis = use_redis

        # LanceDB 向量存储
        self.use_lancedb = use_lancedb and LANCEDB_AVAILABLE
        self._lancedb: Optional[LanceDBStore] = None

        # 内存缓存 (用于结果数据和精确匹配)
        self._cache: dict[str, CacheEntry] = {}
        self._embedding_index: list[tuple[str, list[float]]] = []  # 回退用

        # Redis 客户端 (延迟初始化)
        self._redis: Optional[Any] = None

        # Embedding 客户端 (延迟初始化)
        self._embedding_client = None

    def _get_lancedb(self) -> Optional[LanceDBStore]:
        """获取 LanceDB 存储"""
        if not self.use_lancedb:
            return None

        if self._lancedb is None:
            try:
                self._lancedb = LanceDBStore(
                    table_name="query_cache",
                    vector_dim=1024,
                )
            except Exception as e:
                print(f"LanceDB init failed: {e}, using memory fallback")
                self.use_lancedb = False
                return None

        return self._lancedb

    async def _get_embedding_client(self):
        """获取 Embedding 客户端"""
        if self._embedding_client is None:
            from ..llm.embeddings import get_embedding_client
            self._embedding_client = get_embedding_client()
        return self._embedding_client

    async def _get_redis(self):
        """获取 Redis 客户端"""
        if not self.use_redis:
            return None

        if self._redis is None:
            try:
                import redis.asyncio as redis
                settings = get_settings()
                self._redis = redis.from_url(
                    settings.redis_url,
                    password=settings.redis_password or None,
                    encoding="utf-8",
                    decode_responses=True,
                )
                # 测试连接
                await self._redis.ping()
            except Exception as e:
                print(f"Redis connection failed: {e}, falling back to memory cache")
                self._redis = None
                self.use_redis = False

        return self._redis

    def _hash_query(self, query: str, database: str) -> str:
        """生成查询哈希"""
        normalized = f"{database}:{query.lower().strip()}"
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    def _is_expired(self, entry: CacheEntry) -> bool:
        """检查缓存是否过期"""
        return time.time() - entry.created_at > entry.ttl_seconds

    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """计算余弦相似度"""
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    async def _get_embedding(self, text: str) -> list[float]:
        """获取文本向量"""
        try:
            client = await self._get_embedding_client()
            return await client.embed_single(text)
        except Exception as e:
            print(f"Embedding error: {e}")
            return []

    # ============================================
    # 核心方法: get / set
    # ============================================

    async def get(
        self,
        query: str,
        database: str = "default",
        use_semantic: bool = True,
    ) -> Optional[CacheHit]:
        """
        获取缓存结果

        Args:
            query: 用户查询
            database: 数据库名称
            use_semantic: 是否使用语义匹配

        Returns:
            缓存命中结果，或 None
        """
        query_hash = self._hash_query(query, database)

        # 1. 精确匹配 (哈希)
        exact_hit = await self._get_exact(query_hash)
        if exact_hit:
            return exact_hit

        # 2. 语义匹配 (向量相似度)
        if use_semantic:
            semantic_hit = await self._get_semantic(query, database)
            if semantic_hit:
                return semantic_hit

        return None

    async def _get_exact(self, query_hash: str) -> Optional[CacheHit]:
        """精确匹配 (哈希)"""
        # 尝试 Redis
        redis = await self._get_redis()
        if redis:
            try:
                data = await redis.get(f"datapilot:cache:{query_hash}")
                if data:
                    entry_dict = json.loads(data)
                    entry = CacheEntry(**entry_dict)
                    if not self._is_expired(entry):
                        # 更新命中计数
                        entry.hit_count += 1
                        entry.last_hit_at = time.time()
                        await redis.setex(
                            f"datapilot:cache:{query_hash}",
                            self.ttl_seconds,
                            json.dumps(asdict(entry)),
                        )
                        return CacheHit(
                            query=entry.query,
                            sql=entry.sql,
                            data=entry.result,
                            row_count=entry.row_count,
                            similarity=1.0,
                            hit_count=entry.hit_count,
                            cache_age_seconds=time.time() - entry.created_at,
                        )
            except Exception as e:
                print(f"Redis get error: {e}")

        # 尝试内存缓存
        if query_hash in self._cache:
            entry = self._cache[query_hash]
            if not self._is_expired(entry):
                entry.hit_count += 1
                entry.last_hit_at = time.time()
                return CacheHit(
                    query=entry.query,
                    sql=entry.sql,
                    data=entry.result,
                    row_count=entry.row_count,
                    similarity=1.0,
                    hit_count=entry.hit_count,
                    cache_age_seconds=time.time() - entry.created_at,
                )
            else:
                # 删除过期条目
                del self._cache[query_hash]
                self._embedding_index = [
                    (h, e) for h, e in self._embedding_index if h != query_hash
                ]

        return None

    async def _get_semantic(self, query: str, database: str) -> Optional[CacheHit]:
        """语义匹配 (向量相似度)"""
        # 获取查询向量
        query_embedding = await self._get_embedding(query)
        if not query_embedding:
            return None

        # 优先使用 LanceDB 进行向量搜索
        lancedb = self._get_lancedb()
        if lancedb:
            return await self._get_semantic_lancedb(query_embedding, database)

        # 回退到内存索引搜索
        return await self._get_semantic_memory(query_embedding, database)

    async def _get_semantic_lancedb(
        self,
        query_embedding: list[float],
        database: str,
    ) -> Optional[CacheHit]:
        """使用 LanceDB 进行语义匹配"""
        lancedb = self._get_lancedb()
        if not lancedb:
            return None

        try:
            results = lancedb.search(
                query_vector=query_embedding,
                limit=1,
                score_threshold=self.similarity_threshold,
                filter_expr=f"database = '{database}'",
            )

            if not results:
                return None

            best = results[0]
            query_hash = best.metadata.get('query_hash')

            # 从内存缓存获取完整数据
            if query_hash and query_hash in self._cache:
                entry = self._cache[query_hash]
                if not self._is_expired(entry):
                    entry.hit_count += 1
                    entry.last_hit_at = time.time()
                    return CacheHit(
                        query=entry.query,
                        sql=entry.sql,
                        data=entry.result,
                        row_count=entry.row_count,
                        similarity=best.score,
                        hit_count=entry.hit_count,
                        cache_age_seconds=time.time() - entry.created_at,
                    )

            # 从 LanceDB metadata 恢复 (如果内存缓存丢失)
            sql = best.metadata.get('sql', '')
            result_json = best.metadata.get('result', '[]')
            try:
                result = json.loads(result_json) if isinstance(result_json, str) else result_json
            except:
                result = []

            if sql:
                return CacheHit(
                    query=best.text,
                    sql=sql,
                    data=result,
                    row_count=len(result),
                    similarity=best.score,
                    hit_count=1,
                    cache_age_seconds=0,
                )

        except Exception as e:
            print(f"LanceDB search error: {e}")

        return None

    async def _get_semantic_memory(
        self,
        query_embedding: list[float],
        database: str,
    ) -> Optional[CacheHit]:
        """使用内存索引进行语义匹配 (回退方案)"""
        best_match: Optional[tuple[str, float]] = None
        best_similarity = 0.0

        # 在内存索引中搜索
        for entry_hash, entry_embedding in self._embedding_index:
            if entry_hash not in self._cache:
                continue

            entry = self._cache[entry_hash]
            if entry.database != database:
                continue
            if self._is_expired(entry):
                continue

            similarity = self._cosine_similarity(query_embedding, entry_embedding)
            if similarity >= self.similarity_threshold and similarity > best_similarity:
                best_similarity = similarity
                best_match = (entry_hash, similarity)

        if best_match:
            entry_hash, similarity = best_match
            entry = self._cache[entry_hash]
            entry.hit_count += 1
            entry.last_hit_at = time.time()

            return CacheHit(
                query=entry.query,
                sql=entry.sql,
                data=entry.result,
                row_count=entry.row_count,
                similarity=similarity,
                hit_count=entry.hit_count,
                cache_age_seconds=time.time() - entry.created_at,
            )

        return None

    async def set(
        self,
        query: str,
        sql: str,
        result: list[dict],
        row_count: int,
        database: str = "default",
    ) -> None:
        """
        设置缓存

        Args:
            query: 用户查询
            sql: 生成的 SQL
            result: 查询结果
            row_count: 结果行数
            database: 数据库名称
        """
        query_hash = self._hash_query(query, database)

        # 获取查询向量
        query_embedding = await self._get_embedding(query)

        # 创建缓存条目
        entry = CacheEntry(
            query=query,
            query_hash=query_hash,
            embedding=query_embedding,
            sql=sql,
            result=result,
            row_count=row_count,
            database=database,
            created_at=time.time(),
            ttl_seconds=self.ttl_seconds,
        )

        # 容量检查和淘汰
        if len(self._cache) >= self.max_entries:
            await self._evict()

        # 存储到内存
        self._cache[query_hash] = entry

        # 存储到 LanceDB (向量索引)
        lancedb = self._get_lancedb()
        if lancedb and query_embedding:
            try:
                # 限制结果大小，避免 metadata 过大
                result_for_storage = result[:100] if len(result) > 100 else result
                lancedb.upsert(
                    ids=[query_hash],
                    vectors=[query_embedding],
                    texts=[query],
                    metadatas=[{
                        'query_hash': query_hash,
                        'sql': sql,
                        'result': json.dumps(result_for_storage, ensure_ascii=False),
                        'row_count': row_count,
                        'database': database,
                        'created_at': entry.created_at,
                    }],
                    database=database,
                    record_type='cache',
                )
            except Exception as e:
                print(f"LanceDB upsert error: {e}")
        else:
            # 回退到内存索引
            if query_embedding:
                self._embedding_index.append((query_hash, query_embedding))

        # 存储到 Redis (精确匹配)
        redis = await self._get_redis()
        if redis:
            try:
                # 存储条目 (不包含 embedding，节省空间)
                entry_dict = asdict(entry)
                entry_dict["embedding"] = []  # Redis 中不存储向量
                await redis.setex(
                    f"datapilot:cache:{query_hash}",
                    self.ttl_seconds,
                    json.dumps(entry_dict),
                )
            except Exception as e:
                print(f"Redis set error: {e}")

    async def _evict(self) -> None:
        """淘汰策略: LRU (最近最少使用)"""
        if not self._cache:
            return

        # 按 last_hit_at 和 hit_count 排序
        sorted_entries = sorted(
            self._cache.items(),
            key=lambda x: (x[1].last_hit_at or x[1].created_at, x[1].hit_count)
        )

        # 删除最旧的 10%
        to_remove = max(1, len(sorted_entries) // 10)
        hashes_to_remove = [h for h, _ in sorted_entries[:to_remove]]

        for entry_hash in hashes_to_remove:
            del self._cache[entry_hash]

        # 清理内存索引
        self._embedding_index = [
            (h, e) for h, e in self._embedding_index if h not in hashes_to_remove
        ]

        # 清理 LanceDB
        lancedb = self._get_lancedb()
        if lancedb:
            try:
                lancedb.delete(ids=hashes_to_remove)
            except Exception:
                pass

    # ============================================
    # 管理方法
    # ============================================

    async def invalidate(
        self,
        query: Optional[str] = None,
        database: str = "default",
    ) -> int:
        """
        使缓存失效

        Args:
            query: 特定查询，或 None 表示全部
            database: 数据库名称

        Returns:
            失效的条目数
        """
        if query:
            query_hash = self._hash_query(query, database)
            count = 0

            # 内存
            if query_hash in self._cache:
                del self._cache[query_hash]
                self._embedding_index = [
                    (h, e) for h, e in self._embedding_index if h != query_hash
                ]
                count = 1

            # LanceDB
            lancedb = self._get_lancedb()
            if lancedb:
                try:
                    lancedb.delete(ids=[query_hash])
                except Exception:
                    pass

            # Redis
            redis = await self._get_redis()
            if redis:
                try:
                    await redis.delete(f"datapilot:cache:{query_hash}")
                except Exception:
                    pass

            return count
        else:
            # 全部失效
            count = len(self._cache)
            self._cache.clear()
            self._embedding_index.clear()

            # LanceDB
            lancedb = self._get_lancedb()
            if lancedb:
                try:
                    lancedb.clear(database=database)
                except Exception:
                    pass

            # Redis
            redis = await self._get_redis()
            if redis:
                try:
                    keys = await redis.keys("datapilot:cache:*")
                    if keys:
                        await redis.delete(*keys)
                except Exception:
                    pass

            return count

    async def cleanup_expired(self) -> int:
        """清理过期条目"""
        expired_hashes = [
            h for h, entry in self._cache.items()
            if self._is_expired(entry)
        ]

        for h in expired_hashes:
            del self._cache[h]

        self._embedding_index = [
            (h, e) for h, e in self._embedding_index if h not in expired_hashes
        ]

        # 清理 LanceDB
        lancedb = self._get_lancedb()
        if lancedb and expired_hashes:
            try:
                lancedb.delete(ids=expired_hashes)
            except Exception:
                pass

        return len(expired_hashes)

    def stats(self) -> dict:
        """获取缓存统计"""
        total_hits = sum(e.hit_count for e in self._cache.values())
        expired = sum(1 for e in self._cache.values() if self._is_expired(e))
        avg_age = 0.0
        if self._cache:
            avg_age = sum(time.time() - e.created_at for e in self._cache.values()) / len(self._cache)

        return {
            "total_entries": len(self._cache),
            "expired_entries": expired,
            "total_hits": total_hits,
            "max_entries": self.max_entries,
            "ttl_seconds": self.ttl_seconds,
            "similarity_threshold": self.similarity_threshold,
            "embedding_index_size": len(self._embedding_index),
            "avg_cache_age_seconds": avg_age,
            "use_redis": self.use_redis,
            "use_lancedb": self.use_lancedb,
            "backend": "lancedb" if self.use_lancedb else "memory",
        }

    async def close(self):
        """关闭连接"""
        if self._redis:
            await self._redis.close()
            self._redis = None

        if self._embedding_client:
            await self._embedding_client.close()
            self._embedding_client = None


# ============================================
# 全局单例
# ============================================

_cache: Optional[SemanticCache] = None


def get_cache() -> SemanticCache:
    """获取全局缓存单例"""
    global _cache
    if _cache is None:
        settings = get_settings()
        _cache = SemanticCache(
            similarity_threshold=settings.cache_similarity_threshold,
            ttl_minutes=settings.cache_ttl_minutes,
            use_redis=bool(settings.redis_url and settings.redis_url != "redis://localhost:6379"),
            use_lancedb=LANCEDB_AVAILABLE,
        )
    return _cache


async def get_cache_async() -> SemanticCache:
    """获取全局缓存单例 (异步版本)"""
    return get_cache()


__all__ = [
    "SemanticCache",
    "CacheEntry",
    "CacheHit",
    "get_cache",
    "get_cache_async",
]
