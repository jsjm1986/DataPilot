# -*- coding: utf-8 -*-
"""
缓存管理 API 路由
"""

from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..cache import get_cache


router = APIRouter(prefix="/api/v1/cache", tags=["Cache Management"])


class CacheStatsResponse(BaseModel):
    """缓存统计响应"""
    success: bool
    data: dict


class CacheEntriesResponse(BaseModel):
    """缓存条目列表响应"""
    success: bool
    data: dict


class ClearCacheResponse(BaseModel):
    """清除缓存响应"""
    success: bool
    message: str
    cleared_count: int


@router.get("/stats", response_model=CacheStatsResponse)
async def get_cache_stats():
    """
    获取缓存统计信息

    返回:
    - total_entries: 总条目数
    - expired_entries: 过期条目数
    - total_hits: 总命中次数
    - max_entries: 最大条目数
    - ttl_seconds: TTL 秒数
    - similarity_threshold: 相似度阈值
    - backend: 存储后端类型
    """
    try:
        cache = get_cache()
        stats = cache.stats()

        # 计算命中率 (如果有足够数据)
        hit_rate = 0.0
        if stats.get("total_hits", 0) > 0 and stats.get("total_entries", 0) > 0:
            # 简单估算: 命中次数 / (命中次数 + 条目数)
            hit_rate = stats["total_hits"] / (stats["total_hits"] + stats["total_entries"])

        stats["hit_rate"] = round(hit_rate, 4)

        return CacheStatsResponse(success=True, data=stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取缓存统计失败: {str(e)}")


@router.get("/entries", response_model=CacheEntriesResponse)
async def list_cache_entries(
    page: int = 1,
    page_size: int = 20,
    database: Optional[str] = None,
):
    """
    获取缓存条目列表

    Args:
        page: 页码 (从 1 开始)
        page_size: 每页条目数
        database: 筛选数据库
    """
    try:
        cache = get_cache()

        # 获取所有条目
        entries = []
        for query_hash, entry in cache._cache.items():
            # 筛选数据库
            if database and entry.database != database:
                continue

            entries.append({
                "id": query_hash,
                "query": entry.query[:100] + "..." if len(entry.query) > 100 else entry.query,
                "sql": entry.sql[:200] + "..." if len(entry.sql) > 200 else entry.sql,
                "database": entry.database,
                "row_count": entry.row_count,
                "hit_count": entry.hit_count,
                "created_at": entry.created_at,
                "ttl_seconds": entry.ttl_seconds,
                "is_expired": cache._is_expired(entry),
                "age_seconds": round(cache._cache and (
                    __import__("time").time() - entry.created_at
                ) or 0, 1),
            })

        # 按命中次数排序
        entries.sort(key=lambda x: x["hit_count"], reverse=True)

        # 分页
        total = len(entries)
        start = (page - 1) * page_size
        end = start + page_size
        paginated = entries[start:end]

        return CacheEntriesResponse(
            success=True,
            data={
                "entries": paginated,
                "total": total,
                "page": page,
                "page_size": page_size,
                "total_pages": (total + page_size - 1) // page_size,
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取缓存条目失败: {str(e)}")


@router.delete("/entries/{entry_id}")
async def delete_cache_entry(entry_id: str):
    """
    删除指定缓存条目

    Args:
        entry_id: 缓存条目 ID (query_hash)
    """
    try:
        cache = get_cache()

        if entry_id not in cache._cache:
            raise HTTPException(status_code=404, detail="缓存条目不存在")

        # 获取条目信息用于删除
        entry = cache._cache[entry_id]
        count = await cache.invalidate(query=entry.query, database=entry.database)

        return {
            "success": True,
            "message": f"已删除缓存条目",
            "deleted_count": count,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除缓存条目失败: {str(e)}")


@router.delete("/clear", response_model=ClearCacheResponse)
async def clear_cache(database: Optional[str] = None):
    """
    清空缓存

    Args:
        database: 指定数据库，或 None 清空全部
    """
    try:
        cache = get_cache()

        if database:
            # 清空指定数据库的缓存
            count = 0
            hashes_to_remove = [
                h for h, e in cache._cache.items()
                if e.database == database
            ]
            for h in hashes_to_remove:
                entry = cache._cache[h]
                await cache.invalidate(query=entry.query, database=entry.database)
                count += 1

            return ClearCacheResponse(
                success=True,
                message=f"已清空数据库 {database} 的缓存",
                cleared_count=count,
            )
        else:
            # 清空全部缓存
            count = await cache.invalidate(query=None, database="default")

            return ClearCacheResponse(
                success=True,
                message="已清空全部缓存",
                cleared_count=count,
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"清空缓存失败: {str(e)}")


@router.post("/cleanup")
async def cleanup_expired():
    """
    清理过期缓存条目
    """
    try:
        cache = get_cache()
        count = await cache.cleanup_expired()

        return {
            "success": True,
            "message": f"已清理 {count} 个过期条目",
            "cleaned_count": count,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"清理过期缓存失败: {str(e)}")


# 导出
__all__ = ["router"]
