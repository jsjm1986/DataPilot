# -*- coding: utf-8 -*-
"""
审计日志 API 路由
"""

from typing import Optional, Literal
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from ..observability.audit import get_audit_store


router = APIRouter(prefix="/api/v1/audit", tags=["Audit Logs"])


class AuditLogResponse(BaseModel):
    """审计日志响应"""
    success: bool
    data: dict


class AuditStatsResponse(BaseModel):
    """审计统计响应"""
    success: bool
    data: dict


@router.get("/logs", response_model=AuditLogResponse)
async def list_audit_logs(
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页条目数"),
    user_id: Optional[str] = Query(None, description="用户 ID 过滤"),
    tenant_id: Optional[str] = Query(None, description="租户 ID 过滤"),
    database: Optional[str] = Query(None, description="数据库过滤"),
    status: Optional[str] = Query(None, description="状态过滤 (success/error/rejected/timeout)"),
    start_time: Optional[str] = Query(None, description="开始时间 (ISO 格式)"),
    end_time: Optional[str] = Query(None, description="结束时间 (ISO 格式)"),
    search: Optional[str] = Query(None, description="搜索查询文本"),
    order_by: str = Query("timestamp", description="排序字段"),
    order_dir: Literal["ASC", "DESC"] = Query("DESC", description="排序方向"),
):
    """
    获取审计日志列表

    支持分页、筛选和搜索
    """
    try:
        store = get_audit_store()

        # 计算偏移量
        offset = (page - 1) * page_size

        # 查询日志
        logs = await store.query(
            user_id=user_id,
            tenant_id=tenant_id,
            database=database,
            status=status,
            start_time=start_time,
            end_time=end_time,
            search_query=search,
            limit=page_size,
            offset=offset,
            order_by=order_by,
            order_dir=order_dir,
        )

        # 获取总数
        total = await store.count(
            tenant_id=tenant_id,
            status=status,
            start_time=start_time,
            end_time=end_time,
        )

        return AuditLogResponse(
            success=True,
            data={
                "logs": logs,
                "total": total,
                "page": page,
                "page_size": page_size,
                "total_pages": (total + page_size - 1) // page_size,
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取审计日志失败: {str(e)}")


@router.get("/logs/{trace_id}")
async def get_audit_log_detail(trace_id: str):
    """
    获取单条审计日志详情

    Args:
        trace_id: 追踪 ID
    """
    try:
        store = get_audit_store()
        log = await store.get_by_trace_id(trace_id)

        if not log:
            raise HTTPException(status_code=404, detail="审计日志不存在")

        return {
            "success": True,
            "data": log,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取审计日志详情失败: {str(e)}")


@router.get("/stats", response_model=AuditStatsResponse)
async def get_audit_statistics(
    tenant_id: Optional[str] = Query(None, description="租户 ID 过滤"),
    start_time: Optional[str] = Query(None, description="开始时间 (ISO 格式)"),
    end_time: Optional[str] = Query(None, description="结束时间 (ISO 格式)"),
):
    """
    获取审计统计信息

    返回:
    - overall: 总体统计 (总查询数、成功数、错误数、缓存命中率等)
    - by_database: 按数据库统计
    - by_status: 按状态统计
    - top_users: 活跃用户 Top 10
    - daily_trend: 每日趋势 (最近 30 天)
    """
    try:
        store = get_audit_store()
        stats = await store.get_statistics(
            tenant_id=tenant_id,
            start_time=start_time,
            end_time=end_time,
        )

        return AuditStatsResponse(success=True, data=stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取审计统计失败: {str(e)}")


@router.post("/cleanup")
async def cleanup_old_logs():
    """
    清理过期审计日志

    根据配置的保留天数清理过期日志
    """
    try:
        store = get_audit_store()
        deleted = await store.cleanup_old_logs()

        return {
            "success": True,
            "message": f"已清理 {deleted} 条过期日志",
            "deleted_count": deleted,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"清理审计日志失败: {str(e)}")


# 导出
__all__ = ["router"]
