# -*- coding: utf-8 -*-
"""
成本分析 API 路由
"""

from typing import Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from ..llm.cost_control import (
    get_cost_controller,
    CostLimit,
    MODEL_PRICING,
)


router = APIRouter(prefix="/api/v1/cost", tags=["Cost Analysis"])


class CostStatusResponse(BaseModel):
    """成本状态响应"""
    success: bool
    data: dict


class CostLimitResponse(BaseModel):
    """成本限制响应"""
    success: bool
    data: dict


@router.get("/status", response_model=CostStatusResponse)
async def get_cost_status(
    tenant_id: str = Query("default", description="租户 ID"),
):
    """
    获取成本状态

    返回:
    - hourly_cost: 小时成本
    - daily_cost: 日成本
    - monthly_cost: 月成本
    - hourly_tokens: 小时 token 数
    - daily_tokens: 日 token 数
    - circuit_state: 熔断状态
    - is_rate_limited: 是否被限流
    - warning_message: 警告信息
    """
    try:
        controller = get_cost_controller()
        status = controller.get_status(tenant_id)

        return CostStatusResponse(
            success=True,
            data={
                "hourly_cost": round(status.hourly_cost, 4),
                "daily_cost": round(status.daily_cost, 4),
                "monthly_cost": round(status.monthly_cost, 4),
                "hourly_tokens": status.hourly_tokens,
                "daily_tokens": status.daily_tokens,
                "circuit_state": status.circuit_state.name,
                "is_rate_limited": status.is_rate_limited,
                "warning_message": status.warning_message,
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取成本状态失败: {str(e)}")


@router.get("/limits", response_model=CostLimitResponse)
async def get_cost_limits(
    tenant_id: str = Query("default", description="租户 ID"),
):
    """
    获取成本限制配置

    返回:
    - hourly_limit: 小时限制 (USD)
    - daily_limit: 日限制 (USD)
    - monthly_limit: 月限制 (USD)
    - max_tokens_per_request: 单次请求最大 token
    - requests_per_minute: 每分钟最大请求数
    - warning_threshold: 警告阈值
    - circuit_threshold: 熔断阈值
    """
    try:
        controller = get_cost_controller()
        limit = controller.get_limit(tenant_id)

        return CostLimitResponse(
            success=True,
            data={
                "hourly_limit": limit.hourly_limit,
                "daily_limit": limit.daily_limit,
                "monthly_limit": limit.monthly_limit,
                "max_tokens_per_request": limit.max_tokens_per_request,
                "requests_per_minute": limit.requests_per_minute,
                "warning_threshold": limit.warning_threshold,
                "circuit_threshold": limit.circuit_threshold,
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取成本限制失败: {str(e)}")


@router.put("/limits")
async def update_cost_limits(
    tenant_id: str = Query("default", description="租户 ID"),
    hourly_limit: Optional[float] = Query(None, description="小时限制 (USD)"),
    daily_limit: Optional[float] = Query(None, description="日限制 (USD)"),
    monthly_limit: Optional[float] = Query(None, description="月限制 (USD)"),
    max_tokens_per_request: Optional[int] = Query(None, description="单次请求最大 token"),
    requests_per_minute: Optional[int] = Query(None, description="每分钟最大请求数"),
):
    """
    更新成本限制配置
    """
    try:
        controller = get_cost_controller()
        current_limit = controller.get_limit(tenant_id)

        # 更新限制
        new_limit = CostLimit(
            hourly_limit=hourly_limit if hourly_limit is not None else current_limit.hourly_limit,
            daily_limit=daily_limit if daily_limit is not None else current_limit.daily_limit,
            monthly_limit=monthly_limit if monthly_limit is not None else current_limit.monthly_limit,
            max_tokens_per_request=max_tokens_per_request if max_tokens_per_request is not None else current_limit.max_tokens_per_request,
            requests_per_minute=requests_per_minute if requests_per_minute is not None else current_limit.requests_per_minute,
            warning_threshold=current_limit.warning_threshold,
            circuit_threshold=current_limit.circuit_threshold,
        )

        controller.set_tenant_limit(tenant_id, new_limit)

        return {
            "success": True,
            "message": "成本限制已更新",
            "data": {
                "hourly_limit": new_limit.hourly_limit,
                "daily_limit": new_limit.daily_limit,
                "monthly_limit": new_limit.monthly_limit,
                "max_tokens_per_request": new_limit.max_tokens_per_request,
                "requests_per_minute": new_limit.requests_per_minute,
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"更新成本限制失败: {str(e)}")


@router.get("/usage")
async def get_usage_history(
    tenant_id: str = Query("default", description="租户 ID"),
    days: int = Query(7, ge=1, le=30, description="查询天数"),
):
    """
    获取使用历史

    返回按日期分组的使用统计
    """
    try:
        controller = get_cost_controller()
        now = datetime.utcnow()
        cutoff = now - timedelta(days=days)

        # 获取记录
        records = [
            r for r in controller.usage_records
            if r.tenant_id == tenant_id and r.timestamp > cutoff
        ]

        # 按日期分组
        daily_usage = {}
        for record in records:
            date_str = record.timestamp.strftime("%Y-%m-%d")
            if date_str not in daily_usage:
                daily_usage[date_str] = {
                    "date": date_str,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "cost": 0.0,
                    "requests": 0,
                }
            daily_usage[date_str]["input_tokens"] += record.input_tokens
            daily_usage[date_str]["output_tokens"] += record.output_tokens
            daily_usage[date_str]["total_tokens"] += record.input_tokens + record.output_tokens
            daily_usage[date_str]["cost"] += record.cost
            daily_usage[date_str]["requests"] += 1

        # 转换为列表并排序
        usage_list = sorted(daily_usage.values(), key=lambda x: x["date"])

        # 计算总计
        total = {
            "input_tokens": sum(u["input_tokens"] for u in usage_list),
            "output_tokens": sum(u["output_tokens"] for u in usage_list),
            "total_tokens": sum(u["total_tokens"] for u in usage_list),
            "cost": round(sum(u["cost"] for u in usage_list), 4),
            "requests": sum(u["requests"] for u in usage_list),
        }

        return {
            "success": True,
            "data": {
                "daily": usage_list,
                "total": total,
                "days": days,
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取使用历史失败: {str(e)}")


@router.get("/models")
async def get_model_pricing():
    """
    获取模型定价信息

    返回各模型的 token 定价 (每 1M tokens, USD)
    """
    try:
        pricing_list = []
        for model, pricing in MODEL_PRICING.items():
            pricing_list.append({
                "model": model,
                "input_price": pricing.input_price,
                "output_price": pricing.output_price,
            })

        return {
            "success": True,
            "data": pricing_list,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取模型定价失败: {str(e)}")


@router.post("/circuit/reset")
async def reset_circuit_breaker(
    tenant_id: str = Query("default", description="租户 ID"),
):
    """
    重置熔断器

    手动将熔断器状态重置为 CLOSED
    """
    try:
        controller = get_cost_controller()
        controller.reset_circuit(tenant_id)

        return {
            "success": True,
            "message": f"租户 {tenant_id} 的熔断器已重置",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"重置熔断器失败: {str(e)}")


# 导出
__all__ = ["router"]
