# -*- coding: utf-8 -*-
"""
LLM 成本控制模块

实现成本熔断机制，防止 LLM 调用成本失控

功能:
1. Token 使用量追踪
2. 成本估算
3. 多级熔断 (警告/限流/熔断)
4. 租户级别配额管理
5. 自动恢复机制
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Callable, Any
from collections import defaultdict
import threading

from prometheus_client import Counter, Gauge


# ============================================
# Prometheus 指标
# ============================================

COST_TOTAL = Counter(
    "datapilot_llm_cost_total",
    "Total LLM cost in USD",
    labelnames=["model", "tenant_id"],
)

TOKENS_TOTAL = Counter(
    "datapilot_llm_tokens_total",
    "Total tokens used",
    labelnames=["model", "tenant_id", "type"],  # type: input/output
)

CIRCUIT_STATE = Gauge(
    "datapilot_llm_circuit_state",
    "Circuit breaker state (0=closed, 1=half_open, 2=open)",
    labelnames=["tenant_id"],
)

RATE_LIMIT_REMAINING = Gauge(
    "datapilot_llm_rate_limit_remaining",
    "Remaining requests in rate limit window",
    labelnames=["tenant_id"],
)


# ============================================
# 数据结构
# ============================================

class CircuitState(Enum):
    """熔断器状态"""
    CLOSED = 0      # 正常
    HALF_OPEN = 1   # 半开 (尝试恢复)
    OPEN = 2        # 熔断


@dataclass
class TokenPricing:
    """Token 定价 (每 1M tokens)"""
    input_price: float   # 输入 token 价格
    output_price: float  # 输出 token 价格


@dataclass
class UsageRecord:
    """使用记录"""
    timestamp: datetime
    input_tokens: int
    output_tokens: int
    cost: float
    model: str
    tenant_id: str


@dataclass
class CostLimit:
    """成本限制配置"""
    # 每小时限制
    hourly_limit: float = 10.0  # USD
    # 每日限制
    daily_limit: float = 100.0  # USD
    # 每月限制
    monthly_limit: float = 1000.0  # USD
    # 单次请求最大 token
    max_tokens_per_request: int = 4096
    # 每分钟最大请求数
    requests_per_minute: int = 60
    # 警告阈值 (百分比)
    warning_threshold: float = 0.8
    # 熔断阈值 (百分比)
    circuit_threshold: float = 0.95


@dataclass
class CostStatus:
    """成本状态"""
    hourly_cost: float = 0.0
    daily_cost: float = 0.0
    monthly_cost: float = 0.0
    hourly_tokens: int = 0
    daily_tokens: int = 0
    circuit_state: CircuitState = CircuitState.CLOSED
    is_rate_limited: bool = False
    warning_message: Optional[str] = None


# ============================================
# 定价配置
# ============================================

# DeepSeek 定价 (每 1M tokens, USD)
MODEL_PRICING = {
    "deepseek-chat": TokenPricing(input_price=0.14, output_price=0.28),
    "deepseek-coder": TokenPricing(input_price=0.14, output_price=0.28),
    "deepseek-v3": TokenPricing(input_price=0.27, output_price=1.10),
    # 默认定价
    "default": TokenPricing(input_price=0.50, output_price=1.00),
}


def get_pricing(model: str) -> TokenPricing:
    """获取模型定价"""
    # 尝试精确匹配
    if model in MODEL_PRICING:
        return MODEL_PRICING[model]
    # 尝试前缀匹配
    for key, pricing in MODEL_PRICING.items():
        if model.startswith(key):
            return pricing
    return MODEL_PRICING["default"]


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """估算成本"""
    pricing = get_pricing(model)
    input_cost = (input_tokens / 1_000_000) * pricing.input_price
    output_cost = (output_tokens / 1_000_000) * pricing.output_price
    return input_cost + output_cost


# ============================================
# 成本控制器
# ============================================

class CostController:
    """
    LLM 成本控制器

    功能:
    1. 追踪 token 使用量和成本
    2. 实现多级熔断机制
    3. 支持租户级别配额
    4. 自动恢复
    """

    def __init__(self, default_limit: Optional[CostLimit] = None):
        self.default_limit = default_limit or CostLimit()
        self.tenant_limits: dict[str, CostLimit] = {}
        self.usage_records: list[UsageRecord] = []
        self.circuit_states: dict[str, CircuitState] = defaultdict(lambda: CircuitState.CLOSED)
        self.circuit_open_time: dict[str, datetime] = {}
        self.request_counts: dict[str, list[float]] = defaultdict(list)
        self._lock = threading.Lock()

        # 熔断恢复时间 (秒)
        self.circuit_recovery_time = 60

        # 回调函数
        self.on_warning: Optional[Callable[[str, str], None]] = None
        self.on_circuit_open: Optional[Callable[[str], None]] = None

    def set_tenant_limit(self, tenant_id: str, limit: CostLimit):
        """设置租户限制"""
        self.tenant_limits[tenant_id] = limit

    def get_limit(self, tenant_id: str) -> CostLimit:
        """获取租户限制"""
        return self.tenant_limits.get(tenant_id, self.default_limit)

    def record_usage(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        tenant_id: str = "default",
    ):
        """记录使用量"""
        cost = estimate_cost(model, input_tokens, output_tokens)

        record = UsageRecord(
            timestamp=datetime.utcnow(),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            model=model,
            tenant_id=tenant_id,
        )

        with self._lock:
            self.usage_records.append(record)
            # 清理旧记录 (保留 30 天)
            cutoff = datetime.utcnow() - timedelta(days=30)
            self.usage_records = [r for r in self.usage_records if r.timestamp > cutoff]

        # 更新 Prometheus 指标
        COST_TOTAL.labels(model=model, tenant_id=tenant_id).inc(cost)
        TOKENS_TOTAL.labels(model=model, tenant_id=tenant_id, type="input").inc(input_tokens)
        TOKENS_TOTAL.labels(model=model, tenant_id=tenant_id, type="output").inc(output_tokens)

        # 检查是否需要触发熔断
        self._check_circuit(tenant_id)

    def get_status(self, tenant_id: str = "default") -> CostStatus:
        """获取成本状态"""
        now = datetime.utcnow()
        hour_ago = now - timedelta(hours=1)
        day_ago = now - timedelta(days=1)
        month_ago = now - timedelta(days=30)

        with self._lock:
            tenant_records = [r for r in self.usage_records if r.tenant_id == tenant_id]

            hourly_records = [r for r in tenant_records if r.timestamp > hour_ago]
            daily_records = [r for r in tenant_records if r.timestamp > day_ago]
            monthly_records = [r for r in tenant_records if r.timestamp > month_ago]

        hourly_cost = sum(r.cost for r in hourly_records)
        daily_cost = sum(r.cost for r in daily_records)
        monthly_cost = sum(r.cost for r in monthly_records)
        hourly_tokens = sum(r.input_tokens + r.output_tokens for r in hourly_records)
        daily_tokens = sum(r.input_tokens + r.output_tokens for r in daily_records)

        limit = self.get_limit(tenant_id)
        circuit_state = self.circuit_states[tenant_id]

        # 检查警告
        warning_message = None
        if hourly_cost >= limit.hourly_limit * limit.warning_threshold:
            warning_message = f"Hourly cost ({hourly_cost:.2f}) approaching limit ({limit.hourly_limit:.2f})"
        elif daily_cost >= limit.daily_limit * limit.warning_threshold:
            warning_message = f"Daily cost ({daily_cost:.2f}) approaching limit ({limit.daily_limit:.2f})"

        return CostStatus(
            hourly_cost=hourly_cost,
            daily_cost=daily_cost,
            monthly_cost=monthly_cost,
            hourly_tokens=hourly_tokens,
            daily_tokens=daily_tokens,
            circuit_state=circuit_state,
            is_rate_limited=self._is_rate_limited(tenant_id),
            warning_message=warning_message,
        )

    def check_allowed(self, tenant_id: str = "default", estimated_tokens: int = 0) -> tuple[bool, str]:
        """
        检查是否允许请求

        Returns:
            (allowed, reason)
        """
        limit = self.get_limit(tenant_id)
        status = self.get_status(tenant_id)

        # 检查熔断状态
        if status.circuit_state == CircuitState.OPEN:
            # 检查是否可以尝试恢复
            if self._can_try_recovery(tenant_id):
                self.circuit_states[tenant_id] = CircuitState.HALF_OPEN
                CIRCUIT_STATE.labels(tenant_id=tenant_id).set(1)
            else:
                return False, "Circuit breaker is OPEN. Please wait for recovery."

        # 检查速率限制
        if self._is_rate_limited(tenant_id):
            return False, f"Rate limited. Max {limit.requests_per_minute} requests per minute."

        # 检查成本限制
        if status.hourly_cost >= limit.hourly_limit:
            self._open_circuit(tenant_id, "Hourly cost limit exceeded")
            return False, f"Hourly cost limit exceeded ({status.hourly_cost:.2f} >= {limit.hourly_limit:.2f})"

        if status.daily_cost >= limit.daily_limit:
            self._open_circuit(tenant_id, "Daily cost limit exceeded")
            return False, f"Daily cost limit exceeded ({status.daily_cost:.2f} >= {limit.daily_limit:.2f})"

        # 检查单次请求 token 限制
        if estimated_tokens > limit.max_tokens_per_request:
            return False, f"Request too large ({estimated_tokens} > {limit.max_tokens_per_request} tokens)"

        # 记录请求
        self._record_request(tenant_id)

        return True, "OK"

    def _check_circuit(self, tenant_id: str):
        """检查是否需要触发熔断"""
        limit = self.get_limit(tenant_id)
        status = self.get_status(tenant_id)

        # 检查是否达到熔断阈值
        if status.hourly_cost >= limit.hourly_limit * limit.circuit_threshold:
            self._open_circuit(tenant_id, "Approaching hourly limit")
        elif status.daily_cost >= limit.daily_limit * limit.circuit_threshold:
            self._open_circuit(tenant_id, "Approaching daily limit")
        elif self.circuit_states[tenant_id] == CircuitState.HALF_OPEN:
            # 半开状态下成功，关闭熔断器
            self.circuit_states[tenant_id] = CircuitState.CLOSED
            CIRCUIT_STATE.labels(tenant_id=tenant_id).set(0)

    def _open_circuit(self, tenant_id: str, reason: str):
        """打开熔断器"""
        if self.circuit_states[tenant_id] != CircuitState.OPEN:
            self.circuit_states[tenant_id] = CircuitState.OPEN
            self.circuit_open_time[tenant_id] = datetime.utcnow()
            CIRCUIT_STATE.labels(tenant_id=tenant_id).set(2)

            if self.on_circuit_open:
                self.on_circuit_open(tenant_id)

            if self.on_warning:
                self.on_warning(tenant_id, f"Circuit breaker OPEN: {reason}")

    def _can_try_recovery(self, tenant_id: str) -> bool:
        """检查是否可以尝试恢复"""
        if tenant_id not in self.circuit_open_time:
            return True

        elapsed = (datetime.utcnow() - self.circuit_open_time[tenant_id]).total_seconds()
        return elapsed >= self.circuit_recovery_time

    def _is_rate_limited(self, tenant_id: str) -> bool:
        """检查是否被速率限制"""
        limit = self.get_limit(tenant_id)
        now = time.time()
        window_start = now - 60  # 1 分钟窗口

        with self._lock:
            # 清理过期记录
            self.request_counts[tenant_id] = [
                t for t in self.request_counts[tenant_id] if t > window_start
            ]
            count = len(self.request_counts[tenant_id])

        RATE_LIMIT_REMAINING.labels(tenant_id=tenant_id).set(
            max(0, limit.requests_per_minute - count)
        )

        return count >= limit.requests_per_minute

    def _record_request(self, tenant_id: str):
        """记录请求时间"""
        with self._lock:
            self.request_counts[tenant_id].append(time.time())

    def reset_circuit(self, tenant_id: str):
        """手动重置熔断器"""
        self.circuit_states[tenant_id] = CircuitState.CLOSED
        if tenant_id in self.circuit_open_time:
            del self.circuit_open_time[tenant_id]
        CIRCUIT_STATE.labels(tenant_id=tenant_id).set(0)


# ============================================
# 全局实例
# ============================================

_controller: Optional[CostController] = None


def get_cost_controller() -> CostController:
    """获取成本控制器单例"""
    global _controller
    if _controller is None:
        _controller = CostController()
    return _controller


# ============================================
# 装饰器
# ============================================

def cost_controlled(tenant_id: str = "default"):
    """
    成本控制装饰器

    用于包装 LLM 调用函数，自动进行成本检查和记录

    Usage:
        @cost_controlled(tenant_id="user_123")
        async def my_llm_call():
            ...
    """
    def decorator(func: Callable) -> Callable:
        async def wrapper(*args, **kwargs) -> Any:
            controller = get_cost_controller()

            # 检查是否允许
            allowed, reason = controller.check_allowed(tenant_id)
            if not allowed:
                raise CostLimitExceeded(reason)

            # 执行函数
            result = await func(*args, **kwargs)

            return result

        return wrapper
    return decorator


class CostLimitExceeded(Exception):
    """成本限制超出异常"""
    pass


# ============================================
# 导出
# ============================================

__all__ = [
    "CostController",
    "CostLimit",
    "CostStatus",
    "CircuitState",
    "TokenPricing",
    "UsageRecord",
    "get_cost_controller",
    "cost_controlled",
    "CostLimitExceeded",
    "estimate_cost",
    "get_pricing",
    "MODEL_PRICING",
]
