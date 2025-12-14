# -*- coding: utf-8 -*-
"""
MCP Tool 基类

实现 Model Context Protocol (MCP) 标准的工具基类

功能:
1. 统一的工具接口定义
2. JSON Schema 输入验证
3. 执行指标追踪
4. 错误处理和日志
5. 工具注册和发现
6. 超时和重试支持 (使用配置)
"""

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional, TypeVar, Generic
from functools import wraps

from prometheus_client import Counter, Histogram

from ..config.settings import get_settings


# ============================================
# Prometheus 指标
# ============================================

MCP_TOOL_CALLS = Counter(
    "datapilot_mcp_tool_calls_total",
    "Total MCP tool calls",
    labelnames=["tool_name", "status"],
)

MCP_TOOL_LATENCY = Histogram(
    "datapilot_mcp_tool_latency_seconds",
    "MCP tool execution latency",
    labelnames=["tool_name"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)


# ============================================
# 数据结构
# ============================================

class ToolCategory(Enum):
    """工具类别"""
    DATABASE = "database"      # 数据库操作
    SCHEMA = "schema"          # Schema 相关
    QUERY = "query"            # 查询执行
    ANALYSIS = "analysis"      # 分析工具
    UTILITY = "utility"        # 通用工具


@dataclass
class ToolParameter:
    """工具参数定义"""
    name: str
    type: str  # string, integer, number, boolean, array, object
    description: str
    required: bool = True
    default: Any = None
    enum: Optional[list] = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None


@dataclass
class ToolSchema:
    """工具 Schema 定义 (符合 JSON Schema)"""
    parameters: list[ToolParameter] = field(default_factory=list)

    def to_json_schema(self) -> dict:
        """转换为 JSON Schema 格式"""
        properties = {}
        required = []

        for param in self.parameters:
            prop = {
                "type": param.type,
                "description": param.description,
            }
            if param.enum:
                prop["enum"] = param.enum
            if param.minimum is not None:
                prop["minimum"] = param.minimum
            if param.maximum is not None:
                prop["maximum"] = param.maximum
            if param.default is not None:
                prop["default"] = param.default

            properties[param.name] = prop

            if param.required:
                required.append(param.name)

        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }


@dataclass
class ToolResult:
    """工具执行结果"""
    success: bool
    data: Any = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """转换为字典"""
        result = {
            "success": self.success,
            "execution_time_ms": self.execution_time_ms,
        }
        if self.success:
            result["data"] = self.data
        else:
            result["error"] = self.error
        if self.metadata:
            result["metadata"] = self.metadata
        return result


# ============================================
# MCP Tool 基类
# ============================================

class MCPTool(ABC):
    """
    MCP Tool 基类

    所有 MCP 工具都应继承此类并实现:
    - name: 工具名称
    - description: 工具描述
    - category: 工具类别
    - schema: 输入参数 Schema
    - _execute: 实际执行逻辑
    """

    def __init__(self):
        self._call_count = 0
        self._total_time = 0.0
        self._last_called: Optional[datetime] = None

    @property
    @abstractmethod
    def name(self) -> str:
        """工具名称 (唯一标识)"""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """工具描述"""
        pass

    @property
    def category(self) -> ToolCategory:
        """工具类别"""
        return ToolCategory.UTILITY

    @property
    def schema(self) -> ToolSchema:
        """输入参数 Schema"""
        return ToolSchema()

    @abstractmethod
    async def _execute(self, **kwargs) -> Any:
        """
        实际执行逻辑 (子类实现)

        Args:
            **kwargs: 工具参数

        Returns:
            执行结果数据
        """
        pass

    async def execute(self, **kwargs) -> ToolResult:
        """
        执行工具 (带指标追踪、超时、重试和错误处理)

        Args:
            **kwargs: 工具参数

        Returns:
            ToolResult 执行结果
        """
        settings = get_settings()
        timeout_seconds = settings.mcp_timeout_seconds
        max_retries = settings.mcp_retries

        start_time = time.perf_counter()
        status = "success"
        last_error = None

        try:
            # 参数验证
            validation_error = self._validate_params(kwargs)
            if validation_error:
                status = "validation_error"
                return ToolResult(
                    success=False,
                    error=validation_error,
                    execution_time_ms=0,
                )

            # 带重试的执行
            for attempt in range(max_retries + 1):
                try:
                    # 带超时的执行
                    result = await asyncio.wait_for(
                        self._execute(**kwargs),
                        timeout=timeout_seconds
                    )

                    execution_time = (time.perf_counter() - start_time) * 1000

                    # 更新统计
                    self._call_count += 1
                    self._total_time += execution_time
                    self._last_called = datetime.utcnow()

                    return ToolResult(
                        success=True,
                        data=result,
                        execution_time_ms=execution_time,
                        metadata={
                            "tool_name": self.name,
                            "call_count": self._call_count,
                            "attempts": attempt + 1,
                        },
                    )

                except asyncio.TimeoutError:
                    last_error = f"Tool execution timed out after {timeout_seconds}s"
                    status = "timeout"
                    if attempt < max_retries:
                        continue  # 重试
                    break

                except Exception as e:
                    last_error = str(e)
                    status = "error"
                    if attempt < max_retries:
                        continue  # 重试
                    break

            # 所有重试都失败
            execution_time = (time.perf_counter() - start_time) * 1000
            return ToolResult(
                success=False,
                error=last_error,
                execution_time_ms=execution_time,
                metadata={
                    "tool_name": self.name,
                    "attempts": max_retries + 1,
                },
            )

        except Exception as e:
            status = "error"
            execution_time = (time.perf_counter() - start_time) * 1000

            return ToolResult(
                success=False,
                error=str(e),
                execution_time_ms=execution_time,
                metadata={"tool_name": self.name},
            )

        finally:
            # 记录指标
            MCP_TOOL_CALLS.labels(tool_name=self.name, status=status).inc()
            MCP_TOOL_LATENCY.labels(tool_name=self.name).observe(
                (time.perf_counter() - start_time)
            )

    def _validate_params(self, params: dict) -> Optional[str]:
        """验证参数"""
        schema = self.schema

        # 检查必需参数
        for param in schema.parameters:
            if param.required and param.name not in params:
                if param.default is None:
                    return f"Missing required parameter: {param.name}"

            # 类型检查 (简单版本)
            if param.name in params:
                value = params[param.name]
                if not self._check_type(value, param.type):
                    return f"Invalid type for {param.name}: expected {param.type}"

                # 范围检查
                if param.minimum is not None and isinstance(value, (int, float)):
                    if value < param.minimum:
                        return f"{param.name} must be >= {param.minimum}"
                if param.maximum is not None and isinstance(value, (int, float)):
                    if value > param.maximum:
                        return f"{param.name} must be <= {param.maximum}"

                # 枚举检查
                if param.enum and value not in param.enum:
                    return f"{param.name} must be one of {param.enum}"

        return None

    def _check_type(self, value: Any, expected_type: str) -> bool:
        """检查值类型"""
        type_map = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict,
        }
        expected = type_map.get(expected_type)
        if expected is None:
            return True  # 未知类型，跳过检查
        if value is None:
            return True  # None 值单独处理
        return isinstance(value, expected)

    def to_mcp_definition(self) -> dict:
        """
        转换为 MCP 标准工具定义

        Returns:
            MCP 工具定义字典
        """
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.schema.to_json_schema(),
        }

    def get_stats(self) -> dict:
        """获取工具统计信息"""
        return {
            "name": self.name,
            "call_count": self._call_count,
            "total_time_ms": self._total_time,
            "avg_time_ms": self._total_time / self._call_count if self._call_count > 0 else 0,
            "last_called": self._last_called.isoformat() if self._last_called else None,
        }


# ============================================
# 工具注册表
# ============================================

class ToolRegistry:
    """
    MCP 工具注册表

    管理所有注册的 MCP 工具
    """

    _instance: Optional["ToolRegistry"] = None
    _tools: dict[str, MCPTool]

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._tools = {}
        return cls._instance

    def register(self, tool: MCPTool):
        """注册工具"""
        self._tools[tool.name] = tool

    def unregister(self, name: str):
        """注销工具"""
        if name in self._tools:
            del self._tools[name]

    def get(self, name: str) -> Optional[MCPTool]:
        """获取工具"""
        return self._tools.get(name)

    def list_tools(self) -> list[MCPTool]:
        """列出所有工具"""
        return list(self._tools.values())

    def list_by_category(self, category: ToolCategory) -> list[MCPTool]:
        """按类别列出工具"""
        return [t for t in self._tools.values() if t.category == category]

    def get_mcp_definitions(self) -> list[dict]:
        """获取所有工具的 MCP 定义"""
        return [t.to_mcp_definition() for t in self._tools.values()]

    def get_all_stats(self) -> list[dict]:
        """获取所有工具的统计信息"""
        return [t.get_stats() for t in self._tools.values()]


def get_tool_registry() -> ToolRegistry:
    """获取工具注册表单例"""
    return ToolRegistry()


# ============================================
# 装饰器 (简化工具创建)
# ============================================

def mcp_tool(
    name: str,
    description: str,
    category: ToolCategory = ToolCategory.UTILITY,
    parameters: Optional[list[ToolParameter]] = None,
):
    """
    MCP 工具装饰器

    将普通异步函数转换为 MCP 工具

    Usage:
        @mcp_tool(
            name="my_tool",
            description="My tool description",
            parameters=[
                ToolParameter(name="arg1", type="string", description="Argument 1"),
            ]
        )
        async def my_tool(arg1: str) -> dict:
            return {"result": arg1}
    """
    def decorator(func: Callable) -> MCPTool:
        class DecoratedTool(MCPTool):
            @property
            def name(self) -> str:
                return name

            @property
            def description(self) -> str:
                return description

            @property
            def category(self) -> ToolCategory:
                return category

            @property
            def schema(self) -> ToolSchema:
                return ToolSchema(parameters=parameters or [])

            async def _execute(self, **kwargs) -> Any:
                return await func(**kwargs)

        tool = DecoratedTool()

        # 自动注册到注册表
        get_tool_registry().register(tool)

        return tool

    return decorator


# ============================================
# 导出
# ============================================

__all__ = [
    "MCPTool",
    "ToolCategory",
    "ToolParameter",
    "ToolSchema",
    "ToolResult",
    "ToolRegistry",
    "get_tool_registry",
    "mcp_tool",
    "MCP_TOOL_CALLS",
    "MCP_TOOL_LATENCY",
]
