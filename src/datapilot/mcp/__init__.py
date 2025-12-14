# -*- coding: utf-8 -*-
"""
MCP Tools Protocol Layer

实现 Model Context Protocol (MCP) 标准的工具层

功能:
1. 统一的工具基类 (MCPTool)
2. JSON Schema 输入验证
3. 执行指标追踪
4. 工具注册和发现
"""

# 导入基类
from .base import (
    MCPTool,
    ToolCategory,
    ToolParameter,
    ToolSchema,
    ToolResult,
    ToolRegistry,
    get_tool_registry,
    mcp_tool,
    MCP_TOOL_CALLS,
    MCP_TOOL_LATENCY,
)

# 导入工具函数 (兼容旧接口)
from .tools import (
    list_tables,
    get_ddl,
    search_values,
    read_only_query,
    plan_explain,
)

# 导入工具类
from .tools import (
    ListTablesTool,
    GetDDLTool,
    SearchValuesTool,
    ReadOnlyQueryTool,
    PlanExplainTool,
)

__all__ = [
    # 基类
    "MCPTool",
    "ToolCategory",
    "ToolParameter",
    "ToolSchema",
    "ToolResult",
    "ToolRegistry",
    "get_tool_registry",
    "mcp_tool",
    # Prometheus 指标
    "MCP_TOOL_CALLS",
    "MCP_TOOL_LATENCY",
    # 工具函数 (兼容旧接口)
    "list_tables",
    "get_ddl",
    "search_values",
    "read_only_query",
    "plan_explain",
    # 工具类
    "ListTablesTool",
    "GetDDLTool",
    "SearchValuesTool",
    "ReadOnlyQueryTool",
    "PlanExplainTool",
]
