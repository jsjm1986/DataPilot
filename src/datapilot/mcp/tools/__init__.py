# -*- coding: utf-8 -*-
"""
MCP Tools Implementation

所有工具都继承自 MCPTool 基类，符合 MCP 标准
"""

# 导入工具类
from .list_tables import ListTablesTool, list_tables
from .get_ddl import GetDDLTool, get_ddl
from .search_values import SearchValuesTool, search_values
from .read_only_query import ReadOnlyQueryTool, read_only_query
from .plan_explain import PlanExplainTool, plan_explain

# 导入基类 (方便外部使用)
from ..base import (
    MCPTool,
    ToolCategory,
    ToolParameter,
    ToolSchema,
    ToolResult,
    ToolRegistry,
    get_tool_registry,
    mcp_tool,
)

__all__ = [
    # 工具类
    "ListTablesTool",
    "GetDDLTool",
    "SearchValuesTool",
    "ReadOnlyQueryTool",
    "PlanExplainTool",
    # 兼容旧接口的函数
    "list_tables",
    "get_ddl",
    "search_values",
    "read_only_query",
    "plan_explain",
    # 基类和工具
    "MCPTool",
    "ToolCategory",
    "ToolParameter",
    "ToolSchema",
    "ToolResult",
    "ToolRegistry",
    "get_tool_registry",
    "mcp_tool",
]
