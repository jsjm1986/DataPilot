# -*- coding: utf-8 -*-
"""
Plan explain MCP tool

增强版执行计划分析工具，支持：
- SQLite EXPLAIN QUERY PLAN 完整解析
- MySQL EXPLAIN 完整解析 (包括 JSON 格式)
- PostgreSQL EXPLAIN (FORMAT JSON) 完整解析
- 成本估算和性能风险识别
"""

import json
import re
from typing import Any, Optional
from dataclasses import dataclass, field, asdict

from ..base import MCPTool, ToolCategory, ToolParameter, ToolSchema, get_tool_registry
from ...db.connector import get_db_manager


@dataclass
class ExplainSummary:
    """执行计划摘要"""
    # 通用字段
    scan_type: str = "unknown"  # table_scan, index_scan, index_seek, etc.
    uses_index: bool = False
    tables_accessed: list = field(default_factory=list)
    estimated_rows: int = 0
    estimated_cost: float = 0.0

    # MySQL 特定
    access_type: str = ""  # ALL, index, range, ref, eq_ref, const
    rows_examined: int = 0
    extra: str = ""
    key: str = ""
    key_len: str = ""
    possible_keys: list = field(default_factory=list)

    # PostgreSQL 特定
    node_type: str = ""  # Seq Scan, Index Scan, Hash Join, etc.
    total_cost: float = 0.0
    startup_cost: float = 0.0
    plan_rows: int = 0
    plan_width: int = 0
    actual_rows: Optional[int] = None  # EXPLAIN ANALYZE
    actual_time: Optional[float] = None  # EXPLAIN ANALYZE

    # SQLite 特定
    detail: str = ""
    using_index: str = ""

    # 风险和建议
    risks: list = field(default_factory=list)
    recommendations: list = field(default_factory=list)

    def to_dict(self) -> dict:
        """转换为字典"""
        return asdict(self)


class PlanExplainTool(MCPTool):
    """
    SQL 执行计划分析工具

    MCP 标准实现，支持完整的执行计划解析
    """

    @property
    def name(self) -> str:
        return "plan_explain"

    @property
    def description(self) -> str:
        return "获取 SQL 执行计划，分析查询性能，支持 SQLite/MySQL/PostgreSQL"

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.ANALYSIS

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(parameters=[
            ToolParameter(
                name="sql",
                type="string",
                description="SQL 查询语句",
                required=True,
            ),
            ToolParameter(
                name="database",
                type="string",
                description="数据库名称",
                required=False,
            ),
            ToolParameter(
                name="analyze",
                type="boolean",
                description="是否使用 EXPLAIN ANALYZE (仅 PostgreSQL)",
                required=False,
            ),
        ])

    async def _execute(
        self,
        sql: str,
        database: Optional[str] = None,
        analyze: bool = False,
    ) -> dict:
        """执行计划分析"""
        db_manager = get_db_manager()
        connector = db_manager.get(database)

        # 根据数据库类型构建 EXPLAIN 语句
        db_type = connector.db_type

        if db_type == "sqlite":
            explain_sql = f"EXPLAIN QUERY PLAN {sql}"
        elif db_type == "mysql":
            # MySQL 8.0+ 支持 JSON 格式
            explain_sql = f"EXPLAIN FORMAT=JSON {sql}"
        elif db_type == "postgresql":
            if analyze:
                explain_sql = f"EXPLAIN (ANALYZE, FORMAT JSON) {sql}"
            else:
                explain_sql = f"EXPLAIN (FORMAT JSON) {sql}"
        elif db_type == "sqlserver":
            # SQL Server 需要单独执行 SET SHOWPLAN_TEXT
            # 这里标记为特殊处理，在下面的 try 块中处理
            explain_sql = sql  # 占位，实际在下面特殊处理
        elif db_type == "clickhouse":
            # ClickHouse 使用 EXPLAIN
            explain_sql = f"EXPLAIN {sql}"
        elif db_type == "duckdb":
            # DuckDB 使用 EXPLAIN
            explain_sql = f"EXPLAIN {sql}"
        else:
            explain_sql = f"EXPLAIN {sql}"

        try:
            if db_type == "sqlserver":
                # SQL Server 特殊处理：需要分开执行 SET SHOWPLAN_TEXT
                try:
                    # 先开启 SHOWPLAN
                    await connector.execute_query("SET SHOWPLAN_TEXT ON")
                    # 执行查询获取执行计划 (不会实际执行，只返回计划)
                    data = await connector.execute_query(sql)
                    # 关闭 SHOWPLAN
                    await connector.execute_query("SET SHOWPLAN_TEXT OFF")
                except Exception:
                    # 如果失败，返回空计划并提供基本分析
                    data = []
            else:
                data = await connector.execute_query(explain_sql)
        except Exception as e:
            # 如果 JSON 格式失败，尝试普通格式
            if db_type == "mysql":
                explain_sql = f"EXPLAIN {sql}"
                data = await connector.execute_query(explain_sql)
            elif db_type == "sqlserver":
                # SQL Server 可能不支持 SHOWPLAN，返回空计划
                data = []
            else:
                raise

        # 解析执行计划
        plan_info = {
            "database": database or "default",
            "db_type": db_type,
            "sql": sql,
            "success": True,
            "plan": data,
        }

        # 根据数据库类型解析
        if db_type == "sqlite" and data:
            plan_info["summary"] = self._parse_sqlite_plan(data)
        elif db_type == "mysql" and data:
            plan_info["summary"] = self._parse_mysql_plan(data)
        elif db_type == "postgresql" and data:
            plan_info["summary"] = self._parse_postgres_plan(data)
        elif db_type == "sqlserver" and data:
            plan_info["summary"] = self._parse_sqlserver_plan(data)
        elif db_type == "clickhouse" and data:
            plan_info["summary"] = self._parse_clickhouse_plan(data)
        elif db_type == "duckdb" and data:
            plan_info["summary"] = self._parse_duckdb_plan(data)

        return plan_info

    def _parse_sqlite_plan(self, plan_data: list) -> dict:
        """
        解析 SQLite EXPLAIN QUERY PLAN

        SQLite 输出格式:
        - id, parent, notused, detail
        - detail 包含: SCAN TABLE xxx, SEARCH TABLE xxx USING INDEX xxx, etc.
        """
        summary = ExplainSummary()
        summary.tables_accessed = []
        summary.risks = []
        summary.recommendations = []

        all_details = []

        for row in plan_data:
            detail = str(row.get("detail", ""))
            detail_lower = detail.lower()
            all_details.append(detail)

            # 解析扫描类型
            if "scan table" in detail_lower:
                if "using covering index" in detail_lower:
                    summary.scan_type = "covering_index_scan"
                    summary.uses_index = True
                elif "using index" in detail_lower:
                    summary.scan_type = "index_scan"
                    summary.uses_index = True
                else:
                    summary.scan_type = "table_scan"
                    summary.risks.append("Full table scan detected")
                    summary.recommendations.append("Consider adding an index on filtered columns")

            elif "search table" in detail_lower:
                summary.uses_index = True
                if "using integer primary key" in detail_lower:
                    summary.scan_type = "pk_lookup"
                elif "using index" in detail_lower:
                    summary.scan_type = "index_seek"
                else:
                    summary.scan_type = "index_scan"

            # 提取索引名
            index_match = re.search(r'using (?:covering )?index (\w+)', detail_lower)
            if index_match:
                summary.using_index = index_match.group(1)

            # 提取表名
            table_match = re.search(r'(?:scan|search) table (\w+)', detail_lower)
            if table_match:
                table_name = table_match.group(1)
                if table_name not in summary.tables_accessed:
                    summary.tables_accessed.append(table_name)

            # 检测临时 B-Tree (排序/分组)
            if "use temp b-tree" in detail_lower:
                if "order by" in detail_lower:
                    summary.risks.append("Using temp B-tree for ORDER BY")
                    summary.recommendations.append("Add index on ORDER BY columns")
                elif "group by" in detail_lower:
                    summary.risks.append("Using temp B-tree for GROUP BY")
                    summary.recommendations.append("Add index on GROUP BY columns")
                elif "distinct" in detail_lower:
                    summary.risks.append("Using temp B-tree for DISTINCT")

            # 检测自动索引
            if "automatic index" in detail_lower or "auto-index" in detail_lower:
                summary.risks.append("SQLite creating automatic index at runtime")
                summary.recommendations.append("Create permanent index to avoid runtime overhead")

            # 检测子查询
            if "correlated" in detail_lower:
                summary.risks.append("Correlated subquery detected")
                summary.recommendations.append("Consider rewriting as JOIN")

            # 检测 COMPOUND 查询
            if "compound" in detail_lower:
                summary.risks.append("Compound query (UNION/INTERSECT/EXCEPT)")

        summary.detail = "; ".join(all_details)

        return summary.to_dict()

    def _parse_mysql_plan(self, plan_data: list) -> dict:
        """
        解析 MySQL EXPLAIN

        支持两种格式:
        1. JSON 格式 (MySQL 5.7+): EXPLAIN FORMAT=JSON
        2. 表格格式: 传统 EXPLAIN
        """
        summary = ExplainSummary()
        summary.tables_accessed = []
        summary.possible_keys = []
        summary.risks = []
        summary.recommendations = []

        # 检查是否是 JSON 格式
        if plan_data and isinstance(plan_data, list):
            first_row = plan_data[0]

            # JSON 格式返回 {"EXPLAIN": "..."}
            if "EXPLAIN" in first_row:
                try:
                    json_plan = json.loads(first_row["EXPLAIN"])
                    return self._parse_mysql_json_plan(json_plan)
                except (json.JSONDecodeError, TypeError):
                    pass

            # 表格格式
            return self._parse_mysql_table_plan(plan_data)

        return summary.to_dict()

    def _parse_mysql_json_plan(self, json_plan: dict) -> dict:
        """解析 MySQL JSON 格式的 EXPLAIN"""
        summary = ExplainSummary()
        summary.tables_accessed = []
        summary.possible_keys = []
        summary.risks = []
        summary.recommendations = []

        query_block = json_plan.get("query_block", {})

        # 提取成本信息
        cost_info = query_block.get("cost_info", {})
        summary.estimated_cost = float(cost_info.get("query_cost", 0))

        # 递归解析查询块
        self._parse_mysql_query_block(query_block, summary)

        return summary.to_dict()

    def _parse_mysql_query_block(self, block: dict, summary: ExplainSummary):
        """递归解析 MySQL 查询块"""
        # 解析表信息
        table = block.get("table", {})
        if table:
            table_name = table.get("table_name", "")
            if table_name and table_name not in summary.tables_accessed:
                summary.tables_accessed.append(table_name)

            # 访问类型
            access_type = table.get("access_type", "")
            if not summary.access_type or access_type == "ALL":
                summary.access_type = access_type

            # 行数
            rows = table.get("rows_examined_per_scan", 0)
            summary.rows_examined += rows
            summary.estimated_rows += rows

            # 索引信息
            key = table.get("key", "")
            if key:
                summary.uses_index = True
                summary.key = key

            possible_keys = table.get("possible_keys", [])
            summary.possible_keys.extend(possible_keys)

            # 成本
            cost_info = table.get("cost_info", {})
            read_cost = float(cost_info.get("read_cost", 0))
            eval_cost = float(cost_info.get("eval_cost", 0))
            summary.estimated_cost += read_cost + eval_cost

            # 检测风险
            if access_type == "ALL":
                summary.scan_type = "table_scan"
                summary.risks.append(f"Full table scan on {table_name}")
                summary.recommendations.append("Add index on filtered columns")
            elif access_type == "index":
                summary.scan_type = "index_scan"
                summary.risks.append(f"Full index scan on {table_name}")
            elif access_type in ("range", "ref", "eq_ref", "const"):
                summary.scan_type = "index_seek"

            # 检查 filesort 和 temporary
            if table.get("using_filesort"):
                summary.risks.append("Using filesort")
                summary.recommendations.append("Add index on ORDER BY columns")
            if table.get("using_temporary_table"):
                summary.risks.append("Using temporary table")
                summary.recommendations.append("Optimize GROUP BY or add covering index")

        # 递归处理嵌套查询
        nested_loop = block.get("nested_loop", [])
        for nested in nested_loop:
            self._parse_mysql_query_block(nested, summary)

        ordering_operation = block.get("ordering_operation", {})
        if ordering_operation:
            if ordering_operation.get("using_filesort"):
                summary.risks.append("Using filesort for ORDER BY")
            nested = ordering_operation.get("nested_loop", [])
            for n in nested:
                self._parse_mysql_query_block(n, summary)

        grouping_operation = block.get("grouping_operation", {})
        if grouping_operation:
            if grouping_operation.get("using_temporary_table"):
                summary.risks.append("Using temporary table for GROUP BY")
            nested = grouping_operation.get("nested_loop", [])
            for n in nested:
                self._parse_mysql_query_block(n, summary)

    def _parse_mysql_table_plan(self, plan_data: list) -> dict:
        """解析 MySQL 表格格式的 EXPLAIN"""
        summary = ExplainSummary()
        summary.tables_accessed = []
        summary.possible_keys = []
        summary.risks = []
        summary.recommendations = []

        for row in plan_data:
            # 表名
            table = row.get("table", "")
            if table and table not in summary.tables_accessed:
                summary.tables_accessed.append(table)

            # 访问类型 (type 字段)
            access_type = row.get("type", "")
            if not summary.access_type or access_type == "ALL":
                summary.access_type = access_type

            # 行数
            rows = row.get("rows", 0) or 0
            summary.rows_examined += rows
            summary.estimated_rows += rows

            # 索引
            key = row.get("key", "")
            if key:
                summary.uses_index = True
                summary.key = key

            key_len = row.get("key_len", "")
            if key_len:
                summary.key_len = key_len

            possible_keys = row.get("possible_keys", "")
            if possible_keys:
                summary.possible_keys.extend(possible_keys.split(","))

            # Extra 字段
            extra = row.get("Extra", "") or row.get("extra", "") or ""
            summary.extra = extra

            # 检测风险
            if access_type == "ALL":
                summary.scan_type = "table_scan"
                summary.risks.append(f"Full table scan on {table} (type=ALL)")
                summary.recommendations.append("Add WHERE clause or create index")
            elif access_type == "index":
                summary.scan_type = "index_scan"
                summary.risks.append(f"Full index scan on {table} (type=index)")
            elif access_type in ("range", "ref", "eq_ref", "const", "system"):
                summary.scan_type = "index_seek"

            # Extra 字段分析
            extra_lower = extra.lower()
            if "using filesort" in extra_lower:
                summary.risks.append("Using filesort")
                summary.recommendations.append("Add index on ORDER BY columns")
            if "using temporary" in extra_lower:
                summary.risks.append("Using temporary table")
                summary.recommendations.append("Optimize GROUP BY or add covering index")
            if "using where" not in extra_lower and access_type in ("ALL", "index"):
                summary.risks.append("No WHERE filtering on full scan")
            if "using index" in extra_lower:
                summary.uses_index = True
            if "using join buffer" in extra_lower:
                summary.risks.append("Using join buffer - may indicate missing index")

        return summary.to_dict()

    def _parse_postgres_plan(self, plan_data: list) -> dict:
        """
        解析 PostgreSQL EXPLAIN (FORMAT JSON)

        JSON 格式返回嵌套的计划树
        """
        summary = ExplainSummary()
        summary.tables_accessed = []
        summary.risks = []
        summary.recommendations = []

        if not plan_data:
            return summary.to_dict()

        # PostgreSQL JSON EXPLAIN 返回 [{"Plan": {...}}]
        try:
            if isinstance(plan_data, list) and plan_data:
                first_row = plan_data[0]
                # 可能是 {"QUERY PLAN": "..."} 或直接是 plan 对象
                if "QUERY PLAN" in first_row:
                    plan_json = first_row["QUERY PLAN"]
                    if isinstance(plan_json, str):
                        plan_obj = json.loads(plan_json)
                    else:
                        plan_obj = plan_json
                elif "Plan" in first_row:
                    plan_obj = first_row
                else:
                    # 尝试直接解析
                    plan_obj = first_row

                if isinstance(plan_obj, list) and plan_obj:
                    plan_obj = plan_obj[0]

                plan = plan_obj.get("Plan", plan_obj)
                self._parse_postgres_plan_node(plan, summary)

        except (json.JSONDecodeError, TypeError, KeyError) as e:
            summary.risks.append(f"Failed to parse plan: {str(e)}")

        return summary.to_dict()

    def _parse_postgres_plan_node(self, node: dict, summary: ExplainSummary, depth: int = 0):
        """递归解析 PostgreSQL 计划节点"""
        if not isinstance(node, dict):
            return

        node_type = node.get("Node Type", "")

        # 记录根节点信息
        if depth == 0:
            summary.node_type = node_type
            summary.total_cost = node.get("Total Cost", 0)
            summary.startup_cost = node.get("Startup Cost", 0)
            summary.plan_rows = node.get("Plan Rows", 0)
            summary.plan_width = node.get("Plan Width", 0)
            summary.estimated_rows = summary.plan_rows
            summary.estimated_cost = summary.total_cost

            # EXPLAIN ANALYZE 数据
            if "Actual Rows" in node:
                summary.actual_rows = node.get("Actual Rows")
            if "Actual Total Time" in node:
                summary.actual_time = node.get("Actual Total Time")

        # 提取表名
        relation_name = node.get("Relation Name", "")
        if relation_name and relation_name not in summary.tables_accessed:
            summary.tables_accessed.append(relation_name)

        alias = node.get("Alias", "")
        if alias and alias != relation_name and alias not in summary.tables_accessed:
            summary.tables_accessed.append(alias)

        # 分析节点类型
        if node_type == "Seq Scan":
            summary.scan_type = "table_scan"
            rows = node.get("Plan Rows", 0)
            table = relation_name or alias
            if rows > 10000:
                summary.risks.append(f"Sequential scan on {table} ({rows:,} rows)")
                summary.recommendations.append(f"Add index on {table} for filtered columns")
            elif rows > 1000:
                summary.risks.append(f"Sequential scan on {table}")

        elif node_type == "Index Scan":
            summary.scan_type = "index_seek"
            summary.uses_index = True
            index_name = node.get("Index Name", "")
            if index_name:
                summary.using_index = index_name

        elif node_type == "Index Only Scan":
            summary.scan_type = "covering_index_scan"
            summary.uses_index = True
            index_name = node.get("Index Name", "")
            if index_name:
                summary.using_index = index_name

        elif node_type == "Bitmap Heap Scan":
            summary.scan_type = "bitmap_scan"
            summary.uses_index = True

        elif node_type == "Bitmap Index Scan":
            summary.uses_index = True
            index_name = node.get("Index Name", "")
            if index_name:
                summary.using_index = index_name

        elif node_type == "Nested Loop":
            rows = node.get("Plan Rows", 0)
            if rows > 100000:
                summary.risks.append(f"Nested loop on large dataset ({rows:,} rows)")
                summary.recommendations.append("Consider using hash join or adding index")

        elif node_type == "Hash Join":
            rows = node.get("Plan Rows", 0)
            if rows > 1000000:
                summary.risks.append(f"Large hash join ({rows:,} rows)")
                summary.recommendations.append("Ensure sufficient work_mem")

        elif node_type == "Merge Join":
            # Merge join 通常效率较高
            pass

        elif node_type == "Sort":
            rows = node.get("Plan Rows", 0)
            if rows > 100000:
                summary.risks.append(f"Sorting large result set ({rows:,} rows)")
                summary.recommendations.append("Add index on ORDER BY columns")

        elif node_type == "Aggregate":
            strategy = node.get("Strategy", "")
            if strategy == "Hashed":
                rows = node.get("Plan Rows", 0)
                if rows > 1000000:
                    summary.risks.append("Large hash aggregate")
                    summary.recommendations.append("Ensure sufficient work_mem")

        elif node_type == "Hash":
            # Hash 节点用于 Hash Join
            pass

        elif node_type == "Materialize":
            summary.risks.append("Materializing subquery result")

        elif node_type == "Subquery Scan":
            summary.risks.append("Subquery scan detected")

        # 检查行估计准确性 (EXPLAIN ANALYZE)
        if "Actual Rows" in node and "Plan Rows" in node:
            actual = node["Actual Rows"]
            planned = node["Plan Rows"]
            if planned > 0:
                ratio = actual / planned
                if ratio > 10 or ratio < 0.1:
                    summary.risks.append(
                        f"Row estimate off by {ratio:.1f}x at {node_type} "
                        f"(planned: {planned}, actual: {actual})"
                    )
                    summary.recommendations.append("Run ANALYZE on affected tables")

        # 检查实际执行时间
        if "Actual Total Time" in node:
            actual_time = node["Actual Total Time"]
            if actual_time > 5000:  # 5 秒
                summary.risks.append(f"Slow node: {node_type} took {actual_time:.0f}ms")

        # 递归处理子节点
        plans = node.get("Plans", [])
        for child in plans:
            self._parse_postgres_plan_node(child, summary, depth + 1)

    def _parse_sqlserver_plan(self, plan_data: list) -> dict:
        """
        解析 SQL Server SHOWPLAN_TEXT 输出
        """
        summary = ExplainSummary()
        summary.tables_accessed = []
        summary.risks = []
        summary.recommendations = []

        for row in plan_data:
            plan_text = str(row).lower()

            # 检测扫描类型
            if "table scan" in plan_text:
                summary.scan_type = "table_scan"
                summary.risks.append("Full table scan detected")
                summary.recommendations.append("Add index on filtered columns")
            elif "clustered index scan" in plan_text:
                summary.scan_type = "index_scan"
                summary.uses_index = True
                summary.risks.append("Clustered index scan (full)")
            elif "index scan" in plan_text:
                summary.scan_type = "index_scan"
                summary.uses_index = True
            elif "index seek" in plan_text:
                summary.scan_type = "index_seek"
                summary.uses_index = True
            elif "clustered index seek" in plan_text:
                summary.scan_type = "index_seek"
                summary.uses_index = True

            # 检测其他操作
            if "sort" in plan_text:
                summary.risks.append("Sort operation detected")
            if "hash match" in plan_text:
                summary.risks.append("Hash match (join or aggregate)")
            if "nested loops" in plan_text:
                summary.risks.append("Nested loops join")
            if "key lookup" in plan_text:
                summary.risks.append("Key lookup - consider covering index")
                summary.recommendations.append("Add columns to index to avoid key lookup")

        return summary.to_dict()

    def _parse_clickhouse_plan(self, plan_data: list) -> dict:
        """
        解析 ClickHouse EXPLAIN 输出
        """
        summary = ExplainSummary()
        summary.tables_accessed = []
        summary.risks = []
        summary.recommendations = []

        for row in plan_data:
            plan_text = str(row).lower()

            # 检测读取类型
            if "readfrommergetree" in plan_text:
                summary.scan_type = "mergetree_read"
                summary.uses_index = True
            elif "readfromstorage" in plan_text:
                summary.scan_type = "storage_read"
            elif "expression" in plan_text:
                summary.scan_type = "expression"

            # 检测操作
            if "aggregating" in plan_text:
                summary.risks.append("Aggregation operation")
            if "sorting" in plan_text:
                summary.risks.append("Sorting operation")
            if "join" in plan_text:
                summary.risks.append("Join operation")
            if "filter" in plan_text:
                # Filter 通常是好的
                pass

            # 提取表名
            table_match = re.search(r'table:\s*(\w+)', plan_text)
            if table_match:
                table_name = table_match.group(1)
                if table_name not in summary.tables_accessed:
                    summary.tables_accessed.append(table_name)

        return summary.to_dict()

    def _parse_duckdb_plan(self, plan_data: list) -> dict:
        """
        解析 DuckDB EXPLAIN 输出
        DuckDB 输出格式类似 PostgreSQL
        """
        summary = ExplainSummary()
        summary.tables_accessed = []
        summary.risks = []
        summary.recommendations = []

        for row in plan_data:
            plan_text = str(row).lower()

            # 检测扫描类型
            if "seq_scan" in plan_text or "table_scan" in plan_text:
                summary.scan_type = "table_scan"
                summary.risks.append("Sequential scan detected")
                summary.recommendations.append("Add filter or create index")
            elif "index_scan" in plan_text:
                summary.scan_type = "index_scan"
                summary.uses_index = True
            elif "filter" in plan_text:
                summary.scan_type = "filter"

            # 检测操作
            if "hash_join" in plan_text:
                summary.risks.append("Hash join operation")
            if "nested_loop" in plan_text:
                summary.risks.append("Nested loop join")
            if "order_by" in plan_text or "sort" in plan_text:
                summary.risks.append("Sort operation")
            if "aggregate" in plan_text:
                summary.risks.append("Aggregation operation")
            if "projection" in plan_text:
                # Projection 是正常的
                pass

            # 提取表名
            table_match = re.search(r'(?:scan|read)\s+(\w+)', plan_text)
            if table_match:
                table_name = table_match.group(1)
                if table_name not in summary.tables_accessed:
                    summary.tables_accessed.append(table_name)

        return summary.to_dict()


# 创建工具实例并注册
_tool_instance = PlanExplainTool()
get_tool_registry().register(_tool_instance)


# 兼容旧接口的函数
async def plan_explain(
    sql: str,
    database: Optional[str] = None,
) -> dict:
    """
    获取 SQL 执行计划 (兼容旧接口)

    Args:
        sql: SQL 查询语句
        database: 数据库名称

    Returns:
        执行计划信息
    """
    result = await _tool_instance.execute(sql=sql, database=database)
    if result.success:
        return result.data
    else:
        return {
            "database": database or "default",
            "sql": sql,
            "success": False,
            "error": result.error,
        }
