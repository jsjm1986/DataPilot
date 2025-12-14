# -*- coding: utf-8 -*-
"""
SQL 成本分析器
分析 SQL 执行计划和成本
"""

from typing import Optional, Any
from ..db.connector import get_db_manager


class CostAnalyzer:
    """
    SQL 成本分析器

    功能:
    1. 获取执行计划 (EXPLAIN)
    2. 解析执行计划
    3. 估算查询成本
    4. 检测性能问题
    """

    def __init__(self, database: str = "default"):
        self.database = database
        self.db_manager = get_db_manager()

    async def get_explain(self, sql: str) -> dict:
        """
        获取 SQL 执行计划

        Args:
            sql: SQL 语句

        Returns:
            执行计划信息
        """
        connector = self.db_manager.get(self.database)
        db_type = connector.db_type

        try:
            if db_type == "mysql":
                return await self._explain_mysql(sql, connector)
            elif db_type == "postgresql":
                return await self._explain_postgres(sql, connector)
            elif db_type == "sqlite":
                return await self._explain_sqlite(sql, connector)
            elif db_type == "sqlserver":
                return await self._explain_sqlserver(sql, connector)
            elif db_type == "clickhouse":
                return await self._explain_clickhouse(sql, connector)
            elif db_type == "duckdb":
                return await self._explain_duckdb(sql, connector)
            else:
                return {"success": False, "error": f"Unsupported database: {db_type}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _explain_mysql(self, sql: str, connector) -> dict:
        """MySQL EXPLAIN"""
        explain_sql = f"EXPLAIN FORMAT=JSON {sql}"
        try:
            result = await connector.execute_query(explain_sql)
            if result:
                import json
                plan_json = result[0].get("EXPLAIN", "{}")
                plan = json.loads(plan_json) if isinstance(plan_json, str) else plan_json

                # 提取关键信息
                query_block = plan.get("query_block", {})
                cost_info = query_block.get("cost_info", {})

                return {
                    "success": True,
                    "plan": plan,
                    "summary": {
                        "cost": float(cost_info.get("query_cost", 0)),
                        "rows_examined": self._extract_mysql_rows(query_block),
                        "scan_type": self._extract_mysql_scan_type(query_block),
                        "estimated_rows": self._extract_mysql_rows(query_block),
                    },
                }
        except Exception as e:
            # 尝试简单 EXPLAIN
            try:
                simple_result = await connector.execute_query(f"EXPLAIN {sql}")
                return {
                    "success": True,
                    "plan": simple_result,
                    "summary": self._parse_simple_mysql_explain(simple_result),
                }
            except Exception as e2:
                return {"success": False, "error": str(e2)}

        return {"success": False, "error": "Failed to get explain"}

    def _extract_mysql_rows(self, query_block: dict) -> int:
        """提取 MySQL 行数估算"""
        if "table" in query_block:
            return query_block["table"].get("rows_examined_per_scan", 0)
        if "nested_loop" in query_block:
            total = 0
            for item in query_block["nested_loop"]:
                if "table" in item:
                    total += item["table"].get("rows_examined_per_scan", 0)
            return total
        return 0

    def _extract_mysql_scan_type(self, query_block: dict) -> str:
        """提取 MySQL 扫描类型"""
        if "table" in query_block:
            access_type = query_block["table"].get("access_type", "")
            if access_type == "ALL":
                return "table_scan"
            return access_type
        return "unknown"

    def _parse_simple_mysql_explain(self, result: list) -> dict:
        """解析简单 EXPLAIN 结果"""
        if not result:
            return {}

        row = result[0]
        return {
            "scan_type": row.get("type", ""),
            "estimated_rows": row.get("rows", 0),
            "cost": 0,
            "rows_examined": row.get("rows", 0),
        }

    async def _explain_postgres(self, sql: str, connector) -> dict:
        """PostgreSQL EXPLAIN"""
        explain_sql = f"EXPLAIN (FORMAT JSON, ANALYZE false) {sql}"
        try:
            result = await connector.execute_query(explain_sql)
            if result:
                plan = result[0].get("QUERY PLAN", [])
                if isinstance(plan, list) and plan:
                    plan_data = plan[0].get("Plan", {})

                    return {
                        "success": True,
                        "plan": plan,
                        "summary": {
                            "total_cost": plan_data.get("Total Cost", 0),
                            "startup_cost": plan_data.get("Startup Cost", 0),
                            "estimated_rows": plan_data.get("Plan Rows", 0),
                            "scan_type": plan_data.get("Node Type", ""),
                        },
                    }
        except Exception as e:
            return {"success": False, "error": str(e)}

        return {"success": False, "error": "Failed to get explain"}

    async def _explain_sqlite(self, sql: str, connector) -> dict:
        """SQLite EXPLAIN QUERY PLAN"""
        explain_sql = f"EXPLAIN QUERY PLAN {sql}"
        try:
            result = await connector.execute_query(explain_sql)

            # 分析执行计划
            scan_type = "unknown"
            estimated_rows = 0

            for row in result:
                detail = row.get("detail", "").lower()
                if "scan" in detail:
                    if "using index" in detail:
                        scan_type = "index_scan"
                    else:
                        scan_type = "table_scan"
                elif "search" in detail:
                    scan_type = "index_search"

            return {
                "success": True,
                "plan": result,
                "summary": {
                    "scan_type": scan_type,
                    "estimated_rows": estimated_rows,
                },
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _explain_sqlserver(self, sql: str, connector) -> dict:
        """SQL Server EXPLAIN

        注意: SQL Server 的 SET SHOWPLAN_TEXT 需要在会话级别设置，
        不能与查询在同一语句中执行。这里使用简化的方式返回基本信息。
        """
        try:
            # 尝试使用 SET SHOWPLAN_TEXT (需要分开执行)
            # 先开启 SHOWPLAN
            await connector.execute_query("SET SHOWPLAN_TEXT ON")
            # 执行查询获取执行计划
            result = await connector.execute_query(sql)
            # 关闭 SHOWPLAN
            await connector.execute_query("SET SHOWPLAN_TEXT OFF")

            scan_type = "unknown"
            estimated_rows = 0

            for row in result:
                plan_text = str(row).lower()
                if "table scan" in plan_text:
                    scan_type = "table_scan"
                elif "clustered index scan" in plan_text:
                    scan_type = "index_scan"
                elif "index scan" in plan_text:
                    scan_type = "index_scan"
                elif "index seek" in plan_text:
                    scan_type = "index_seek"
                elif "clustered index seek" in plan_text:
                    scan_type = "index_seek"

            return {
                "success": True,
                "plan": result,
                "summary": {
                    "scan_type": scan_type,
                    "estimated_rows": estimated_rows,
                },
            }
        except Exception as e:
            # SQL Server 可能不支持 SHOWPLAN 或权限不足，返回基本信息
            return {
                "success": True,
                "plan": [],
                "summary": {
                    "scan_type": "unknown",
                    "estimated_rows": 0,
                    "note": f"SQL Server EXPLAIN not available: {str(e)}",
                },
            }

    async def _explain_clickhouse(self, sql: str, connector) -> dict:
        """ClickHouse EXPLAIN"""
        explain_sql = f"EXPLAIN {sql}"
        try:
            result = await connector.execute_query(explain_sql)

            scan_type = "unknown"
            estimated_rows = 0

            for row in result:
                plan_text = str(row).lower()
                if "readfrommergetree" in plan_text:
                    scan_type = "mergetree_read"
                elif "expression" in plan_text:
                    scan_type = "expression"

            return {
                "success": True,
                "plan": result,
                "summary": {
                    "scan_type": scan_type,
                    "estimated_rows": estimated_rows,
                },
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _explain_duckdb(self, sql: str, connector) -> dict:
        """DuckDB EXPLAIN"""
        explain_sql = f"EXPLAIN {sql}"
        try:
            result = await connector.execute_query(explain_sql)

            scan_type = "unknown"
            estimated_rows = 0

            for row in result:
                plan_text = str(row).lower()
                if "seq_scan" in plan_text:
                    scan_type = "table_scan"
                elif "index_scan" in plan_text:
                    scan_type = "index_scan"
                elif "filter" in plan_text:
                    scan_type = "filter"

            return {
                "success": True,
                "plan": result,
                "summary": {
                    "scan_type": scan_type,
                    "estimated_rows": estimated_rows,
                },
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def analyze(self, sql: str) -> dict:
        """
        完整分析 SQL

        Returns:
            分析结果
        """
        explain = await self.get_explain(sql)

        if not explain.get("success"):
            return explain

        summary = explain.get("summary", {})
        risks = []
        recommendations = []

        # 检测风险
        scan_type = summary.get("scan_type", "")
        if scan_type in ("table_scan", "ALL", "Seq Scan"):
            risks.append("Full table scan detected")
            recommendations.append("Consider adding an index")

        estimated_rows = summary.get("estimated_rows", 0)
        if estimated_rows > 100000:
            risks.append(f"Large result set: {estimated_rows:,} rows")
            recommendations.append("Add LIMIT clause or more filters")

        cost = summary.get("cost", summary.get("total_cost", 0))
        if cost > 100000:
            risks.append(f"High query cost: {cost:,.2f}")
            recommendations.append("Optimize query or add indexes")

        return {
            "success": True,
            "plan": explain.get("plan"),
            "summary": summary,
            "risks": risks,
            "recommendations": recommendations,
        }


# 便捷函数
async def analyze_sql_cost(sql: str, database: str = "default") -> dict:
    """分析 SQL 成本"""
    analyzer = CostAnalyzer(database)
    return await analyzer.analyze(sql)


async def get_explain_plan(sql: str, database: str = "default") -> dict:
    """获取执行计划"""
    analyzer = CostAnalyzer(database)
    return await analyzer.get_explain(sql)


__all__ = [
    "CostAnalyzer",
    "analyze_sql_cost",
    "get_explain_plan",
]
