# -*- coding: utf-8 -*-
"""
Intent Classifier - 查询意图分类器 (LLM 驱动版)

基于 LLM 语义理解进行意图分类，而非硬编码关键词匹配。
符合 README 设计原则：复杂推理统一由 DeepSeek-V3.2 处理。

意图类型:
1. 聚合查询 (aggregation) - SUM, COUNT, AVG 等
2. 简单查找 (lookup) - 单表简单查询
3. 对比分析 (comparison) - 多维度对比
4. 趋势分析 (trend) - 时间序列分析
5. 排名查询 (ranking) - TOP N, 排序
6. 复杂查询 (complex) - 多种意图组合
"""

import json
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from ..llm.deepseek import get_deepseek_client


class QueryIntent(Enum):
    """查询意图类型"""
    AGGREGATION = "aggregation"  # 聚合查询
    LOOKUP = "lookup"            # 简单查找
    COMPARISON = "comparison"    # 对比分析
    TREND = "trend"              # 趋势分析
    RANKING = "ranking"          # 排名查询
    COMPLEX = "complex"          # 复杂查询 (多种意图组合)
    UNKNOWN = "unknown"          # 未知


@dataclass
class IntentResult:
    """意图分类结果"""
    primary_intent: QueryIntent
    confidence: float  # 0-1
    secondary_intents: list[QueryIntent]
    features: dict  # 检测到的特征
    routing_hint: str  # 路由建议
    sql_hints: dict  # SQL 生成提示


# 意图分类的 System Prompt
INTENT_CLASSIFICATION_PROMPT = """你是一个专业的数据库查询意图分类器。

## 任务
分析用户的自然语言查询，识别其查询意图类型。

## 意图类型定义

1. **aggregation** (聚合查询)
   - 需要统计、汇总、计算的查询
   - 特征: 总数、平均值、最大/最小值、求和、计数
   - SQL 特征: COUNT, SUM, AVG, MAX, MIN, GROUP BY

2. **lookup** (简单查找)
   - 直接查询特定数据的简单查询
   - 特征: 查找某条记录、获取某个值、显示某些数据
   - SQL 特征: 简单 SELECT, WHERE 条件

3. **comparison** (对比分析)
   - 比较两个或多个对象/时间段的查询
   - 特征: A和B对比、差异、哪个更好
   - SQL 特征: CASE WHEN, 多表 JOIN, UNION

4. **trend** (趋势分析)
   - 分析数据随时间变化的查询
   - 特征: 趋势、走势、变化、增长、下降、历史
   - SQL 特征: 时间字段, ORDER BY date, 时间分组

5. **ranking** (排名查询)
   - 获取排序后的前N名或后N名
   - 特征: 排名、TOP N、最高、最低、第一
   - SQL 特征: ORDER BY, LIMIT, ROW_NUMBER

6. **complex** (复杂查询)
   - 包含多种意图的复杂查询
   - 特征: 多个子问题、嵌套条件

7. **unknown** (未知)
   - 无法明确分类的查询

## 输出格式
返回 JSON:
```json
{
    "primary_intent": "意图类型",
    "confidence": 0.0-1.0,
    "secondary_intents": ["次要意图1", "次要意图2"],
    "reasoning": "分类理由",
    "features": {
        "detected_entities": ["实体1", "实体2"],
        "time_reference": true/false,
        "comparison_targets": ["对象1", "对象2"],
        "aggregation_type": "count/sum/avg/max/min/none"
    },
    "sql_hints": {
        "suggested_clauses": ["建议的SQL子句"],
        "avoid_patterns": ["应避免的模式"],
        "optimization_tips": ["优化建议"]
    },
    "routing_hint": "路由建议"
}
```

## 注意
- 仔细分析查询的语义，不要仅依赖关键词
- 考虑查询的真实意图，而非表面表达
- 如果有多个意图，选择最主要的作为 primary_intent
- confidence 应反映你对分类的确信程度
"""


class IntentClassifier:
    """
    查询意图分类器 (LLM 驱动)

    使用 LLM 进行语义理解和意图分类，
    而非硬编码的关键词匹配。
    """

    def __init__(self):
        self.llm = get_deepseek_client()

    async def classify_async(self, query: str) -> IntentResult:
        """
        异步分类查询意图 (推荐使用)

        Args:
            query: 用户查询

        Returns:
            意图分类结果
        """
        messages = [
            {"role": "system", "content": INTENT_CLASSIFICATION_PROMPT},
            {"role": "user", "content": f"请分析以下查询的意图:\n\n{query}"},
        ]

        try:
            response = await self.llm.chat(messages)
            return self._parse_response(response, query)
        except Exception as e:
            # LLM 调用失败，返回 UNKNOWN
            return IntentResult(
                primary_intent=QueryIntent.UNKNOWN,
                confidence=0.0,
                secondary_intents=[],
                features={"error": str(e)},
                routing_hint="LLM classification failed, analyze manually",
                sql_hints={},
            )

    def classify(self, query: str) -> IntentResult:
        """
        同步分类查询意图 (兼容旧接口)

        注意: 这是一个同步包装器，内部使用 asyncio.run()
        在已有事件循环的环境中可能会有问题

        Args:
            query: 用户查询

        Returns:
            意图分类结果
        """
        import asyncio

        try:
            # 尝试获取当前事件循环
            loop = asyncio.get_running_loop()
            # 如果在事件循环中，创建一个新任务
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, self.classify_async(query))
                return future.result()
        except RuntimeError:
            # 没有运行中的事件循环，直接运行
            return asyncio.run(self.classify_async(query))

    def _parse_response(self, response: str, original_query: str) -> IntentResult:
        """解析 LLM 响应"""
        import re

        # 尝试提取 JSON
        json_match = re.search(r'\{[\s\S]*\}', response)
        if not json_match:
            return self._fallback_result(original_query)

        try:
            data = json.loads(json_match.group())

            # 解析主要意图
            primary_str = data.get("primary_intent", "unknown").lower()
            try:
                primary_intent = QueryIntent(primary_str)
            except ValueError:
                primary_intent = QueryIntent.UNKNOWN

            # 解析次要意图
            secondary_strs = data.get("secondary_intents", [])
            secondary_intents = []
            for s in secondary_strs:
                try:
                    secondary_intents.append(QueryIntent(s.lower()))
                except ValueError:
                    pass

            # 解析置信度
            confidence = float(data.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))

            # 解析特征
            features = data.get("features", {})
            if "reasoning" in data:
                features["reasoning"] = data["reasoning"]

            # 解析 SQL 提示
            sql_hints = data.get("sql_hints", {})

            # 解析路由提示
            routing_hint = data.get("routing_hint", self._get_default_routing_hint(primary_intent))

            return IntentResult(
                primary_intent=primary_intent,
                confidence=confidence,
                secondary_intents=secondary_intents,
                features=features,
                routing_hint=routing_hint,
                sql_hints=sql_hints,
            )

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            return self._fallback_result(original_query, error=str(e))

    def _fallback_result(self, query: str, error: str = None) -> IntentResult:
        """解析失败时的回退结果"""
        features = {}
        if error:
            features["parse_error"] = error

        return IntentResult(
            primary_intent=QueryIntent.UNKNOWN,
            confidence=0.3,
            secondary_intents=[],
            features=features,
            routing_hint="Unable to parse LLM response, proceed with caution",
            sql_hints={},
        )

    def _get_default_routing_hint(self, intent: QueryIntent) -> str:
        """获取默认路由提示"""
        hints = {
            QueryIntent.AGGREGATION: "Use aggregation functions, consider GROUP BY",
            QueryIntent.LOOKUP: "Simple SELECT query, minimal joins",
            QueryIntent.COMPARISON: "May need multiple queries or CASE statements",
            QueryIntent.TREND: "Include date/time columns, consider ORDER BY date",
            QueryIntent.RANKING: "Use ORDER BY with LIMIT, consider window functions",
            QueryIntent.COMPLEX: "Complex query, use decomposition strategy",
            QueryIntent.UNKNOWN: "Analyze query structure carefully",
        }
        return hints.get(intent, "")

    def get_sql_hints(self, intent_result: IntentResult) -> dict:
        """
        获取 SQL 生成提示

        Args:
            intent_result: 意图分类结果

        Returns:
            SQL 生成提示
        """
        # 如果 LLM 已经返回了 sql_hints，直接使用
        if intent_result.sql_hints:
            return intent_result.sql_hints

        # 否则返回基于意图的默认提示
        intent = intent_result.primary_intent
        default_hints = {
            QueryIntent.AGGREGATION: {
                "suggested_clauses": ["GROUP BY", "HAVING", "COUNT/SUM/AVG"],
                "avoid_patterns": ["SELECT * without aggregation"],
                "optimization_tips": ["Consider adding indexes on GROUP BY columns"],
            },
            QueryIntent.LOOKUP: {
                "suggested_clauses": ["WHERE", "LIMIT"],
                "avoid_patterns": ["Unnecessary JOINs", "SELECT *"],
                "optimization_tips": ["Use indexed columns in WHERE"],
            },
            QueryIntent.COMPARISON: {
                "suggested_clauses": ["CASE WHEN", "UNION", "JOIN"],
                "avoid_patterns": ["Multiple separate queries when one suffices"],
                "optimization_tips": ["Consider using CTEs for clarity"],
            },
            QueryIntent.TREND: {
                "suggested_clauses": ["ORDER BY date", "DATE functions", "GROUP BY time_period"],
                "avoid_patterns": ["Missing time column in SELECT"],
                "optimization_tips": ["Ensure date column is indexed"],
            },
            QueryIntent.RANKING: {
                "suggested_clauses": ["ORDER BY", "LIMIT", "ROW_NUMBER()"],
                "avoid_patterns": ["Fetching all rows then filtering"],
                "optimization_tips": ["Use covering index for ORDER BY columns"],
            },
            QueryIntent.COMPLEX: {
                "suggested_clauses": ["CTEs (WITH clause)", "Subqueries"],
                "avoid_patterns": ["Overly nested subqueries"],
                "optimization_tips": ["Break down into smaller queries if possible"],
            },
            QueryIntent.UNKNOWN: {
                "suggested_clauses": [],
                "avoid_patterns": [],
                "optimization_tips": ["Analyze query structure carefully"],
            },
        }
        return default_hints.get(intent, {})


# 便捷函数
async def classify_intent_async(query: str) -> IntentResult:
    """异步分类查询意图"""
    classifier = IntentClassifier()
    return await classifier.classify_async(query)


def classify_intent(query: str) -> IntentResult:
    """同步分类查询意图 (兼容旧接口)"""
    classifier = IntentClassifier()
    return classifier.classify(query)


__all__ = [
    "IntentClassifier",
    "IntentResult",
    "QueryIntent",
    "classify_intent",
    "classify_intent_async",
]
