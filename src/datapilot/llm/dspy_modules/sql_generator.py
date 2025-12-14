# -*- coding: utf-8 -*-
"""
DSPy SQL 生成模块
基于 DSPy 的 SQL 生成器实现
"""

from typing import Optional
import dspy

from .signatures import (
    Text2SQLSignature,
    SQLDecomposeSignature,
    SQLRefineSignature,
    SchemaSelectionSignature,
)


class Text2SQLModule(dspy.Module):
    """
    基础 Text-to-SQL 模块

    使用 Chain-of-Thought 推理生成 SQL
    """

    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(Text2SQLSignature)

    def forward(
        self,
        question: str,
        schema: str,
        dialect: str = "sqlite",
    ) -> dspy.Prediction:
        """
        生成 SQL

        Args:
            question: 用户问题
            schema: 数据库 Schema
            dialect: SQL 方言

        Returns:
            包含 sql 和 explanation 的预测结果
        """
        return self.generate(
            question=question,
            db_schema=schema,
            dialect=dialect,
        )


class DecomposeAndGenerateModule(dspy.Module):
    """
    分解-生成模块 (MAC-SQL 策略)

    1. 将复杂问题分解为子问题
    2. 为每个子问题生成 SQL
    3. 合并结果
    """

    def __init__(self):
        super().__init__()
        self.decompose = dspy.ChainOfThought(SQLDecomposeSignature)
        self.generate = dspy.ChainOfThought(Text2SQLSignature)

    def forward(
        self,
        question: str,
        schema: str,
        dialect: str = "sqlite",
    ) -> dspy.Prediction:
        """
        分解并生成 SQL
        """
        # 1. 分解问题
        decomposition = self.decompose(
            question=question,
            db_schema=schema,
        )

        sub_questions = decomposition.sub_questions

        # 2. 如果只有一个子问题，直接生成
        if len(sub_questions) <= 1:
            return self.generate(
                question=question,
                db_schema=schema,
                dialect=dialect,
            )

        # 3. 为每个子问题生成 SQL
        sub_sqls = []
        for sub_q in sub_questions:
            result = self.generate(
                question=sub_q,
                db_schema=schema,
                dialect=dialect,
            )
            sub_sqls.append(result.sql)

        # 4. 合并 SQL (使用 CTE 或子查询)
        combined_sql = self._combine_sqls(sub_sqls, dialect)

        return dspy.Prediction(
            sql=combined_sql,
            explanation=f"分解为 {len(sub_questions)} 个子问题后合并",
            sub_questions=sub_questions,
            sub_sqls=sub_sqls,
        )

    def _combine_sqls(self, sqls: list[str], dialect: str) -> str:
        """合并多个 SQL"""
        if len(sqls) == 1:
            return sqls[0]

        # 使用 CTE 合并
        cte_parts = []
        for i, sql in enumerate(sqls):
            # 移除末尾分号
            sql_clean = sql.rstrip(';').strip()
            cte_parts.append(f"sub_{i} AS ({sql_clean})")

        # 简单合并 - 实际应用中需要更智能的合并逻辑
        combined = "WITH " + ",\n".join(cte_parts)
        combined += f"\nSELECT * FROM sub_{len(sqls)-1}"

        return combined


class SQLRefineModule(dspy.Module):
    """
    SQL 修正模块

    根据错误反馈修正 SQL
    """

    def __init__(self):
        super().__init__()
        self.refine = dspy.ChainOfThought(SQLRefineSignature)

    def forward(
        self,
        original_sql: str,
        error_message: str,
        schema: str,
    ) -> dspy.Prediction:
        """
        修正 SQL
        """
        return self.refine(
            original_sql=original_sql,
            error_message=error_message,
            db_schema=schema,
        )


class SchemaSelectionModule(dspy.Module):
    """
    Schema 选择模块

    从完整 Schema 中选择相关表
    """

    def __init__(self):
        super().__init__()
        self.select = dspy.ChainOfThought(SchemaSelectionSignature)

    def forward(
        self,
        question: str,
        all_tables: list[str],
        table_descriptions: str,
    ) -> dspy.Prediction:
        """
        选择相关表
        """
        return self.select(
            question=question,
            all_tables=all_tables,
            table_descriptions=table_descriptions,
        )


class MultiPathSQLGenerator(dspy.Module):
    """
    多路 SQL 生成器

    使用多种策略生成 SQL，选择最佳结果
    """

    def __init__(self):
        super().__init__()
        self.direct_gen = Text2SQLModule()
        self.decompose_gen = DecomposeAndGenerateModule()
        self.refine = SQLRefineModule()

    def forward(
        self,
        question: str,
        schema: str,
        dialect: str = "sqlite",
        use_decompose: bool = False,
    ) -> dspy.Prediction:
        """
        多路生成 SQL

        Args:
            question: 用户问题
            schema: 数据库 Schema
            dialect: SQL 方言
            use_decompose: 是否使用分解策略

        Returns:
            最佳 SQL 结果
        """
        candidates = []

        # 1. 直接生成
        direct_result = self.direct_gen(
            question=question,
            db_schema=schema,
            dialect=dialect,
        )
        candidates.append({
            "strategy": "direct",
            "sql": direct_result.sql,
            "explanation": direct_result.explanation,
        })

        # 2. 分解生成 (可选)
        if use_decompose:
            decompose_result = self.decompose_gen(
                question=question,
                db_schema=schema,
                dialect=dialect,
            )
            candidates.append({
                "strategy": "decompose",
                "sql": decompose_result.sql,
                "explanation": decompose_result.explanation,
            })

        # 3. 选择最佳 (简单策略：选择第一个)
        # 实际应用中可以使用 Judge 评估
        best = candidates[0]

        return dspy.Prediction(
            sql=best["sql"],
            explanation=best["explanation"],
            strategy=best["strategy"],
            candidates=candidates,
        )


__all__ = [
    "Text2SQLModule",
    "DecomposeAndGenerateModule",
    "SQLRefineModule",
    "SchemaSelectionModule",
    "MultiPathSQLGenerator",
]
