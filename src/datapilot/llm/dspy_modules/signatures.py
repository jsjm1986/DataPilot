# -*- coding: utf-8 -*-
"""
DSPy Signatures 定义
定义 SQL 生成任务的输入输出签名
"""

import dspy


class Text2SQLSignature(dspy.Signature):
    """
    自然语言到 SQL 的转换签名

    输入: 用户问题 + 数据库 Schema
    输出: SQL 查询 + 解释
    """
    question: str = dspy.InputField(desc="用户的自然语言问题")
    db_schema: str = dspy.InputField(desc="数据库 Schema (DDL)")
    dialect: str = dspy.InputField(desc="SQL 方言 (mysql/postgresql/sqlite)", default="sqlite")

    sql: str = dspy.OutputField(desc="生成的 SQL 查询语句")
    explanation: str = dspy.OutputField(desc="SQL 查询的解释说明")


class SQLDecomposeSignature(dspy.Signature):
    """
    复杂问题分解签名 (MAC-SQL 策略)

    将复杂问题分解为多个子问题
    """
    question: str = dspy.InputField(desc="用户的复杂问题")
    db_schema: str = dspy.InputField(desc="数据库 Schema")

    sub_questions: list[str] = dspy.OutputField(desc="分解后的子问题列表")
    reasoning: str = dspy.OutputField(desc="分解的推理过程")


class SQLRefineSignature(dspy.Signature):
    """
    SQL 修正签名

    根据错误信息修正 SQL
    """
    original_sql: str = dspy.InputField(desc="原始 SQL 语句")
    error_message: str = dspy.InputField(desc="错误信息")
    db_schema: str = dspy.InputField(desc="数据库 Schema")

    refined_sql: str = dspy.OutputField(desc="修正后的 SQL")
    fix_explanation: str = dspy.OutputField(desc="修正说明")


class SchemaSelectionSignature(dspy.Signature):
    """
    Schema 选择签名

    从完整 Schema 中选择相关表
    """
    question: str = dspy.InputField(desc="用户问题")
    all_tables: list[str] = dspy.InputField(desc="所有表名列表")
    table_descriptions: str = dspy.InputField(desc="表的描述信息")

    relevant_tables: list[str] = dspy.OutputField(desc="相关表名列表")
    reasoning: str = dspy.OutputField(desc="选择理由")


class ValueMappingSignature(dspy.Signature):
    """
    值映射签名

    将用户输入的实体映射到数据库值
    """
    user_term: str = dspy.InputField(desc="用户输入的术语")
    candidates: list[str] = dspy.InputField(desc="候选数据库值列表")
    context: str = dspy.InputField(desc="上下文信息")

    best_match: str = dspy.OutputField(desc="最佳匹配值")
    confidence: float = dspy.OutputField(desc="匹配置信度 0-1")


class AmbiguityDetectionSignature(dspy.Signature):
    """
    歧义检测签名
    """
    question: str = dspy.InputField(desc="用户问题")
    schema_context: str = dspy.InputField(desc="Schema 上下文")

    has_ambiguity: bool = dspy.OutputField(desc="是否存在歧义")
    ambiguity_type: str = dspy.OutputField(desc="歧义类型")
    clarify_question: str = dspy.OutputField(desc="澄清问题")
    options: list[str] = dspy.OutputField(desc="选项列表")


class ChartRecommendSignature(dspy.Signature):
    """
    图表推荐签名
    """
    question: str = dspy.InputField(desc="用户问题")
    sql: str = dspy.InputField(desc="SQL 查询")
    columns: list[str] = dspy.InputField(desc="结果列名")
    sample_data: str = dspy.InputField(desc="示例数据")

    chart_type: str = dspy.OutputField(desc="推荐图表类型 (bar/line/pie/table)")
    x_axis: str = dspy.OutputField(desc="X 轴字段")
    y_axis: str = dspy.OutputField(desc="Y 轴字段")
    title: str = dspy.OutputField(desc="图表标题")


__all__ = [
    "Text2SQLSignature",
    "SQLDecomposeSignature",
    "SQLRefineSignature",
    "SchemaSelectionSignature",
    "ValueMappingSignature",
    "AmbiguityDetectionSignature",
    "ChartRecommendSignature",
]
