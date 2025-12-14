# -*- coding: utf-8 -*-
"""
LLM-Native SQL Generator - 纯 LLM 驱动的 SQL 生成器

核心理念:
1. 不做硬编码规则，一切交给 LLM 处理
2. 提供丰富的数据库上下文（元信息、样本值）
3. LLM 根据真实数据生成 SQL，而非依赖预设规则
4. 支持多轮对话，自我修正

与传统方法的区别:
- 传统: 硬编码时间解析 → 生成 SQL
- 本方案: 告诉 LLM 有哪些时间字段 → LLM 自己决定如何处理

优势:
- 适应任何数据库结构
- 无需维护复杂的规则引擎
- 利用 LLM 的推理能力处理模糊查询
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional
from datetime import datetime

from ..db.schema_introspector import SchemaIntrospector, SchemaMetadata
from ..llm.deepseek import get_deepseek_client


@dataclass
class GenerationContext:
    """SQL 生成上下文"""
    query: str                          # 用户查询
    schema_metadata: SchemaMetadata     # 数据库元信息
    schema_context: str                 # 格式化的 Schema 上下文
    time_context: str                   # 时间字段上下文
    enum_context: str                   # 枚举字段上下文
    conversation_history: list = field(default_factory=list)  # 对话历史
    current_date: str = ""              # 当前日期

    def __post_init__(self):
        if not self.current_date:
            self.current_date = datetime.now().strftime("%Y-%m-%d")


@dataclass
class GenerationResult:
    """生成结果"""
    success: bool
    sql: Optional[str] = None
    explanation: str = ""
    confidence: float = 0.0
    warnings: list[str] = field(default_factory=list)
    needs_clarification: bool = False
    clarification_question: Optional[str] = None
    clarification_options: list[str] = field(default_factory=list)


# 核心 System Prompt - 让 LLM 完全理解它需要做什么
NATIVE_SQL_SYSTEM_PROMPT = """你是一个专业的 SQL 生成专家。你的任务是根据用户的自然语言查询和提供的数据库信息，生成准确的 SQL 语句。

## 关键原则

### 1. 基于真实数据库结构
- 只使用提供的表和列，不要假设不存在的结构
- 注意列的数据类型，特别是时间字段
- 利用样本值来理解数据的实际格式

### 2. 时间处理
- 查看提供的时间字段列表，选择合适的字段
- 根据今天的日期计算相对时间（如"最近7天"、"上个月"）
- 注意时间字段的实际格式（DATE、DATETIME、TIMESTAMP）

### 3. 值匹配
- 使用提供的枚举值/样本值来匹配用户的表述
- 如果用户说"苹果手机"而样本值中有"iPhone"，使用"iPhone"
- 对于模糊匹配，使用 LIKE 并说明原因

### 4. 不确定时请求澄清
- 如果查询有歧义，列出可能的解释
- 如果找不到匹配的值，说明并建议可能的替代
- 不要猜测，宁可请求澄清

### 5. 安全性
- 不生成 DROP、DELETE、TRUNCATE 等危险语句
- 始终使用参数化的方式思考（虽然输出是纯 SQL）
- 大表查询要考虑添加 LIMIT

## 输出格式

返回 JSON:
```json
{
    "success": true/false,
    "sql": "生成的 SQL 语句",
    "explanation": "SQL 解释和生成理由",
    "confidence": 0.0-1.0,
    "warnings": ["警告信息"],
    "needs_clarification": true/false,
    "clarification_question": "澄清问题（如果需要）",
    "clarification_options": ["选项1", "选项2"]
}
```

## 示例

用户: "最近一周的订单"
时间字段: orders.created_at (DATETIME)
今天日期: 2025-01-15

思考过程:
1. 需要查询 orders 表
2. 时间字段是 created_at
3. "最近一周" = 2025-01-08 到 2025-01-15
4. 生成 SQL

输出:
```json
{
    "success": true,
    "sql": "SELECT * FROM orders WHERE created_at >= '2025-01-08' AND created_at < '2025-01-16' ORDER BY created_at DESC",
    "explanation": "查询最近7天的订单，使用 created_at 字段过滤",
    "confidence": 0.95,
    "warnings": ["建议添加 LIMIT 限制结果数量"],
    "needs_clarification": false
}
```
"""


class LLMNativeSQLGenerator:
    """
    纯 LLM 驱动的 SQL 生成器

    不依赖硬编码规则，完全依靠 LLM 的理解和推理能力
    """

    def __init__(self, connector):
        """
        初始化

        Args:
            connector: 数据库连接器
        """
        self.connector = connector
        self.introspector = SchemaIntrospector(connector)
        self.llm = get_deepseek_client()
        self._metadata_cache: Optional[SchemaMetadata] = None

    async def get_metadata(self, force_refresh: bool = False) -> SchemaMetadata:
        """获取数据库元信息（带缓存）"""
        if self._metadata_cache is None or force_refresh:
            self._metadata_cache = await self.introspector.introspect(
                include_samples=True,
                sample_limit=20,
                include_row_counts=True,
            )
        return self._metadata_cache

    async def generate(
        self,
        query: str,
        conversation_history: Optional[list] = None,
    ) -> GenerationResult:
        """
        生成 SQL

        Args:
            query: 用户查询
            conversation_history: 对话历史（用于多轮对话）

        Returns:
            生成结果
        """
        # 获取元信息
        metadata = await self.get_metadata()

        # 构建上下文
        context = GenerationContext(
            query=query,
            schema_metadata=metadata,
            schema_context=self.introspector.generate_llm_context(metadata),
            time_context=self.introspector.generate_time_fields_context(metadata),
            enum_context=self.introspector.generate_enum_fields_context(metadata),
            conversation_history=conversation_history or [],
        )

        # 调用 LLM
        return await self._call_llm(context)

    async def _call_llm(self, context: GenerationContext) -> GenerationResult:
        """调用 LLM 生成 SQL"""
        # 构建用户消息
        user_message = self._build_user_message(context)

        messages = [
            {"role": "system", "content": NATIVE_SQL_SYSTEM_PROMPT},
        ]

        # 添加对话历史
        for hist in context.conversation_history:
            messages.append(hist)

        messages.append({"role": "user", "content": user_message})

        try:
            response = await self.llm.chat(messages)
            return self._parse_response(response)
        except Exception as e:
            return GenerationResult(
                success=False,
                explanation=f"LLM 调用失败: {str(e)}",
            )

    def _build_user_message(self, context: GenerationContext) -> str:
        """构建发送给 LLM 的消息"""
        return f"""## 当前日期
{context.current_date}

## 数据库结构
{context.schema_context}

## 时间字段
{context.time_context}

## 枚举/状态字段
{context.enum_context}

## 用户查询
{context.query}

请根据以上信息生成 SQL。"""

    def _parse_response(self, response: str) -> GenerationResult:
        """解析 LLM 响应"""
        # 尝试提取 JSON
        json_match = re.search(r'\{[\s\S]*\}', response)
        if not json_match:
            # 尝试直接提取 SQL
            sql_match = re.search(r'(?:```sql\s*)?(SELECT[\s\S]*?)(?:```|$)', response, re.IGNORECASE)
            if sql_match:
                return GenerationResult(
                    success=True,
                    sql=sql_match.group(1).strip(),
                    explanation="从响应中直接提取 SQL",
                    confidence=0.7,
                )
            return GenerationResult(
                success=False,
                explanation="无法解析 LLM 响应",
            )

        try:
            data = json.loads(json_match.group())
            return GenerationResult(
                success=data.get("success", False),
                sql=data.get("sql"),
                explanation=data.get("explanation", ""),
                confidence=data.get("confidence", 0.0),
                warnings=data.get("warnings", []),
                needs_clarification=data.get("needs_clarification", False),
                clarification_question=data.get("clarification_question"),
                clarification_options=data.get("clarification_options", []),
            )
        except (json.JSONDecodeError, KeyError) as e:
            return GenerationResult(
                success=False,
                explanation=f"JSON 解析失败: {str(e)}",
            )

    async def generate_with_clarification(
        self,
        query: str,
        clarification_answer: str,
        original_result: GenerationResult,
    ) -> GenerationResult:
        """
        处理用户澄清后重新生成

        Args:
            query: 原始查询
            clarification_answer: 用户的澄清回答
            original_result: 原始生成结果

        Returns:
            新的生成结果
        """
        # 构建对话历史
        history = [
            {"role": "user", "content": query},
            {"role": "assistant", "content": json.dumps({
                "needs_clarification": True,
                "clarification_question": original_result.clarification_question,
                "clarification_options": original_result.clarification_options,
            }, ensure_ascii=False)},
            {"role": "user", "content": f"我选择: {clarification_answer}"},
        ]

        return await self.generate(
            query=f"{query} (用户澄清: {clarification_answer})",
            conversation_history=history,
        )

    async def refine_sql(
        self,
        original_sql: str,
        error_message: str,
        query: str,
    ) -> GenerationResult:
        """
        修正 SQL 错误

        Args:
            original_sql: 原始 SQL
            error_message: 错误信息
            query: 原始查询

        Returns:
            修正后的结果
        """
        metadata = await self.get_metadata()

        refine_prompt = f"""## 原始查询
{query}

## 生成的 SQL
{original_sql}

## 执行错误
{error_message}

## 数据库结构
{self.introspector.generate_llm_context(metadata)}

请分析错误原因并修正 SQL。"""

        messages = [
            {"role": "system", "content": NATIVE_SQL_SYSTEM_PROMPT},
            {"role": "user", "content": refine_prompt},
        ]

        try:
            response = await self.llm.chat(messages)
            return self._parse_response(response)
        except Exception as e:
            return GenerationResult(
                success=False,
                explanation=f"SQL 修正失败: {str(e)}",
            )


async def create_native_generator(database: str = "default") -> LLMNativeSQLGenerator:
    """
    创建 LLM 原生 SQL 生成器的便捷函数

    Args:
        database: 数据库名称

    Returns:
        生成器实例
    """
    from ..db.connector import get_db_manager
    db_manager = get_db_manager()
    connector = db_manager.get(database)
    return LLMNativeSQLGenerator(connector)


__all__ = [
    "LLMNativeSQLGenerator",
    "GenerationContext",
    "GenerationResult",
    "create_native_generator",
]
