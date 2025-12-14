# -*- coding: utf-8 -*-
"""
Ambi-Resolver Agent (LLM 驱动版)
歧义消解 Agent - 使用 LLM 检测并解决用户查询中的歧义

符合 README 设计原则:
- 准确性优先: 通过 LLM 语义理解检测歧义
- Human-in-the-loop: 生成澄清选项供用户选择
- 复杂推理统一由 DeepSeek 处理
"""

import json
import re
from dataclasses import dataclass
from typing import Any, Optional

from ..core.state import DataPilotState, ClarifyOption
from ..llm.deepseek import get_deepseek_client
from ..db.connector import get_db_manager
from ..vector.schema_index import get_schema_index
from ..vector.value_index import get_value_index


@dataclass
class AmbiguityResult:
    """歧义检测结果"""
    has_ambiguity: bool
    ambiguity_type: Optional[str] = None
    question: Optional[str] = None
    options: list[str] = None
    confidence: float = 0.0  # 置信度 0-1
    reasoning: str = ""  # LLM 的推理过程

    def __post_init__(self):
        if self.options is None:
            self.options = []


# 歧义检测的 System Prompt
AMBIGUITY_DETECTION_PROMPT = """你是一个专业的数据库查询歧义检测器。

## 任务
分析用户的自然语言查询，判断是否存在需要澄清的歧义。

## 歧义类型

1. **time** (时间歧义)
   - 时间范围不明确
   - 例如: "最近"、"近期"、"之前" 没有具体时间
   - 需要澄清: 具体的时间范围

2. **metric** (指标歧义)
   - 排序/比较的指标不明确
   - 例如: "最高"、"最好" 没有说明按什么指标
   - 需要澄清: 具体的排序指标

3. **scope** (范围歧义)
   - 数据范围不明确
   - 例如: "所有产品" 是否包含下架产品
   - 需要澄清: 数据的筛选范围

4. **granularity** (粒度歧义)
   - 统计粒度不明确
   - 例如: "平均销量" 是按天/周/月平均
   - 需要澄清: 统计的时间粒度

5. **entity** (实体歧义)
   - 指代不明确
   - 例如: "它"、"这个" 指代什么
   - 需要澄清: 具体指代的实体

6. **none** (无歧义)
   - 查询意图明确，无需澄清

## 输出格式
返回 JSON:
```json
{
    "has_ambiguity": true/false,
    "ambiguity_type": "time/metric/scope/granularity/entity/none",
    "confidence": 0.0-1.0,
    "reasoning": "判断理由",
    "question": "澄清问题 (如果有歧义)",
    "options": ["选项1", "选项2", "选项3"] (如果有歧义，2-5个选项)
}
```

## 注意
- 仔细分析查询的语义，不要仅依赖关键词
- 只检测真正影响查询结果的歧义
- 如果查询已经足够明确，不要强行找歧义
- 选项应该是用户可能的真实意图，具体且可操作
- confidence 应反映你对歧义判断的确信程度
"""


class AmbiResolver:
    """
    Ambi-Resolver Agent - 歧义消解者 (LLM 驱动)

    使用 LLM 进行语义理解来检测歧义，
    而非硬编码的关键词匹配。

    职责:
    1. 使用 LLM 检测用户查询中的歧义
    2. 生成澄清选项供用户选择
    3. 根据用户选择更新查询上下文
    """

    def __init__(self, confidence_threshold: Optional[float] = None):
        """
        初始化 AmbiResolver

        Args:
            confidence_threshold: 置信度阈值，高于此值才需要澄清。
                                  如果为 None，则从 settings 读取默认值。
        """
        # 从 settings 读取默认配置
        from ..config.settings import get_settings
        settings = get_settings()

        self.llm = get_deepseek_client()
        self.db_manager = get_db_manager()
        self.confidence_threshold = confidence_threshold if confidence_threshold is not None else settings.ambi_resolver_confidence_threshold

    async def detect_ambiguity(
        self,
        query: str,
        database: str = 'default',
        schema_context: str = '',
    ) -> tuple[Optional[ClarifyOption], AmbiguityResult]:
        """
        使用 LLM 检测查询中的歧义

        Args:
            query: 用户查询
            database: 数据库名称
            schema_context: Schema 上下文 (可选)

        Returns:
            (澄清选项, 歧义检测结果) - 如果没有歧义则澄清选项为 None
        """
        # 获取 Schema 上下文 (帮助 LLM 理解业务背景)
        if not schema_context:
            try:
                schema_index = get_schema_index(database)
                tables = await schema_index.search_tables(query, top_k=3)
                schema_context = ', '.join([t['name'] for t in tables]) if tables else ''
            except Exception:
                schema_context = ''

        # 构建 LLM 请求
        messages = [
            {'role': 'system', 'content': AMBIGUITY_DETECTION_PROMPT},
            {'role': 'user', 'content': self._build_detection_prompt(query, schema_context)},
        ]

        try:
            response = await self.llm.chat(messages)
            result = self._parse_detection_response(response)

            # 根据置信度决定是否需要澄清
            if result.has_ambiguity and result.confidence >= self.confidence_threshold:
                clarify_option = ClarifyOption(
                    question=result.question or '请选择',
                    options=result.options or []
                )
                return clarify_option, result

            return None, result

        except Exception as e:
            # LLM 调用失败，返回无歧义
            return None, AmbiguityResult(
                has_ambiguity=False,
                confidence=0.0,
                reasoning=f"LLM detection failed: {str(e)}"
            )

    def _build_detection_prompt(self, query: str, schema_context: str) -> str:
        """构建检测提示"""
        prompt = f"用户查询: {query}"
        if schema_context:
            prompt += f"\n\n相关数据表: {schema_context}"
        prompt += "\n\n请分析是否存在歧义。"
        return prompt

    def _parse_detection_response(self, response: str) -> AmbiguityResult:
        """解析 LLM 响应"""
        # 尝试提取 JSON
        json_match = re.search(r'\{[\s\S]*\}', response)
        if not json_match:
            return AmbiguityResult(
                has_ambiguity=False,
                confidence=0.0,
                reasoning="Failed to parse LLM response"
            )

        try:
            data = json.loads(json_match.group())

            has_ambiguity = data.get('has_ambiguity', False)
            ambiguity_type = data.get('ambiguity_type', 'none')
            confidence = float(data.get('confidence', 0.0))
            confidence = max(0.0, min(1.0, confidence))

            # 如果类型是 none，则没有歧义
            if ambiguity_type == 'none':
                has_ambiguity = False

            return AmbiguityResult(
                has_ambiguity=has_ambiguity,
                ambiguity_type=ambiguity_type if has_ambiguity else None,
                question=data.get('question'),
                options=data.get('options', []),
                confidence=confidence,
                reasoning=data.get('reasoning', ''),
            )

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            return AmbiguityResult(
                has_ambiguity=False,
                confidence=0.0,
                reasoning=f"JSON parse error: {str(e)}"
            )

    async def resolve_with_selection(
        self,
        query: str,
        selected_option: str,
    ) -> str:
        """
        根据用户选择解析歧义，生成明确的查询

        Args:
            query: 原始查询
            selected_option: 用户选择的选项

        Returns:
            消歧后的查询
        """
        messages = [
            {
                'role': 'system',
                'content': '''你是一个查询改写专家。根据用户的选择，将模糊的查询改写为明确的查询。

## 要求
1. 保持原始查询的意图
2. 将用户选择的条件融入查询
3. 输出改写后的自然语言查询

## 输出格式
直接输出改写后的查询，不要有其他内容。'''
            },
            {
                'role': 'user',
                'content': f'''原始查询: {query}
用户选择: {selected_option}

请改写查询。'''
            }
        ]

        try:
            response = await self.llm.chat(messages)
            return response.strip()
        except Exception:
            # 简单拼接
            return f'{query} ({selected_option})'

    async def search_entity_candidates(
        self,
        entity: str,
        database: str = 'default',
        top_k: int = 5,
    ) -> list[dict]:
        """
        搜索实体候选值

        当用户输入的实体名称模糊时，搜索可能的匹配

        Args:
            entity: 用户输入的实体
            database: 数据库名称
            top_k: 返回数量

        Returns:
            候选值列表
        """
        try:
            value_index = get_value_index(database)
            results = await value_index.search_values(
                query=entity,
                top_k=top_k,
                score_threshold=0.5,
            )
            return results
        except Exception as e:
            print(f'Entity search error: {e}')
            return []

    async def run(self, state: DataPilotState) -> dict[str, Any]:
        """
        执行 Ambi-Resolver Agent

        Args:
            state: 当前工作流状态

        Returns:
            状态更新 (包含歧义分析结果)
        """
        query = state['query']
        database = state.get('database', 'default')
        schema_context = state.get('schema_context', '')

        # 使用 LLM 检测歧义
        clarify_option, ambiguity_result = await self.detect_ambiguity(
            query, database, schema_context
        )

        # 构建歧义分析结果
        ambiguity_analysis = {
            'has_ambiguity': ambiguity_result.has_ambiguity,
            'confidence': ambiguity_result.confidence,
            'ambiguity_type': ambiguity_result.ambiguity_type,
            'reasoning': ambiguity_result.reasoning,
        }

        if clarify_option and clarify_option["options"]:
            # 需要澄清
            return {
                'current_agent': 'ambi_resolver',
                'next_agent': None,  # 等待用户输入
                'clarify_needed': True,
                'clarify_options': clarify_option,
                'ambiguity_analysis': ambiguity_analysis,
            }
        else:
            # 无歧义或置信度过低，继续流程
            return {
                'current_agent': 'ambi_resolver',
                'next_agent': 'data_sniper',
                'clarify_needed': False,
                'clarify_options': None,
                'ambiguity_analysis': ambiguity_analysis,
            }

    async def handle_clarification(
        self,
        state: DataPilotState,
        selected_option: str,
    ) -> dict[str, Any]:
        """
        处理用户的澄清选择

        Args:
            state: 当前状态
            selected_option: 用户选择

        Returns:
            状态更新
        """
        original_query = state['query']

        # 改写查询
        resolved_query = await self.resolve_with_selection(
            original_query,
            selected_option,
        )

        return {
            'query': resolved_query,
            'current_agent': 'ambi_resolver',
            'next_agent': 'data_sniper',
            'clarify_needed': False,
            'clarify_options': None,
        }


# LangGraph 节点函数
async def ambi_resolver_node(state: DataPilotState) -> dict[str, Any]:
    """LangGraph 节点：Ambi-Resolver"""
    resolver = AmbiResolver()
    return await resolver.run(state)


async def clarification_handler_node(
    state: DataPilotState,
    selected_option: str,
) -> dict[str, Any]:
    """LangGraph 节点：处理澄清"""
    resolver = AmbiResolver()
    return await resolver.handle_clarification(state, selected_option)


__all__ = [
    'AmbiResolver',
    'AmbiguityResult',
    'ambi_resolver_node',
    'clarification_handler_node',
]
