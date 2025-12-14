"""
DeepSeek LLM 客户端
使用 LangChain 最新版本集成 DeepSeek API

增强功能:
- 成本熔断机制
- 租户级别配额管理
"""

from typing import Any, Optional
from time import perf_counter
from prometheus_client import Counter, Histogram

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.callbacks import BaseCallbackHandler
from langchain_openai import ChatOpenAI

from ..config.settings import get_settings
from .cost_control import (
    get_cost_controller,
    estimate_cost,
    CostLimitExceeded,
)

# Prometheus metrics
LLM_REQUEST_COUNT = Counter(
    "datapilot_llm_request_total",
    "LLM request count",
    labelnames=["model", "status"],
)
LLM_REQUEST_LATENCY = Histogram(
    "datapilot_llm_request_latency_seconds",
    "LLM request latency",
    labelnames=["model", "status"],
    buckets=(0.1, 0.5, 1, 2, 5, 10, 20),
)


class DeepSeekClient:
    """
    DeepSeek LLM 客户端封装

    增强功能:
    - 成本熔断机制
    - 租户级别配额管理
    - Token 使用量追踪
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tenant_id: str = "default",
        enable_cost_control: bool = True,
    ):
        settings = get_settings()

        self.api_key = api_key or settings.deepseek_api_key
        self.base_url = base_url or settings.deepseek_base_url
        self.model = model or settings.deepseek_model
        self.temperature = temperature if temperature is not None else settings.deepseek_temperature
        self.max_tokens = max_tokens or settings.deepseek_max_tokens
        self.tenant_id = tenant_id
        self.enable_cost_control = enable_cost_control

        # 成本控制器
        self.cost_controller = get_cost_controller() if enable_cost_control else None

        # 创建 LangChain ChatOpenAI 实例（DeepSeek 兼容 OpenAI API）
        self._llm = ChatOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

    @property
    def llm(self) -> BaseChatModel:
        """获取 LangChain LLM 实例"""
        return self._llm

    async def chat(
        self,
        messages: list[dict[str, str]],
        tenant_id: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """
        异步聊天接口 (带成本控制)

        Args:
            messages: 消息列表，格式为 [{"role": "user", "content": "..."}]
            tenant_id: 租户 ID (覆盖默认值)
            **kwargs: 其他参数

        Returns:
            AI 回复内容

        Raises:
            CostLimitExceeded: 成本限制超出
        """
        tenant = tenant_id or self.tenant_id

        # 成本控制检查
        if self.cost_controller:
            # 估算输入 token (粗略估计: 1 token ≈ 4 字符)
            input_text = " ".join(m.get("content", "") for m in messages)
            estimated_input_tokens = len(input_text) // 4

            allowed, reason = self.cost_controller.check_allowed(
                tenant_id=tenant,
                estimated_tokens=estimated_input_tokens,
            )
            if not allowed:
                raise CostLimitExceeded(reason)

        # 转换消息格式
        langchain_messages = self._convert_messages(messages)

        status = "success"
        start = perf_counter()
        input_tokens = 0
        output_tokens = 0

        try:
            response = await self._llm.ainvoke(langchain_messages, **kwargs)

            # 尝试获取 token 使用量
            if hasattr(response, 'response_metadata'):
                usage = response.response_metadata.get('token_usage', {})
                input_tokens = usage.get('prompt_tokens', 0)
                output_tokens = usage.get('completion_tokens', 0)

            # 如果没有获取到，使用估算值
            if input_tokens == 0:
                input_text = " ".join(m.get("content", "") for m in messages)
                input_tokens = len(input_text) // 4
            if output_tokens == 0:
                output_tokens = len(response.content) // 4

            # 记录成本
            if self.cost_controller:
                self.cost_controller.record_usage(
                    model=self.model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    tenant_id=tenant,
                )

            return response.content
        except CostLimitExceeded:
            status = "cost_limited"
            raise
        except Exception:
            status = "error"
            raise
        finally:
            duration = perf_counter() - start
            LLM_REQUEST_COUNT.labels(model=self.model, status=status).inc()
            LLM_REQUEST_LATENCY.labels(model=self.model, status=status).observe(duration)

    def chat_sync(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> str:
        """
        同步聊天接口

        Args:
            messages: 消息列表
            **kwargs: 其他参数

        Returns:
            AI 回复内容
        """
        langchain_messages = self._convert_messages(messages)
        status = "success"
        start = perf_counter()
        try:
            response = self._llm.invoke(langchain_messages, **kwargs)
            return response.content
        except Exception:
            status = "error"
            raise
        finally:
            duration = perf_counter() - start
            LLM_REQUEST_COUNT.labels(model=self.model, status=status).inc()
            LLM_REQUEST_LATENCY.labels(model=self.model, status=status).observe(duration)

    async def generate_sql(
        self,
        query: str,
        schema: str,
        dialect: str = "mysql",
        examples: Optional[list[dict]] = None,
    ) -> dict[str, Any]:
        """
        生成 SQL 查询

        Args:
            query: 用户自然语言问题
            schema: 数据库 Schema (DDL)
            dialect: SQL 方言 (mysql/postgresql/sqlite)
            examples: Few-shot 示例

        Returns:
            包含 SQL 和解释的字典
        """
        # 构建系统提示
        system_prompt = f"""你是一个专业的 SQL 专家。根据用户的自然语言问题，生成正确的 {dialect.upper()} SQL 查询。

## 数据库 Schema
{schema}

## 要求
1. 只生成 SELECT 查询，不要生成任何修改数据的语句
2. SQL 必须语法正确，符合 {dialect.upper()} 方言
3. 使用清晰的别名和格式化
4. 如果问题模糊，做出合理假设并在解释中说明

## 输出格式
请按以下 JSON 格式输出：
```json
{{
    "sql": "你生成的 SQL 查询",
    "explanation": "SQL 逻辑的简要解释",
    "assumptions": ["如果有假设，列在这里"]
}}
```
"""

        # 构建用户消息
        user_message = f"问题：{query}"

        # 添加 few-shot 示例
        messages = [{"role": "system", "content": system_prompt}]

        if examples:
            for ex in examples[:3]:  # 最多 3 个示例
                messages.append({"role": "user", "content": f"问题：{ex['question']}"})
                messages.append({"role": "assistant", "content": f"```json\n{ex['answer']}\n```"})

        messages.append({"role": "user", "content": user_message})

        # 调用 LLM
        response = await self.chat(messages)

        # 解析响应
        return self._parse_sql_response(response)

    def _convert_messages(self, messages: list[dict[str, str]]) -> list[BaseMessage]:
        """转换消息格式为 LangChain 格式"""
        langchain_messages = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                langchain_messages.append(SystemMessage(content=content))
            elif role == "assistant":
                langchain_messages.append(AIMessage(content=content))
            else:
                langchain_messages.append(HumanMessage(content=content))

        return langchain_messages

    def _parse_sql_response(self, response: str) -> dict[str, Any]:
        """解析 SQL 生成响应"""
        import json
        import re

        # 尝试提取 JSON
        json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)

        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # 尝试直接解析
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # 回退：尝试提取 SQL
        sql_match = re.search(r"```sql\s*(.*?)\s*```", response, re.DOTALL)
        if sql_match:
            return {
                "sql": sql_match.group(1).strip(),
                "explanation": "从响应中提取的 SQL",
                "assumptions": [],
            }

        # 最后回退
        return {
            "sql": response.strip(),
            "explanation": "原始响应",
            "assumptions": [],
            "raw_response": response,
        }


# 全局客户端实例
_client: Optional[DeepSeekClient] = None


def get_deepseek_client() -> DeepSeekClient:
    """获取 DeepSeek 客户端单例"""
    global _client
    if _client is None:
        _client = DeepSeekClient()
    return _client


# 导出
__all__ = ["DeepSeekClient", "get_deepseek_client"]
