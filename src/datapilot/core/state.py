"""
DataPilot LangGraph 全局状态定义
定义工作流中各 Agent 共享的状态结构
"""

from datetime import datetime
from typing import Annotated, Any, Literal, Optional
from uuid import uuid4

from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


# ============================================
# 数据结构定义
# ============================================

class ValueMapping(TypedDict):
    """值映射结果"""
    user_term: str          # 用户输入的术语
    db_value: str           # 数据库中的实际值
    table_name: str         # 表名
    column_name: str        # 列名
    score: float            # 匹配置信度


class SQLCandidate(TypedDict):
    """SQL 候选"""
    id: str                 # 候选 ID
    sql: str                # SQL 语句
    explanation: str        # SQL 解释
    confidence: float       # 置信度
    strategy: str           # 生成策略 (direct/decompose/...)


class ClarifyOption(TypedDict):
    """澄清选项"""
    question: str           # 澄清问题
    options: list[str]      # 可选项


class ExecutionResult(TypedDict):
    """执行结果"""
    success: bool           # 是否成功
    data: list[dict]        # 查询结果数据
    row_count: int          # 行数
    columns: list[str]      # 列名
    execution_time_ms: int  # 执行时间(毫秒)
    error: Optional[str]    # 错误信息


class ChartConfig(TypedDict):
    """图表配置"""
    chart_type: str         # 图表类型 (bar/line/pie/...)
    title: str              # 标题
    config: dict            # ECharts 配置


class SubTaskDef(TypedDict):
    """子任务定义"""
    id: str                 # 子任务 ID
    query: str              # 子任务查询
    task_type: str          # 任务类型
    dependencies: list[str] # 依赖的子任务 ID
    status: str             # pending/running/completed/failed
    result: Optional[dict]  # 执行结果


class TaskExecutionPlan(TypedDict):
    """任务执行计划"""
    task_type: str                      # simple/complex/comparison/trend
    needs_decomposition: bool           # 是否需要拆解
    subtasks: list[SubTaskDef]          # 子任务列表
    execution_order: list[list[str]]    # 执行顺序 (分层，同层可并行)
    aggregation_strategy: str           # union/join/merge/none


# ============================================
# LangGraph 状态定义
# ============================================

class DataPilotState(TypedDict):
    """
    DataPilot 工作流全局状态

    状态流转:
    1. 用户输入 -> Supervisor 路由
    2. Ambi-Resolver 检测歧义 (可选)
    3. Data Sniper 获取 Schema + 值映射
    4. Logic Architect 生成 SQL
    5. Judge 校验 + 执行
    6. Viz Expert 生成图表 (可选)
    """

    # ============================================
    # 输入
    # ============================================
    query: str                                      # 用户自然语言问题
    user_id: str                                    # 用户 ID
    tenant_id: str                                  # 租户 ID
    database: str                                   # 目标数据库名称

    # ============================================
    # 消息历史 (LangGraph 消息累加)
    # ============================================
    messages: Annotated[list, add_messages]         # 对话消息历史

    # ============================================
    # 感知层输出 (Data Sniper)
    # ============================================
    schema_context: str                             # 剪枝后的 DDL
    relevant_tables: list[str]                      # 相关表列表
    value_mappings: dict[str, ValueMapping]         # 值映射 {用户术语: 映射结果}

    # LLM-Native 上下文 (来自 SchemaIntrospector)
    time_context: str                               # 时间字段上下文
    enum_context: str                               # 枚举字段上下文

    # ============================================
    # Supervisor 输出 (任务拆解)
    # ============================================
    task_plan: Optional[TaskExecutionPlan]          # 任务执行计划
    subtask_results: dict[str, dict]                # 子任务结果 {task_id: result}
    intent_result: Optional[dict]                   # 意图分类结果

    # ============================================
    # 决策层输出 (Logic Architect + Judge)
    # ============================================
    candidates: list[SQLCandidate]                  # SQL 候选列表
    winner_sql: Optional[str]                       # 获胜 SQL
    execution_plan: Optional[str]                   # EXPLAIN 执行计划

    # ============================================
    # 执行层输出
    # ============================================
    execution_result: Optional[ExecutionResult]     # SQL 执行结果
    chart_config: Optional[ChartConfig]             # 图表配置
    insight_text: Optional[str]                     # 数据洞察文本

    # ============================================
    # CodeAct 可视化输出 (E2B 沙箱)
    # ============================================
    python_code: Optional[str]                      # 生成的 Python 代码
    sandbox_result: Optional[dict]                  # 沙箱执行结果 {success, stdout, stderr, error, executor}

    # ============================================
    # 缓存信息
    # ============================================
    cache_hit: bool                                 # 是否命中缓存
    cache_similarity: float                         # 缓存相似度 (1.0 = 精确匹配)

    # ============================================
    # 控制流
    # ============================================
    current_agent: str                              # 当前执行的 Agent
    next_agent: Optional[str]                       # 下一个 Agent
    clarify_needed: bool                            # 是否需要澄清
    clarify_options: Optional[ClarifyOption]        # 澄清选项
    human_handoff: bool                             # 是否转人工

    # ============================================
    # 错误处理与重试
    # ============================================
    retries: dict[str, int]                         # 重试计数 {"architect": 0, "judge": 0}
    error_context: Optional[str]                    # 错误上下文
    last_error: Optional[str]                       # 最后一次错误

    # ============================================
    # 元数据
    # ============================================
    trace_id: str                                   # 追踪 ID
    created_at: str                                 # 创建时间
    updated_at: str                                 # 更新时间

    # ============================================
    # Agent 执行时间追踪
    # ============================================
    agent_timings: dict[str, dict]                  # Agent 执行时间 {agent_id: {start_time, end_time, duration_ms}}

    # ============================================
    # 可视化模式
    # ============================================
    viz_mode: str                                   # 可视化模式: echarts / codeact


def create_initial_state(
    query: str,
    user_id: str = "anonymous",
    tenant_id: str = "default",
    database: str = "default",
    viz_mode: str = "echarts",
) -> DataPilotState:
    """
    创建初始状态

    Args:
        query: 用户问题
        user_id: 用户 ID
        tenant_id: 租户 ID
        database: 目标数据库
        viz_mode: 可视化模式 (echarts/codeact)

    Returns:
        初始化的状态字典
    """
    now = datetime.utcnow().isoformat()

    return DataPilotState(
        # 输入
        query=query,
        user_id=user_id,
        tenant_id=tenant_id,
        database=database,

        # 消息历史
        messages=[],

        # 感知层
        schema_context="",
        relevant_tables=[],
        value_mappings={},
        time_context="",
        enum_context="",

        # Supervisor
        task_plan=None,
        subtask_results={},
        intent_result=None,

        # 决策层
        candidates=[],
        winner_sql=None,
        execution_plan=None,

        # 执行层
        execution_result=None,
        chart_config=None,
        insight_text=None,

        # CodeAct 输出
        python_code=None,
        sandbox_result=None,

        # 缓存
        cache_hit=False,
        cache_similarity=0.0,

        # 控制流
        current_agent="supervisor",
        next_agent=None,
        clarify_needed=False,
        clarify_options=None,
        human_handoff=False,

        # 错误处理
        retries={"architect": 0, "judge": 0},
        error_context=None,
        last_error=None,

        # 元数据
        trace_id=str(uuid4()),
        created_at=now,
        updated_at=now,

        # Agent 执行时间追踪
        agent_timings={},

        # 可视化模式
        viz_mode=viz_mode,
    )


# ============================================
# Pydantic 模型 (用于 API)
# ============================================

class QueryRequest(BaseModel):
    """查询请求"""
    query: str = Field(..., description="用户自然语言问题")
    user_id: str = Field(default="anonymous", description="用户 ID")
    tenant_id: str = Field(default="default", description="租户 ID")
    database: str = Field(default="default", description="目标数据库")
    session_id: Optional[str] = Field(default=None, description="会话 ID (用于继续对话)")
    viz_mode: Literal["echarts", "codeact"] = Field(default="echarts", description="可视化模式: echarts(快速) / codeact(智能)")


class LLMCallDetail(BaseModel):
    """LLM 调用详情"""
    model: str = Field(default="", description="模型名称")
    prompt: str = Field(default="", description="发送的 prompt")
    response: str = Field(default="", description="LLM 响应")
    input_tokens: int = Field(default=0, description="输入 token 数")
    output_tokens: int = Field(default=0, description="输出 token 数")
    total_tokens: int = Field(default=0, description="总 token 数")
    latency_ms: float = Field(default=0, description="调用延迟(毫秒)")
    cost_usd: float = Field(default=0, description="成本(美元)")


class AgentTrace(BaseModel):
    """单个 Agent 的追踪数据"""
    agent: str = Field(..., description="Agent 名称")
    display_name: str = Field(default="", description="显示名称")
    description: str = Field(default="", description="Agent 描述")
    status: Literal["pending", "running", "success", "error", "skipped"] = Field(default="pending")
    start_time: Optional[str] = Field(default=None, description="开始时间")
    end_time: Optional[str] = Field(default=None, description="结束时间")
    duration_ms: Optional[float] = Field(default=None, description="耗时(毫秒)")
    input_data: Optional[dict] = Field(default=None, description="输入数据")
    output_data: Optional[dict] = Field(default=None, description="输出数据")
    error: Optional[str] = Field(default=None, description="错误信息")
    logs: list[dict] = Field(default_factory=list, description="日志列表")
    # 新增：LLM 调用详情
    llm_calls: list[LLMCallDetail] = Field(default_factory=list, description="LLM 调用详情列表")
    # 新增：详细步骤
    steps: list[dict] = Field(default_factory=list, description="执行步骤详情")


class PipelineTrace(BaseModel):
    """完整的流水线追踪数据"""
    trace_id: str = Field(..., description="追踪 ID")
    start_time: str = Field(..., description="开始时间")
    end_time: Optional[str] = Field(default=None, description="结束时间")
    total_duration_ms: Optional[float] = Field(default=None, description="总耗时(毫秒)")
    agents: dict[str, AgentTrace] = Field(default_factory=dict, description="各 Agent 追踪数据")
    logs: list[dict] = Field(default_factory=list, description="全局日志")


class QueryResponse(BaseModel):
    """查询响应"""
    session_id: str = Field(..., description="会话 ID")
    trace_id: str = Field(..., description="追踪 ID")

    # 结果
    sql: Optional[str] = Field(default=None, description="生成的 SQL")
    data: Optional[list[dict]] = Field(default=None, description="查询结果")
    row_count: int = Field(default=0, description="结果行数")
    chart_config: Optional[dict] = Field(default=None, description="图表配置")
    insight: Optional[str] = Field(default=None, description="数据洞察")
    python_code: Optional[str] = Field(default=None, description="可视化 Python 代码（沙箱执行用）")
    sandbox_result: Optional[dict] = Field(default=None, description="沙箱执行结果 stdout/stderr/error")

    # 状态
    status: Literal["success", "clarify_needed", "error", "human_handoff"] = Field(
        default="success", description="响应状态"
    )
    clarify_options: Optional[ClarifyOption] = Field(default=None, description="澄清选项")
    error_message: Optional[str] = Field(default=None, description="错误信息")

    # 追踪数据
    trace: Optional[PipelineTrace] = Field(default=None, description="流水线追踪数据")


class ClarifyRequest(BaseModel):
    """澄清请求"""
    session_id: str = Field(..., description="会话 ID")
    selected_option: str = Field(..., description="用户选择的选项")


# 导出
__all__ = [
    "DataPilotState",
    "ValueMapping",
    "SQLCandidate",
    "ClarifyOption",
    "ExecutionResult",
    "ChartConfig",
    "SubTaskDef",
    "TaskExecutionPlan",
    "AgentTrace",
    "PipelineTrace",
    "create_initial_state",
    "QueryRequest",
    "QueryResponse",
    "ClarifyRequest",
]
