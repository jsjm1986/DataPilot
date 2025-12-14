<p align="center">
  <h1 align="center">DataPilot</h1>
  <p align="center">
    <strong>企业级 Agentic BI 平台 - 用自然语言驾驭数据</strong>
  </p>
  <p align="center">
    基于 LangGraph 多 Agent 协作架构，实现从自然语言到数据洞察的端到端自动化
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/React-18-61dafb.svg" alt="React">
  <img src="https://img.shields.io/badge/LangGraph-1.x-green.svg" alt="LangGraph">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</p>

<!--
## 产品截图


Apache-2.0 License

---

**开源说明**
- **License**: 本仓库将使用 **Apache-2.0** 协议开源（见仓库根目录的 `LICENSE` 文件）。
- **脱敏与公开范围**: 公开时我们仅发布后端相关代码（`src/`、后端脚本、`pyproject.toml` 等），不包含前端 `frontend/`、本地 `data/` 目录或任何包含真实密钥的文件（例如 `.env`）。
- **生成脱敏导出**: 仓库已包含脚本 `scripts/prepare_opensource.py`，用于生成脱敏的导出目录。运行示例：

```bash
python scripts/prepare_opensource.py opensource_export
```

- **检查与上传**: 运行后请检查 `opensource_export/`，确认 `.env.example`、`docker-compose.yml` 中为占位符。将 `opensource_export/` 初始化为新仓库并推送到你的 GitHub：

```bash
cd opensource_export
git init
git add .
git commit -m "Initial open-source release - backend only (sanitized)"
git remote add origin https://github.com/jsjm1986/DataPilot.git
git branch -M main
git push -u origin main
```

如果你希望我继续生成导出并打包（不能直接代为推送远程仓库），我可以帮你校验并准备好要上传的导出包。

![可视化结果](docs/images/visualization.png)
-->

---

## 为什么需要 DataPilot？

传统 BI 工具面临的困境：

| 痛点 | 传统方案 | DataPilot 方案 |
|------|---------|---------------|
| **技术门槛高** | 需要掌握 SQL、理解数据模型 | 自然语言提问，零技术门槛 |
| **查询歧义** | 用户猜测字段含义，结果不准确 | 智能歧义消解，主动澄清确认 |
| **复杂查询** | 多表关联、嵌套查询难以编写 | 自动分解复杂逻辑，多路生成 |
| **结果呈现** | 手动选择图表类型和配置 | AI 自动推荐最佳可视化方案 |
| **错误处理** | 报错后用户无从下手 | 自愈机制，智能修正重试 |

**DataPilot 的核心价值**：让业务人员像与数据分析师对话一样，用自然语言探索数据，获得准确、可视化的洞察结果。

---

## 与其他 Text2SQL 项目对比

| 特性 | DataPilot | 传统 Text2SQL | LangChain SQL Agent | Vanna.ai |
|------|-----------|--------------|---------------------|----------|
| **架构模式** | 多 Agent 协作 (LangGraph) | 单模型端到端 | 单 Agent + Tools | RAG + 单模型 |
| **歧义处理** | Human-in-the-Loop 主动澄清 | 无/猜测 | 无 | 无 |
| **错误恢复** | 三级自愈 + ReAct 推理 | 无/简单重试 | 简单重试 | 无 |
| **元数据获取** | AI 自动提取 (SchemaIntrospector) | 手动配置 | 手动配置 | 手动训练 |
| **Prompt 优化** | DSPy 自进化 | 手动调优 | 手动调优 | 需要训练数据 |
| **可视化** | 双模式 (ECharts + CodeAct) | 无 | 无 | 基础图表 |
| **缓存策略** | 语义缓存 (向量相似度) | 无/精确匹配 | 无 | 无 |
| **复杂查询** | MAC-SQL 分解 + 多路生成 | 单次生成 | 单次生成 | 单次生成 |
| **实时反馈** | WebSocket 进度追踪 | 无 | 无 | 无 |

### DataPilot 的核心优势

**1. 生产级可靠性**
```
传统方案: 用户提问 → 生成 SQL → 执行失败 → 用户困惑
DataPilot: 用户提问 → 歧义检测 → 主动澄清 → 多路生成 → 自动修正 → 成功返回
```

**2. 零配置适应新数据库**
- 传统方案需要 DBA 手动维护数据字典、编写示例 SQL
- DataPilot 通过 SchemaIntrospector 自动提取元数据，即插即用

**3. 持续自我优化**
- 传统方案的 Prompt 需要人工迭代调优
- DataPilot 通过 DSPy 自动收集成功案例，持续优化生成质量

**4. 端到端解决方案**
- 不仅生成 SQL，还提供可视化、洞察分析
- 完整的前端界面，开箱即用

---

## 核心创新

### 1. LLM-Native 设计哲学

DataPilot 采用 **"LLM 优先"** 的设计理念，摒弃传统的硬编码规则：

```
传统方案：关键词匹配 → 规则引擎 → 模板填充
DataPilot：语义理解 → LLM 推理 → 动态生成
```

**关键实现**：
- **SchemaIntrospector**：动态提取数据库元信息（表结构、时间字段、枚举值）
- **真实数据驱动**：使用样本值帮助 LLM 理解数据格式和业务语义
- **配置化业务词汇**：支持加载行业特定术语，无需修改代码

#### AI 驱动的元数据生成

传统方案需要 DBA 手动维护数据字典，DataPilot 通过 **SchemaIntrospector** 自动提取：

```python
# 自动识别时间字段（不依赖硬编码规则）
TIME_TYPES = {
    'mysql': ['date', 'datetime', 'timestamp'],
    'postgresql': ['date', 'timestamp', 'timestamptz'],
    ...
}

# 自动提取枚举值（distinct_count <= 50 的列）
if column.distinct_count <= 50:
    column.is_enum_like = True
    column.sample_values = [...]  # 自动获取所有可选值
```

**生成的上下文示例**：
```
## 表: orders
   行数: 125,000
   列:
   - id (int) [主键]
   - status (varchar) [枚举]
     可选值: ['pending', 'paid', 'shipped', 'completed']
   - created_at (datetime) [时间字段]
   - customer_id (int) [外键 -> customers.id]
```

LLM 根据这些真实元数据生成 SQL，而非依赖预设规则

### 2. 多 Agent 协作架构

基于 **LangGraph 1.x** 构建的 Supervisor Pattern 多智能体系统：

```
┌─────────────────────────────────────────────────────────────────┐
│                        Supervisor (总控大脑)                      │
│  ┌─────────────┬─────────────┬─────────────┬─────────────┐     │
│  │ 意图识别    │ 任务路由    │ 状态管理    │ 人机交互    │     │
│  └─────────────┴─────────────┴─────────────┴─────────────┘     │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│  AmbiResolver │    │  DataSniper   │    │LogicArchitect │
│   歧义消解    │    │   数据侦察    │    │   SQL构建     │
└───────────────┘    └───────────────┘    └───────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│    Judge      │    │   VizExpert   │    │  CacheCheck   │
│   执行校验    │    │   可视化专家   │    │   缓存检查    │
└───────────────┘    └───────────────┘    └───────────────┘
```

**三级复杂度路由**：
| 复杂度 | 路由策略 | 适用场景 |
|--------|---------|---------|
| 简单 | 串行流程 | 单表查询、简单聚合 |
| 中等 | 并行执行 | 多表关联、标准分析 |
| 复杂 | ReAct 推理 | 嵌套查询、多步推理 |

### 3. Human-in-the-Loop 原生支持

利用 LangGraph 的 `interrupt()` 机制，实现优雅的人机协作：

```python
# 当检测到歧义时，工作流自动中断等待用户确认
if ambiguity_detected:
    return interrupt({
        "question": "您指的 Q1 是？",
        "options": ["自然年 (1-3月)", "财年 (4-6月)"]
    })
```

**5 种歧义类型智能识别**：
- `time` - 时间范围歧义（Q1、上月、近期）
- `metric` - 指标定义歧义（销售额、GMV、营收）
- `scope` - 数据范围歧义（全部、有效、已完成）
- `granularity` - 粒度歧义（按天、按周、按月）
- `entity` - 实体歧义（产品名、客户名的模糊匹配）

### 4. 双模式可视化引擎

**ECharts 快速模式**：
- LLM 直接生成 ECharts JSON 配置
- 支持 12 种标准图表类型
- 毫秒级响应，稳定可靠

**CodeAct 智能模式**：
- LLM 生成 Python 代码（pandas + plotly/pyecharts）
- 支持复杂数据清洗和自定义可视化
- E2B 安全沙箱执行，隔离风险

```
用户: "对比各区域销售趋势，标注异常点"

CodeAct 模式自动生成:
├── 数据清洗 (空值处理、类型转换)
├── 异常检测 (IQR/Z-score)
├── 趋势计算 (移动平均)
└── 图表生成 (带标注的多系列折线图)
```

### 5. 语义缓存系统

**双重匹配策略**：
```
查询请求 ──┬── 精确匹配 (哈希) ──→ 命中 → 直接返回
           │
           └── 语义匹配 (向量相似度 ≥ 0.85) ──→ 命中 → 返回相似结果
                                              │
                                              └── 未命中 → 执行完整流程
```

**多层存储架构**：
| 层级 | 存储 | 用途 |
|------|------|------|
| L1 | 内存 | 热点查询快速响应 |
| L2 | Redis | 分布式精确匹配 |
| L3 | LanceDB | 向量语义检索 |

### 6. DSPy 自进化系统

DataPilot 集成 **DSPy** 框架，实现 Prompt 和 Few-shot 示例的自动优化：

```
┌─────────────────────────────────────────────────────────────┐
│                    自进化闭环                                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   用户查询 ──→ SQL 生成 ──→ 执行成功 ──→ 收集训练样本       │
│       ↑                                        │            │
│       │                                        ↓            │
│   部署新模块 ←── 评估通过 ←── DSPy 优化 ←── 样本积累 ≥ 50   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**核心组件**：

| 组件 | 功能 |
|------|------|
| **TrainingDataCollector** | 自动收集成功查询作为训练数据 |
| **SelfEvolutionEngine** | 协调数据收集、优化、部署的完整流程 |
| **ModuleVersionManager** | 模块版本管理和回滚 |
| **DSPyTrainer** | 支持 BootstrapFewShot、RandomSearch 等优化策略 |

**自进化配置**：
```python
EvolutionConfig(
    min_samples_for_evolution=50,    # 最少样本数触发优化
    evolution_interval_hours=24,      # 自动进化间隔
    accuracy_threshold=0.7,           # 部署阈值
    rollback_on_degradation=True,     # 性能下降自动回滚
)
```

**优化效果**：
- 自动从用户查询中学习，无需人工标注
- 持续优化 Prompt，提升 SQL 生成准确率
- 版本管理支持 A/B 测试和快速回滚

---

## 系统架构

### 完整工作流程

```mermaid
graph TD
    User[用户] -->|自然语言| Gateway[API Gateway]
    Gateway --> Cache{语义缓存}
    Cache -->|命中| Response[返回结果]
    Cache -->|未命中| Supervisor

    subgraph "Agent 协作层"
        Supervisor[Supervisor] --> Intent[意图分类]
        Intent -->|需澄清| Ambi[AmbiResolver]
        Ambi -->|中断| User
        User -->|确认| Ambi
        Ambi --> Sniper[DataSniper]
        Intent -->|明确| Sniper
        Sniper --> Architect[LogicArchitect]
        Architect --> Judge[Judge]
        Judge -->|失败| Architect
        Judge -->|成功| Viz[VizExpert]
    end

    subgraph "执行层"
        Viz -->|ECharts| ECharts[配置生成]
        Viz -->|CodeAct| Sandbox[E2B 沙箱]
        Sandbox --> Python[Python 执行]
    end

    ECharts --> Response
    Python --> Response
    Response --> User
```

### Agent 职责矩阵

| Agent | 职责 | 输入 | 输出 | 关键技术 |
|-------|------|------|------|---------|
| **CacheCheck** | 缓存检查 | 用户查询 | 缓存结果/未命中 | 语义向量检索 |
| **IntentClassifier** | 意图分类 | 用户查询 | 意图类型+复杂度 | LLM 分类 |
| **AmbiResolver** | 歧义消解 | 查询+Schema | 明确意图/澄清选项 | interrupt() |
| **DataSniper** | Schema 侦察 | 查询+全量 Schema | 精简 Schema+值映射 | Rerank+图剪枝 |
| **LogicArchitect** | SQL 生成 | Schema+映射 | 候选 SQL 列表 | DSPy+MAC-SQL |
| **Judge** | 执行校验 | SQL+执行结果 | 获胜 SQL/修正建议 | EXPLAIN 分析 |
| **VizExpert** | 可视化 | 结果集 | 图表配置/代码 | ECharts/CodeAct |

---

## 技术栈

### 后端
| 组件 | 技术选型 | 选型理由 |
|------|---------|---------|
| 工作流编排 | LangGraph 1.x | 原生支持 interrupt、Store、Command |
| 提示优化 | DSPy | 自动 Prompt/CoT 优化，替代手写模板 |
| 向量存储 | LanceDB | 轻量级、支持持久化、无需额外服务 |
| 代码沙箱 | E2B | 云端隔离执行、资源限制、安全可控 |
| LLM 推理 | DeepSeek-V3 | 长链推理能力强、成本效益高 |
| Web 框架 | FastAPI | 异步支持、自动文档、类型安全 |
| 缓存 | Redis | 分布式缓存、发布订阅 |

### 前端
| 组件 | 技术选型 | 选型理由 |
|------|---------|---------|
| 框架 | React 18 + TypeScript | 类型安全、生态丰富 |
| 状态管理 | Zustand | 轻量、支持持久化 |
| UI 组件 | Ant Design | 企业级组件库 |
| 图表 | ECharts | 功能强大、配置灵活 |
| 实时通信 | WebSocket | 低延迟、双向通信 |

---

## 支持的数据库

| 数据库 | 状态 | 说明 |
|--------|------|------|
| MySQL | ✅ 完整支持 | 包括 MariaDB |
| PostgreSQL | ✅ 完整支持 | 包括 TimescaleDB |
| SQLite | ✅ 完整支持 | 开发测试首选 |
| SQL Server | ✅ 支持 | 需要 ODBC 驱动 |
| ClickHouse | ✅ 支持 | 大数据分析场景 |
| DuckDB | ✅ 支持 | 嵌入式 OLAP |

---

## 快速开始

### 环境要求
- Python 3.11+
- Node.js 18+
- Redis (可选，用于分布式缓存)

### 1. 克隆项目
```bash
git clone https://github.com/your-org/datapilot.git
cd datapilot
```

### 2. 后端配置
```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -e .

# 配置环境变量
cp .env.example .env
# 编辑 .env 文件

# 启动服务
python -m uvicorn src.datapilot.main:app --host 0.0.0.0 --port 8000
```

### 3. 前端配置
```bash
cd frontend
npm install
npm run dev
```

### 4. 访问应用
- 前端界面: http://localhost:5173
- API 文档: http://localhost:8000/docs

### 环境变量说明

```env
# LLM 配置
OPENAI_API_KEY=your-api-key
OPENAI_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat

# 数据库配置
DATABASE_URL=postgresql://user:pass@localhost:5432/dbname

# 缓存配置
REDIS_URL=redis://localhost:6379
SEMANTIC_CACHE_ENABLED=true
SEMANTIC_CACHE_THRESHOLD=0.85

# 沙箱配置 (CodeAct 模式)
E2B_API_KEY=your-e2b-key

# 安全配置
JWT_SECRET_KEY=your-secret-key
```

---

## 使用示例

### 基本查询
```
用户: 查询上个月销售额最高的前10个产品

执行流程:
[CacheCheck]      检查缓存 ────────────────── 未命中
[IntentClassifier] 识别意图 ────────────────── 数据查询 (简单)
[DataSniper]      定位表 ──────────────────── products, sales
[LogicArchitect]  生成 SQL ─────────────────── SELECT ... ORDER BY ... LIMIT 10
[Judge]           执行校验 ────────────────── 通过 (10 rows, 23ms)
[VizExpert]       生成图表 ────────────────── 柱状图

结果: 数据表格 + 柱状图 + SQL 语句
```

### 歧义消解
```
用户: 查询 Q1 销售数据

[AmbiResolver] 检测到时间歧义

DataPilot: 请问您指的 Q1 是？
  ○ 自然年 Q1 (1-3月)
  ○ 财年 Q1 (4-6月)

用户: 选择 "自然年 Q1"

继续执行...
```

### 复杂查询 (ReAct 模式)
```
用户: 分析各区域销售趋势，找出增长最快的区域，并预测下季度销售额

[Supervisor] 复杂度评估 ──────────────────── 高复杂度，启用 ReAct

ReAct 推理过程:
  Thought: 需要分解为多个子任务
  Action: decompose_task
  Result: [区域销售汇总, 增长率计算, 趋势预测]

  Thought: 先获取区域销售数据
  Action: generate_sql (区域销售汇总)
  Result: SQL_1 生成成功

  Thought: 计算同比增长率
  Action: generate_sql (增长率计算)
  Result: SQL_2 生成成功

  Thought: 执行并验证
  Action: execute_sql
  Result: 数据获取成功

  Thought: 生成可视化和预测
  Action: visualize (CodeAct 模式)
  Result: 趋势图 + 预测曲线

最终结果: 多图表组合 + 洞察摘要
```

---

## 工程实践

### 错误处理与自愈机制

```
┌─────────────────────────────────────────────────────────┐
│                    三级自愈策略                          │
├─────────────────────────────────────────────────────────┤
│  Level 1: Agent 级重试                                  │
│  ├── 每个 Agent 最多 3 次重试                           │
│  └── 错误上下文传递给下次尝试                            │
├─────────────────────────────────────────────────────────┤
│  Level 2: Supervisor ReAct                              │
│  ├── 失败后进入推理模式                                  │
│  ├── 分析错误原因，调整策略                              │
│  └── 调用其他 Agent 协助修复                             │
├─────────────────────────────────────────────────────────┤
│  Level 3: Human Handoff                                 │
│  ├── 连续 3 次失败触发                                   │
│  ├── 保存完整上下文                                      │
│  └── 转人工处理，支持断点续执行                          │
└─────────────────────────────────────────────────────────┘
```

### 安全设计

| 安全层 | 措施 |
|--------|------|
| SQL 注入防护 | SQLGlot AST 解析，拦截 DDL/DML |
| 代码执行隔离 | E2B 沙箱，禁网、限时、限内存 |
| 成本熔断 | EXPLAIN 分析，rows > 1M 拒绝执行 |
| 数据脱敏 | Schema/样本数据自动脱敏 |
| 权限控制 | 数据库只读账号，JWT 认证 |

### 可观测性

- **链路追踪**: LangSmith 全链路追踪
- **指标监控**: Prometheus + Grafana
- **审计日志**: 结构化日志，保留 90 天

---

## 项目结构

```
datapilot/
├── src/datapilot/
│   ├── api/                    # API 路由
│   │   ├── main.py             # FastAPI 入口
│   │   ├── query_routes.py     # 查询 API
│   │   └── websocket.py        # WebSocket 处理
│   ├── agents/                 # Agent 实现
│   │   ├── ambi_resolver.py    # 歧义消解
│   │   ├── data_sniper.py      # 数据侦察
│   │   ├── logic_architect.py  # SQL 生成
│   │   ├── judge.py            # 执行校验
│   │   └── viz_expert.py       # 可视化
│   ├── core/                   # 核心模块
│   │   ├── graph.py            # LangGraph 工作流
│   │   ├── state.py            # 全局状态
│   │   ├── supervisor.py       # 中央协调器
│   │   └── supervisor_react.py # ReAct 推理
│   └── cache/                  # 缓存模块
│       └── semantic_cache.py   # 语义缓存
├── frontend/
│   ├── src/
│   │   ├── components/         # React 组件
│   │   ├── pages/              # 页面
│   │   ├── stores/             # Zustand 状态
│   │   └── hooks/              # 自定义 Hooks
│   └── package.json
├── tests/                      # 测试
├── deploy/                     # 部署配置
└── pyproject.toml
```

---

## 开发指南

### 添加新 Agent

1. 在 `src/datapilot/agents/` 创建 Agent 文件
2. 继承 `BaseAgent`，实现 `run()` 方法
3. 在 `core/graph.py` 注册节点
4. 更新前端 Agent 状态展示

### 运行测试
```bash
# 后端测试
pytest tests/ -v

# 前端测试
cd frontend && npm test
```

### 代码规范
```bash
# 后端
black src/ && ruff check src/

# 前端
cd frontend && npm run lint
```

---

## 路线图

- [x] 多 Agent 协作架构
- [x] Human-in-the-Loop 歧义消解
- [x] 双模式可视化引擎
- [x] 语义缓存系统
- [x] DSPy 自进化系统
- [x] AI 驱动的元数据生成
- [ ] 多数据源联邦查询
- [ ] 自然语言报表定义
- [ ] 移动端适配

---

## 贡献指南

我们欢迎各种形式的贡献！

### 如何贡献

1. **Fork** 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 提交 **Pull Request**

### 贡献类型

- 🐛 Bug 修复
- ✨ 新功能
- 📝 文档改进
- 🎨 代码优化
- 🌐 国际化

### 开发规范

- 后端代码遵循 PEP 8，使用 Black 格式化
- 前端代码使用 ESLint + Prettier
- 提交信息遵循 [Conventional Commits](https://www.conventionalcommits.org/)

---

## 许可证

MIT License

---

## Star History

如果这个项目对你有帮助，请给我们一个 ⭐️

---

<p align="center">
  <strong>DataPilot</strong> - 让数据触手可及
  <br>
  <sub>Built with ❤️ by the DataPilot Team</sub>
</p>
