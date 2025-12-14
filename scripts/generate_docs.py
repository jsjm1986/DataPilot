# -*- coding: utf-8 -*-
"""
Generate FEATURES.md documentation
"""
from pathlib import Path

CONTENT = """# DataPilot 项目功能文档

## 项目概述

**DataPilot** 是一个企业级 Agentic BI 平台，实现自然语言到 SQL 的全链路自动化转换，并支持数据可视化。

- **项目名称**: DataPilot (Deep Semantic Query Logic)
- **版本**: 0.1.0
- **核心功能**: 自然语言 -> SQL -> 可视化图表

---

## 一、系统架构

### 1.1 技术栈

| 层级 | 技术 |
|------|------|
| 后端框架 | FastAPI + Uvicorn |
| LLM | DeepSeek (OpenAI 兼容接口) |
| Embedding | SiliconFlow API (BAAI/bge-large-zh-v1.5) |
| Rerank | SiliconFlow API (Qwen/Qwen3-Reranker-8B) |
| 数据库 | SQLite / MySQL / PostgreSQL |
| ORM | SQLAlchemy (异步) |
| 前端 | React + TypeScript + Vite |
| UI 组件 | Ant Design |
| 图表 | ECharts |

### 1.2 目录结构

```
src/datapilot/           # 后端源码
├── agents/            # Agent 实现
│   ├── data_sniper.py     # Schema 剪枝
│   ├── logic_architect.py # SQL 生成
│   ├── judge.py           # SQL 校验
│   └── viz_expert.py      # 图表推荐
├── api/               # API 路由
│   ├── routes.py          # REST API
│   └── websocket.py       # WebSocket
├── cache/             # 语义缓存
│   └── semantic_cache.py
├── config/            # 配置管理
│   └── settings.py
├── db/                # 数据库连接
│   └── connector.py
├── llm/               # LLM 集成
│   ├── deepseek.py
│   └── embeddings.py
├── mcp/tools/         # MCP 工具
│   ├── list_tables.py
│   ├── get_ddl.py
│   ├── search_values.py
│   ├── read_only_query.py
│   └── plan_explain.py
└── security/          # 安全模块
    ├── injection_guard.py
    └── sanitizer.py

frontend/src/          # 前端源码
├── components/
│   ├── QueryInput/
│   ├── ResultTable/
│   ├── SQLPreview/
│   └── ChartView/
├── pages/Home/
├── hooks/useQuery.ts
├── services/api.ts
└── types/index.ts
```

---

## 二、后端功能模块

### 2.1 配置管理 (config/settings.py)

使用 Pydantic Settings 管理配置，支持 .env 文件。

**主要配置**:
- DeepSeek: api_key, base_url, model, temperature
- Embedding: api_key (SiliconFlow), model (bge-large-zh-v1.5)
- Rerank: model (Qwen/Qwen3-Reranker-8B)
- 数据库: sqlite_database, default_db_type
- 缓存: similarity_threshold (0.85), ttl_minutes (30)
- API: host, port, cors_origins

### 2.2 数据库连接器 (db/connector.py)

**支持的数据库**:
- SQLite (默认)
- MySQL
- PostgreSQL

**核心类**:
- `DatabaseConnector`: 抽象基类
- `SQLiteConnector`: SQLite 实现
- `MySQLConnector`: MySQL 实现
- `PostgreSQLConnector`: PostgreSQL 实现
- `DatabaseManager`: 管理多数据库连接

**特性**:
- 异步支持 (aiosqlite, aiomysql, asyncpg)
- 自动 LIMIT 限制
- Schema 自动提取

### 2.3 Agent 系统

#### DataSniper (agents/data_sniper.py)

**职责**: Schema 剪枝和值映射

**方法**:
- `get_relevant_schema(query, top_k)`: 使用 Rerank 找相关表
- `map_values(query, entities)`: 实体映射到数据库值
- `analyze(query)`: 完整分析流程

#### LogicArchitect (agents/logic_architect.py)

**职责**: 自然语言转 SQL

**方法**:
- `generate_sql(query, schema, database)`: 生成 SQL
- `refine_sql(sql, error, schema)`: 修正 SQL

#### Judge (agents/judge.py)

**职责**: SQL 校验和安全检查

**检测**:
- 危险关键字: DROP, DELETE, UPDATE, INSERT 等
- 注入模式: UNION, 注释, 永真条件等

**方法**:
- `validate(sql)`: 验证 SQL
- `analyze_cost(sql)`: 成本分析
- `judge(sql)`: 完整审判

#### VizExpert (agents/viz_expert.py)

**职责**: 图表推荐和 ECharts 配置

**支持图表**: bar, line, pie, scatter, area, table

**方法**:
- `recommend_chart(data, query)`: 推荐图表类型
- `generate_echarts_config(...)`: 生成配置
- `analyze(data, query)`: 完整分析

### 2.4 LLM 集成

#### DeepSeek (llm/deepseek.py)

使用 LangChain 集成 DeepSeek API。

**方法**:
- `chat(messages)`: 异步聊天
- `generate_sql(query, schema, dialect)`: SQL 生成

#### Embedding (llm/embeddings.py)

使用 SiliconFlow API。

**EmbeddingClient**:
- `embed(texts)`: 批量嵌入
- `embed_single(text)`: 单个嵌入

**RerankClient**:
- `rerank(query, documents, top_n)`: 重排序

### 2.5 MCP 工具 (mcp/tools/)

| 工具 | 功能 |
|------|------|
| list_tables | 获取表列表 |
| get_ddl | 获取表 DDL |
| search_values | 搜索数据库值 |
| read_only_query | 只读查询 |
| plan_explain | 执行计划 |

### 2.6 语义缓存 (cache/semantic_cache.py)

**特性**:
- 精确匹配缓存
- TTL 过期 (30分钟)
- LRU 淘汰
- 命中计数

**方法**:
- `get(query)`: 获取缓存
- `set(query, sql, result, row_count)`: 设置缓存
- `invalidate()`: 清除缓存
- `stats()`: 统计信息

### 2.7 安全模块

#### InjectionGuard (security/injection_guard.py)

**检测模式**:
- 多语句注入
- 注释注入
- UNION 注入
- 布尔注入
- 时间注入

**方法**:
- `check(sql)`: 检查注入
- `sanitize_input(input)`: 清理输入

### 2.8 API 路由 (api/routes.py)

| 端点 | 方法 | 功能 |
|------|------|------|
| /api/v1/health | GET | 健康检查 |
| /api/v1/databases | GET | 数据库列表 |
| /api/v1/schema/{db} | GET | 获取 Schema |
| /api/v1/query | POST | 自然语言查询 |
| /api/v1/execute | POST | 执行 SQL |
| /api/v1/tables/{db} | GET | 表列表 |

**查询流程**:
1. 检查缓存
2. DataSniper - Schema 剪枝
3. LogicArchitect - SQL 生成
4. Judge - SQL 校验
5. 执行 SQL
6. VizExpert - 图表推荐
7. 写入缓存

### 2.9 WebSocket (api/websocket.py)

实时查询，流式进度更新。

**消息类型**: progress, result, error, ping/pong

**阶段**: data_sniper -> logic_architect -> judge -> execute -> viz_expert

---

## 三、前端功能模块

### 3.1 组件

#### QueryInput
- 文本输入框
- 数据库选择
- 示例查询按钮

#### ResultTable
- 动态列生成
- 分页支持
- 行数统计

#### SQLPreview
- SQL 语法高亮
- 复制按钮

#### ChartView (ECharts)
- 柱状图/折线图/饼图
- 图表类型切换
- 渐变色样式
- 工具栏

### 3.2 页面

#### HomePage
- 查询输入
- 结果展示 (Table/Chart 切换)
- 错误提示
- 加载状态

### 3.3 Hooks

#### useQuery
- isLoading: 加载状态
- result: 查询结果
- error: 错误信息
- executeQuery: 执行函数

### 3.4 API 服务 (services/api.ts)

- `query(request)`: 自然语言查询
- `executeSQL(request)`: 执行 SQL
- `getDatabases()`: 数据库列表
- `getSchema(db)`: 获取 Schema

---

## 四、数据库设计

### 4.1 表结构 (8张表)

| 表名 | 说明 |
|------|------|
| categories | 商品分类 |
| products | 商品 |
| customers | 客户 |
| orders | 订单 |
| order_items | 订单明细 |
| sales_daily | 销售日报 |
| inventory_logs | 库存变动 |
| user_behaviors | 用户行为 |

### 4.2 数据量

- 分类: 16 条
- 商品: ~250 条
- 客户: 500 条
- 订单: ~5,400 条
- 订单明细: ~16,000 条
- 销售日报: ~15,000 条
- 用户行为: 5,000 条
- **总计**: ~38,000+ 条

---

## 五、运行指南

### 后端启动

```bash
pip install -r requirements.txt
python scripts/init_sqlite.py
python -m uvicorn src.datapilot.main:app --host 127.0.0.1 --port 8000
```

### 前端启动

```bash
cd frontend
npm install
npm run dev
```

### 访问地址

- 前端: http://localhost:5173
- 后端: http://localhost:8000
- API 文档: http://localhost:8000/docs

---

## 六、已完成功能

### 后端
- [x] 多数据库连接器 (SQLite/MySQL/PostgreSQL)
- [x] Pydantic Settings 配置
- [x] DeepSeek LLM 集成
- [x] SiliconFlow Embedding/Rerank
- [x] DataSniper Agent
- [x] LogicArchitect Agent
- [x] Judge Agent
- [x] VizExpert Agent
- [x] MCP 工具 (5个)
- [x] 语义缓存
- [x] SQL 注入防护
- [x] FastAPI REST API
- [x] WebSocket 实时通信
- [x] SQLite 模拟数据

### 前端
- [x] React + TypeScript + Vite
- [x] Ant Design UI
- [x] QueryInput 组件
- [x] ResultTable 组件
- [x] SQLPreview 组件
- [x] ChartView (ECharts)
- [x] HomePage 页面
- [x] useQuery Hook
- [x] API 服务层

---

## 七、待开发功能

- [ ] 向量检索 (Qdrant)
- [ ] DSPy 优化
- [ ] 歧义消解 Agent
- [ ] 人工兜底机制
- [ ] Kubernetes 部署
- [ ] Prometheus 监控

---

*版本: 0.1.0*
"""

if __name__ == "__main__":
    output_path = Path(__file__).parent.parent / "FEATURES.md"
    output_path.write_text(CONTENT, encoding="utf-8")
    print(f"Created: {output_path}")
