"""
Data Sniper Agent (LLM-Native 版)
负责 Schema 剪枝和值映射

增强功能:
- 基于外键关系的图剪枝
- Rerank + 图扩展的混合策略
- LLM 驱动的实体提取 (替代硬编码正则)
- 配置化的业务词汇映射
- **SchemaIntrospector 集成** - 动态提取数据库元信息

符合 LLM-Native 设计原则:
- 复杂推理统一由 LLM 处理
- 动态提取数据库元信息，无硬编码
- 使用样本值帮助 LLM 理解数据格式
"""

import json
import re
from typing import Optional
from pathlib import Path

from ..db.connector import get_db_manager
from ..db.schema_introspector import SchemaIntrospector, SchemaMetadata
from ..llm.embeddings import get_embedding_client, get_rerank_client
from ..llm.deepseek import get_deepseek_client
from ..mcp.tools import list_tables, get_ddl, search_values
from .schema_graph import SchemaGraph, build_schema_graph
from .vector_index import VectorSchemaIndex


# 实体提取的 System Prompt
ENTITY_EXTRACTION_PROMPT = """你是一个专业的实体提取器。

## 任务
从用户的自然语言查询中提取需要在数据库中查找的实体（如产品名、客户名、地区名等）。

## 输出格式
返回 JSON:
```json
{
    "entities": [
        {"value": "实体值", "type": "entity_type", "context": "上下文说明"}
    ]
}
```

## 实体类型
- product: 产品/商品名称
- customer: 客户/用户名称
- location: 地区/城市/国家
- category: 分类/类别
- time_period: 时间段
- metric: 指标名称
- other: 其他实体

## 注意
- 只提取需要在数据库中查找具体值的实体
- 不要提取通用词汇（如"销量"、"订单"等）
- 引号中的内容通常是实体
- 返回空数组如果没有需要提取的实体
"""


class BusinessKeywordsConfig:
    """
    业务词汇配置管理器

    支持从配置文件加载，实现配置化而非硬编码
    """

    # 默认配置 (作为后备)
    DEFAULT_KEYWORDS = {
        "order": ["order", "订单", "purchase", "buy", "采购"],
        "product": ["product", "商品", "item", "goods", "产品", "货品"],
        "customer": ["customer", "客户", "user", "用户", "会员", "买家"],
        "sale": ["sale", "销售", "revenue", "收入", "营收", "销量"],
        "category": ["category", "分类", "type", "类型", "品类"],
        "inventory": ["inventory", "库存", "stock", "存货"],
        "supplier": ["supplier", "供应商", "vendor", "厂商"],
        "employee": ["employee", "员工", "staff", "职员"],
        "payment": ["payment", "支付", "付款", "结算"],
        "shipping": ["shipping", "物流", "delivery", "配送", "发货"],
    }

    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置

        Args:
            config_path: 配置文件路径 (JSON/YAML)，为 None 时使用默认配置
        """
        self.keywords = self.DEFAULT_KEYWORDS.copy()

        if config_path:
            self._load_config(config_path)

    def _load_config(self, config_path: str):
        """从文件加载配置"""
        path = Path(config_path)
        if not path.exists():
            return

        try:
            with open(path, 'r', encoding='utf-8') as f:
                if path.suffix in ['.yaml', '.yml']:
                    import yaml
                    custom_keywords = yaml.safe_load(f)
                else:
                    custom_keywords = json.load(f)

            if custom_keywords and isinstance(custom_keywords, dict):
                # 合并自定义配置
                for key, values in custom_keywords.items():
                    if key in self.keywords:
                        # 扩展现有关键词
                        self.keywords[key] = list(set(self.keywords[key] + values))
                    else:
                        # 添加新类别
                        self.keywords[key] = values
        except Exception as e:
            print(f"Failed to load business keywords config: {e}")

    def get_keywords(self) -> dict:
        """获取所有业务词汇"""
        return self.keywords

    def add_keywords(self, category: str, keywords: list[str]):
        """动态添加关键词"""
        if category in self.keywords:
            self.keywords[category] = list(set(self.keywords[category] + keywords))
        else:
            self.keywords[category] = keywords


# 全局配置实例
_keywords_config: Optional[BusinessKeywordsConfig] = None


def get_keywords_config(config_path: Optional[str] = None) -> BusinessKeywordsConfig:
    """获取业务词汇配置"""
    global _keywords_config
    if _keywords_config is None:
        _keywords_config = BusinessKeywordsConfig(config_path)
    return _keywords_config


class DataSniper:
    """
    数据侦察 Agent (LLM-Native 版)

    负责 Schema 剪枝和值映射

    核心特性:
    - **SchemaIntrospector 集成**: 动态提取数据库元信息
    - LLM 驱动的实体提取
    - 配置化业务词汇
    - 图剪枝 + 向量索引混合策略

    LLM-Native 设计原则:
    - 不依赖硬编码规则
    - 动态获取数据库结构、时间字段、枚举值、样本值
    - 让 LLM 根据真实数据做决策
    """

    def __init__(
        self,
        database: Optional[str] = None,
        use_graph_pruning: Optional[bool] = None,
        use_vector_index: Optional[bool] = None,
        vector_index_threshold: Optional[int] = None,  # 表数量超过此阈值时启用向量索引
        keywords_config_path: Optional[str] = None,  # 业务词汇配置文件路径
        use_llm_entity_extraction: Optional[bool] = None,  # 是否使用 LLM 提取实体
        use_schema_introspector: bool = True,  # 是否使用 SchemaIntrospector
    ):
        # 从 settings 读取默认配置
        from ..config.settings import get_settings
        settings = get_settings()

        self.database = database
        self.embedding_client = get_embedding_client()
        self.rerank_client = get_rerank_client()
        # 使用 settings 作为默认值，允许参数覆盖
        self.use_graph_pruning = use_graph_pruning if use_graph_pruning is not None else settings.data_sniper_use_graph_pruning
        self.use_vector_index = use_vector_index if use_vector_index is not None else settings.data_sniper_use_vector_index
        self.vector_index_threshold = vector_index_threshold if vector_index_threshold is not None else settings.data_sniper_vector_index_threshold
        self.use_llm_entity_extraction = use_llm_entity_extraction if use_llm_entity_extraction is not None else settings.data_sniper_use_llm_entity_extraction
        self.use_schema_introspector = use_schema_introspector
        # 新增: schema 检索配置
        self.schema_top_k = settings.data_sniper_schema_top_k
        self.schema_max_hops = settings.data_sniper_schema_max_hops
        self._schema_graph: Optional[SchemaGraph] = None
        self._vector_index: Optional[VectorSchemaIndex] = None
        self._introspector: Optional[SchemaIntrospector] = None
        self._schema_metadata: Optional[SchemaMetadata] = None

        # 配置化业务词汇
        self.keywords_config = get_keywords_config(keywords_config_path)

        # LLM 客户端 (用于实体提取)
        self.llm = get_deepseek_client() if use_llm_entity_extraction else None

    async def _get_introspector(self) -> SchemaIntrospector:
        """获取 SchemaIntrospector 实例"""
        if self._introspector is None:
            db_manager = get_db_manager()
            connector = db_manager.get(self.database or "default")
            self._introspector = SchemaIntrospector(connector)
        return self._introspector

    async def get_schema_metadata(self, force_refresh: bool = False) -> SchemaMetadata:
        """
        获取数据库元信息（带缓存）

        这是 LLM-Native 架构的核心:
        - 动态提取表结构、列信息、数据类型
        - 自动识别时间字段
        - 提取枚举值/样本值
        - 分析外键关系

        Returns:
            完整的 Schema 元信息
        """
        if self._schema_metadata is None or force_refresh:
            introspector = await self._get_introspector()
            self._schema_metadata = await introspector.introspect(
                include_samples=True,
                sample_limit=20,
                include_row_counts=True,
            )
        return self._schema_metadata

    async def get_llm_context(self) -> dict:
        """
        生成 LLM 可用的完整上下文

        包含:
        - 数据库结构 (表、列、类型)
        - 时间字段信息
        - 枚举字段信息
        - 样本值

        这些信息让 LLM 能够:
        1. 理解数据模型
        2. 正确处理时间查询
        3. 匹配用户术语到数据库值
        """
        if not self.use_schema_introspector:
            return {"schema_context": "", "time_context": "", "enum_context": ""}

        metadata = await self.get_schema_metadata()
        introspector = await self._get_introspector()

        return {
            "schema_context": introspector.generate_llm_context(metadata),
            "time_context": introspector.generate_time_fields_context(metadata),
            "enum_context": introspector.generate_enum_fields_context(metadata),
            "metadata": metadata,
        }

    async def get_relevant_schema(
        self,
        query: str,
        top_k: int = 5,
        max_hops: int = 2,
    ) -> dict:
        """
        根据用户查询获取相关的 Schema

        使用 Rerank + 图剪枝的混合策略:
        1. Rerank 找出种子表 (高相关性)
        2. 图剪枝扩展关联表 (外键关系)

        Args:
            query: 用户自然语言查询
            top_k: Rerank 返回的种子表数量
            max_hops: 图剪枝最大跳数

        Returns:
            包含相关表和 DDL 的字典
        """
        # 1. 获取所有表信息
        tables_info = await list_tables(self.database)
        tables = tables_info.get("tables", [])

        if not tables:
            return {"error": "No tables found", "schema": "", "tables": []}

        # 2. 构建表描述文档和详细信息
        table_docs = []
        table_details = {}  # name -> {columns, ddl, ...}

        for t in tables:
            # 获取表的详细信息
            ddl_info = await get_ddl(t["name"], self.database)
            columns = ddl_info.get("columns", [])

            # 保存详细信息
            table_details[t["name"]] = {
                "name": t["name"],
                "columns": columns,
                "ddl": ddl_info.get("ddl", ""),
                "comment": t.get("comment", ""),
            }

            # 构建描述
            col_names = [c["name"] for c in columns]
            doc = f"Table: {t['name']}"
            if t.get("comment"):
                doc += f" ({t['comment']})"
            doc += f"\nColumns: {', '.join(col_names)}"
            table_docs.append(doc)

        # 3. 大 Schema 场景使用向量索引预筛选
        use_vector = (
            self.use_vector_index and
            len(tables) >= self.vector_index_threshold
        )

        if use_vector:
            # 构建/更新向量索引
            if self._vector_index is None:
                self._vector_index = VectorSchemaIndex(cache_dir="data/vector_cache")

            await self._vector_index.build_index(list(table_details.values()))

            # 向量检索预筛选
            vector_results = await self._vector_index.search(query, top_k=top_k * 2)

            # 过滤 table_docs 只保留向量检索结果
            vector_table_names = {r["name"] for r in vector_results}
            filtered_docs = []
            filtered_tables = []
            for i, t in enumerate(tables):
                if t["name"] in vector_table_names:
                    filtered_docs.append(table_docs[i])
                    filtered_tables.append(t)

            if filtered_docs:
                table_docs = filtered_docs
                tables = filtered_tables
                print(f"Vector index pre-filtered to {len(tables)} tables")

        # 4. 使用 Rerank 找出种子表
        seed_tables = []
        try:
            rerank_results = await self.rerank_client.rerank(
                query=query,
                documents=table_docs,
                top_n=min(top_k, len(table_docs)),
            )

            for r in rerank_results:
                table_name = tables[r["index"]]["name"]
                seed_tables.append({
                    "name": table_name,
                    "score": r["score"],
                    "source": "vector+rerank" if use_vector else "rerank",
                })
        except Exception as e:
            # Rerank 失败时使用简单的关键词匹配
            print(f"Rerank failed: {e}, falling back to keyword matching")
            seed_tables = self._keyword_match(query, tables, top_k)
            for t in seed_tables:
                t["source"] = "keyword"

        # 5. 图剪枝扩展关联表
        relevant_tables = seed_tables.copy()

        if self.use_graph_pruning and seed_tables:
            # 构建 Schema 图
            if self._schema_graph is None:
                self._schema_graph = build_schema_graph(list(table_details.values()))

            # 获取种子表名
            seed_names = [t["name"] for t in seed_tables]

            # BFS 扩展关联表
            expanded_tables = self._schema_graph.get_related_tables(
                seed_tables=seed_names,
                max_hops=max_hops,
                include_seeds=False,  # 种子表已经在列表中
            )

            # 添加扩展的表 (分数递减)
            for i, table_name in enumerate(expanded_tables):
                if table_name not in seed_names:
                    # 根据跳数计算分数衰减
                    base_score = seed_tables[-1]["score"] if seed_tables else 0.5
                    decay_score = base_score * (0.7 ** (i + 1))

                    relevant_tables.append({
                        "name": table_name,
                        "score": decay_score,
                        "source": "graph_expansion",
                    })

        # 5. 获取相关表的 DDL
        schema_parts = []
        for t in relevant_tables:
            detail = table_details.get(t["name"])
            if detail and detail.get("ddl"):
                schema_parts.append(detail["ddl"])

        return {
            "query": query,
            "relevant_tables": relevant_tables,
            "schema": "\n\n".join(schema_parts),
            "table_count": len(relevant_tables),
            "seed_count": len(seed_tables),
            "expanded_count": len(relevant_tables) - len(seed_tables),
        }

    def _keyword_match(
        self,
        query: str,
        tables: list[dict],
        top_k: int,
    ) -> list[dict]:
        """
        简单的关键词匹配作为后备方案

        使用配置化的业务词汇，而非硬编码
        """
        query_lower = query.lower()
        scores = []

        # 获取配置化的业务词汇
        business_keywords = self.keywords_config.get_keywords()

        for t in tables:
            score = 0
            table_name = t["name"].lower()

            # 表名匹配
            if table_name in query_lower:
                score += 10
            for word in query_lower.split():
                if word in table_name:
                    score += 5

            # 使用配置化的业务词汇匹配
            for table_key, keywords in business_keywords.items():
                if table_key in table_name:
                    for kw in keywords:
                        if kw in query_lower:
                            score += 3

            scores.append({"name": t["name"], "score": score})

        # 按分数排序
        scores.sort(key=lambda x: x["score"], reverse=True)
        return scores[:top_k]

    async def map_values(
        self,
        query: str,
        entities: list[str],
    ) -> dict:
        """
        将用户查询中的实体映射到数据库中的实际值

        Args:
            query: 用户查询
            entities: 从查询中提取的实体列表

        Returns:
            值映射结果
        """
        mappings = {}

        for entity in entities:
            # 搜索数据库中的匹配值
            search_result = await search_values(
                search_term=entity,
                database=self.database,
                limit=10,
            )

            if search_result.get("results"):
                # 使用 Rerank 找出最佳匹配
                candidates = [r["value"] for r in search_result["results"]]

                try:
                    rerank_results = await self.rerank_client.rerank(
                        query=entity,
                        documents=candidates,
                        top_n=3,
                    )

                    if rerank_results:
                        best_match = rerank_results[0]
                        original_result = search_result["results"][best_match["index"]]

                        mappings[entity] = {
                            "db_value": best_match["document"],
                            "table": original_result["table"],
                            "column": original_result["column"],
                            "score": best_match["score"],
                        }
                except Exception:
                    # Rerank 失败时使用第一个结果
                    if search_result["results"]:
                        r = search_result["results"][0]
                        mappings[entity] = {
                            "db_value": r["value"],
                            "table": r["table"],
                            "column": r["column"],
                            "score": 0.5,
                        }

        return {
            "query": query,
            "entities": entities,
            "mappings": mappings,
        }

    async def analyze(self, query: str) -> dict:
        """
        完整的数据分析流程 (LLM-Native 版)

        Args:
            query: 用户自然语言查询

        Returns:
            包含 Schema、元信息上下文和值映射的完整分析结果

        返回内容:
        - schema: 相关表的 DDL
        - schema_context: LLM 可用的完整数据库上下文
        - time_context: 时间字段专用上下文
        - enum_context: 枚举字段专用上下文
        - relevant_tables: 相关表列表
        - value_mappings: 实体到数据库值的映射
        """
        # 1. 获取相关 Schema (Rerank + 图剪枝)，使用配置的 top_k 和 max_hops
        schema_result = await self.get_relevant_schema(
            query,
            top_k=self.schema_top_k,
            max_hops=self.schema_max_hops
        )

        # 2. 获取 LLM 上下文 (SchemaIntrospector)
        llm_context = await self.get_llm_context()

        # 3. 使用 LLM 驱动的实体提取
        if self.use_llm_entity_extraction:
            entities = await self._extract_entities_llm(query)
        else:
            entities = self._extract_entities_regex(query)

        # 4. 值映射
        mapping_result = await self.map_values(query, entities) if entities else {"mappings": {}}

        return {
            "query": query,
            # 传统 Schema 信息
            "schema": schema_result.get("schema", ""),
            "relevant_tables": schema_result.get("relevant_tables", []),
            # LLM-Native 上下文 (核心增强)
            "schema_context": llm_context.get("schema_context", ""),
            "time_context": llm_context.get("time_context", ""),
            "enum_context": llm_context.get("enum_context", ""),
            # 值映射
            "value_mappings": mapping_result.get("mappings", {}),
            "extracted_entities": entities,
        }

    async def _extract_entities_llm(self, query: str) -> list[str]:
        """
        使用 LLM 提取实体 (推荐)

        符合 README 设计原则: 复杂推理统一由 DeepSeek 处理
        """
        if not self.llm:
            return self._extract_entities_regex(query)

        messages = [
            {"role": "system", "content": ENTITY_EXTRACTION_PROMPT},
            {"role": "user", "content": f"请从以下查询中提取实体:\n\n{query}"},
        ]

        try:
            response = await self.llm.chat(messages)

            # 解析 JSON 响应
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                result = json.loads(json_match.group())
                entities = result.get("entities", [])
                return [e["value"] for e in entities if isinstance(e, dict) and "value" in e]

            return []
        except Exception as e:
            print(f"LLM entity extraction failed: {e}, falling back to regex")
            return self._extract_entities_regex(query)

    def _extract_entities_regex(self, query: str) -> list[str]:
        """
        简单的实体提取（基于引号和特定模式）

        作为 LLM 提取的后备方案
        """
        entities = []

        # 提取英文引号中的内容
        quoted = re.findall(r'["\']([^"\']+)["\']', query)
        entities.extend(quoted)

        # 提取中文引号 (使用 Unicode)
        chinese_quoted = re.findall(r'[\u300c\u300e\u201c\u2018]([^\u300d\u300f\u201d\u2019]+)[\u300d\u300f\u201d\u2019]', query)
        entities.extend(chinese_quoted)

        return list(set(entities))

    def _extract_entities(self, query: str) -> list[str]:
        """
        同步版本的实体提取 (兼容旧接口)

        注意: 这是同步方法，只使用正则提取
        推荐使用 _extract_entities_llm() 异步方法
        """
        return self._extract_entities_regex(query)


# 导出
__all__ = [
    "DataSniper",
    "BusinessKeywordsConfig",
    "get_keywords_config",
    "ENTITY_EXTRACTION_PROMPT",
]
