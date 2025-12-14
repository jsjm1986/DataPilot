# -*- coding: utf-8 -*-
"""
Viz Expert Agent (CodeAct 版)

可视化专家 - 支持两种模式:
1. ECharts 配置模式: LLM 直接生成 ECharts JSON 配置
2. CodeAct 模式: LLM 生成 Python 代码，在 E2B 沙箱执行

README 要求:
- Viz Expert 生成 Python 代码
- E2B Sandbox 执行
- 禁网，CPU<0.5 vCPU，Memory<512MB，Timeout<10s
- 白名单依赖 (pandas, numpy, pyecharts/plotly)
"""

import json
import re
from typing import Any, Literal, Optional
from dataclasses import dataclass

from ..llm.deepseek import get_deepseek_client
from ..sandbox.executor import run_python_safe, validate_code


@dataclass
class DataCleaningResult:
    """数据清洗结果"""
    cleaned_data: list[dict]
    original_count: int
    cleaned_count: int
    removed_nulls: int
    type_conversions: int
    outliers_removed: int
    warnings: list[str]


@dataclass
class ValidationResult:
    """校验结果"""
    is_valid: bool
    errors: list[str]
    warnings: list[str]


@dataclass
class ChartConfig:
    """图表配置结果"""
    chart_type: str
    echarts_config: dict
    reason: str
    confidence: float


@dataclass
class CodeActResult:
    """CodeAct 执行结果"""
    success: bool
    python_code: str
    stdout: str
    stderr: str
    error: Optional[str]
    chart_data: Optional[dict]  # 从 stdout 解析的图表数据
    executor: str  # e2b, docker, subprocess


# ============================================
# System Prompts
# ============================================

# LLM 直接生成 ECharts 配置的 System Prompt
ECHARTS_GENERATION_PROMPT = """你是一个专业的数据可视化专家，精通 ECharts 图表库。

## 任务
根据用户查询和数据，直接生成完整的 ECharts 配置 JSON。

## 可用图表类型
- bar (柱状图): 分类对比、排名
- line (折线图): 时间序列、趋势
- pie (饼图): 占比分析 (类别数 ≤ 10)
- scatter (散点图): 相关性、分布
- area (面积图): 累计趋势
- heatmap (热力图): 矩阵数据
- radar (雷达图): 多维度对比
- funnel (漏斗图): 转化分析
- gauge (仪表盘): 单一 KPI
- treemap (矩形树图): 层级占比
- table (表格): 无明确可视化需求时使用

## ECharts 配置要求
1. 必须是有效的 ECharts option 对象
2. 包含 title, tooltip, series 等必要字段
3. 数据直接嵌入配置中
4. 使用合适的颜色方案
5. 添加适当的交互效果

## 输出格式
返回 JSON:
```json
{
    "chart_type": "图表类型",
    "reason": "选择理由",
    "confidence": 0.0-1.0,
    "echarts_config": {
        // 完整的 ECharts option 配置
        "title": {"text": "标题", "left": "center"},
        "tooltip": {...},
        "xAxis": {...},
        "yAxis": {...},
        "series": [...]
    }
}
```

## 注意
- 如果数据不适合可视化，chart_type 设为 "table"，echarts_config 设为 {}
- 确保 echarts_config 是有效的 JSON，可以直接传给 ECharts
- 根据数据量选择合适的展示方式（数据多时考虑聚合或分页）
- 使用中文标签和提示
"""

# CodeAct 模式: 生成 Python 代码的 System Prompt
CODEACT_GENERATION_PROMPT = """你是一个专业的数据可视化专家，需要生成 Python 代码来处理和可视化数据。

## 任务
根据用户查询和数据，生成可执行的 Python 代码，处理数据并生成可视化配置。

## 执行环境
- Python 3.11 沙箱环境
- 禁用网络
- 内存限制 512MB
- 超时 10 秒

## 可用库
只能使用以下白名单库:
- pandas (数据处理)
- numpy (数值计算)
- json (JSON 处理)
- datetime (日期时间)
- math, statistics (数学统计)

## 代码要求
1. 代码必须是完整可执行的
2. 数据通过 DATA 变量传入 (list[dict] 格式)
3. 最终结果必须 print 为 JSON 格式
4. 不要使用 matplotlib/plotly 直接绑图 (前端负责渲染)
5. 输出 ECharts 兼容的配置格式

## 输出格式
代码的最后必须 print 一个 JSON 对象:
```python
result = {
    "chart_type": "bar|line|pie|scatter|...",
    "reason": "选择理由",
    "echarts_config": {
        "title": {"text": "标题"},
        "xAxis": {...},
        "yAxis": {...},
        "series": [...]
    }
}
print(json.dumps(result, ensure_ascii=False))
```

## 代码模板
```python
import json
import pandas as pd
from datetime import datetime

# 输入数据 (由沙箱注入)
DATA = __DATA_PLACEHOLDER__

# 数据处理
df = pd.DataFrame(DATA)

# 分析和处理逻辑
# ...

# 生成 ECharts 配置
result = {
    "chart_type": "选择的图表类型",
    "reason": "选择理由",
    "echarts_config": {
        # ECharts 配置
    }
}

# 输出结果
print(json.dumps(result, ensure_ascii=False))
```

## 注意
- 不要导入不在白名单中的库
- 不要使用 os, subprocess, requests 等危险模块
- 不要读写文件
- 代码要简洁高效
"""


class VizExpert:
    """
    可视化专家 Agent (CodeAct 版)

    职责:
    1. 数据清洗 (处理空值、类型转换、异常值)
    2. 模式选择: ECharts 配置模式 / CodeAct 模式
    3. LLM 生成配置或代码
    4. 沙箱执行 (CodeAct 模式)
    5. 结果校验
    """

    def __init__(self, mode: Literal["echarts", "codeact", "auto"] = None):
        """
        初始化 VizExpert

        Args:
            mode: 工作模式
                - "echarts": 直接生成 ECharts 配置
                - "codeact": 生成 Python 代码并在沙箱执行 (默认，需要 E2B)
                - "auto": 自动选择 (简单数据用 echarts，复杂处理用 codeact)
                - None: 从配置读取，默认 codeact
        """
        from ..config.settings import get_settings
        self.settings = get_settings()

        self.llm = get_deepseek_client()

        # 从配置读取默认模式，如果未指定则使用 codeact
        if mode is None:
            mode = getattr(self.settings, 'viz_expert_mode', 'codeact')

        # 如果选择 codeact 但没有 E2B key，自动回退到 echarts
        if mode == "codeact" and not self.settings.e2b_api_key:
            print("Warning: E2B API key not configured, falling back to echarts mode")
            mode = "echarts"

        self.mode = mode

    # ============================================
    # 数据清洗
    # ============================================

    def clean_data(
        self,
        data: list[dict],
        remove_nulls: Optional[bool] = None,
        convert_types: Optional[bool] = None,
        remove_outliers: Optional[bool] = None,
        outlier_threshold: Optional[float] = None,
    ) -> DataCleaningResult:
        """数据清洗"""
        # 使用 settings 作为默认值
        if remove_nulls is None:
            remove_nulls = self.settings.viz_expert_remove_nulls
        if convert_types is None:
            convert_types = self.settings.viz_expert_convert_types
        if remove_outliers is None:
            remove_outliers = self.settings.viz_expert_remove_outliers
        if outlier_threshold is None:
            outlier_threshold = self.settings.viz_expert_outlier_threshold

        if not data:
            return DataCleaningResult(
                cleaned_data=[],
                original_count=0,
                cleaned_count=0,
                removed_nulls=0,
                type_conversions=0,
                outliers_removed=0,
                warnings=["Empty data provided"],
            )

        original_count = len(data)
        cleaned = list(data)
        warnings = []
        removed_nulls_count = 0
        type_conversions_count = 0
        outliers_removed_count = 0

        # 1. 类型转换
        if convert_types:
            cleaned, type_conversions_count = self._convert_types(cleaned)

        # 2. 移除空值行
        if remove_nulls:
            before_count = len(cleaned)
            cleaned = self._remove_null_rows(cleaned)
            removed_nulls_count = before_count - len(cleaned)
            if removed_nulls_count > 0:
                warnings.append(f"Removed {removed_nulls_count} rows with null values")

        # 3. 移除异常值
        if remove_outliers and cleaned:
            before_count = len(cleaned)
            cleaned = self._remove_outliers(cleaned, outlier_threshold)
            outliers_removed_count = before_count - len(cleaned)
            if outliers_removed_count > 0:
                warnings.append(f"Removed {outliers_removed_count} outlier rows")

        return DataCleaningResult(
            cleaned_data=cleaned,
            original_count=original_count,
            cleaned_count=len(cleaned),
            removed_nulls=removed_nulls_count,
            type_conversions=type_conversions_count,
            outliers_removed=outliers_removed_count,
            warnings=warnings,
        )

    def _convert_types(self, data: list[dict]) -> tuple[list[dict], int]:
        """类型转换"""
        if not data:
            return data, 0

        conversions = 0
        columns = list(data[0].keys())

        for row in data:
            for col in columns:
                value = row.get(col)
                if value is None:
                    continue

                if isinstance(value, str):
                    try:
                        if '.' in value:
                            row[col] = float(value)
                            conversions += 1
                        elif value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
                            row[col] = int(value)
                            conversions += 1
                    except (ValueError, AttributeError):
                        pass

        return data, conversions

    def _remove_null_rows(self, data: list[dict]) -> list[dict]:
        """移除包含空值的行"""
        return [
            row for row in data
            if all(v is not None and v != "" and v != "null" for v in row.values())
        ]

    def _remove_outliers(self, data: list[dict], threshold: float) -> list[dict]:
        """移除异常值 (基于 Z-score)"""
        if not data:
            return data

        numeric_cols = []
        for col in data[0].keys():
            if isinstance(data[0].get(col), (int, float)):
                numeric_cols.append(col)

        if not numeric_cols:
            return data

        stats = {}
        for col in numeric_cols:
            values = [row.get(col) for row in data if isinstance(row.get(col), (int, float))]
            if values:
                mean = sum(values) / len(values)
                variance = sum((x - mean) ** 2 for x in values) / len(values)
                std = variance ** 0.5
                stats[col] = (mean, std)

        cleaned = []
        for row in data:
            is_outlier = False
            for col, (mean, std) in stats.items():
                if std > 0:
                    value = row.get(col)
                    if isinstance(value, (int, float)):
                        z_score = abs((value - mean) / std)
                        if z_score > threshold:
                            is_outlier = True
                            break
            if not is_outlier:
                cleaned.append(row)

        return cleaned

    # ============================================
    # 模式选择
    # ============================================

    def _should_use_codeact(self, data: list[dict], query: str) -> bool:
        """判断是否应该使用 CodeAct 模式"""
        if self.mode == "echarts":
            return False
        if self.mode == "codeact":
            return True

        # auto 模式: 根据数据复杂度和查询内容决定
        # 使用 CodeAct 的情况:
        # 1. 数据量大 (>100 行)
        # 2. 查询包含复杂处理关键词
        # 3. 需要数据聚合/透视

        if len(data) > 100:
            return True

        complex_keywords = [
            "聚合", "透视", "分组", "环比", "同比", "增长率",
            "移动平均", "累计", "占比", "排名", "TopN",
            "异常", "离群", "相关性", "分布",
        ]
        for keyword in complex_keywords:
            if keyword in query:
                return True

        return False

    # ============================================
    # ECharts 配置模式
    # ============================================

    async def generate_chart_config(
        self,
        data: list[dict],
        query: str,
    ) -> ChartConfig:
        """
        使用 LLM 直接生成 ECharts 配置

        Args:
            data: 查询结果数据
            query: 原始查询

        Returns:
            图表配置结果
        """
        if not data:
            return ChartConfig(
                chart_type="table",
                echarts_config={},
                reason="No data available",
                confidence=1.0,
            )

        # 构建数据摘要
        data_summary = self._build_data_summary(data)

        # 构建 LLM 请求
        messages = [
            {"role": "system", "content": ECHARTS_GENERATION_PROMPT},
            {"role": "user", "content": self._build_generation_prompt(query, data_summary, data)},
        ]

        try:
            response = await self.llm.chat(messages)
            return self._parse_chart_response(response)
        except Exception as e:
            # LLM 调用失败，返回表格
            return ChartConfig(
                chart_type="table",
                echarts_config={},
                reason=f"LLM generation failed: {str(e)}",
                confidence=0.0,
            )

    def _build_data_summary(self, data: list[dict]) -> dict:
        """构建数据摘要"""
        if not data:
            return {"columns": [], "row_count": 0, "sample": []}

        columns = list(data[0].keys())
        row_count = len(data)

        column_info = []
        for col in columns:
            sample_values = [row.get(col) for row in data[:5] if row.get(col) is not None]
            if sample_values:
                first_value = sample_values[0]
                if isinstance(first_value, (int, float)):
                    col_type = "numeric"
                elif isinstance(first_value, bool):
                    col_type = "boolean"
                else:
                    str_val = str(first_value).lower()
                    if any(pattern in str_val for pattern in ["-", "/", "date", "time"]):
                        col_type = "date"
                    else:
                        col_type = "text"
            else:
                col_type = "unknown"

            column_info.append({
                "name": col,
                "type": col_type,
                "sample_values": [str(v)[:50] for v in sample_values[:3]],
            })

        return {
            "columns": column_info,
            "row_count": row_count,
        }

    def _build_generation_prompt(self, query: str, data_summary: dict, data: list[dict]) -> str:
        """构建生成提示"""
        columns_desc = "\n".join([
            f"  - {col['name']} ({col['type']}): 样本值 {col['sample_values']}"
            for col in data_summary["columns"]
        ])

        # 限制数据量，避免 token 过多
        sample_data = data[:50] if len(data) > 50 else data
        data_json = json.dumps(sample_data, ensure_ascii=False, indent=2)

        return f"""用户查询: {query}

数据摘要:
- 总行数: {data_summary['row_count']}
- 列信息:
{columns_desc}

完整数据 (前 {len(sample_data)} 行):
{data_json}

请根据查询意图和数据特征，生成最合适的 ECharts 配置。"""

    def _parse_chart_response(self, response: str) -> ChartConfig:
        """解析 LLM 响应"""
        # 尝试提取 JSON
        json_match = re.search(r'\{[\s\S]*\}', response)
        if not json_match:
            return ChartConfig(
                chart_type="table",
                echarts_config={},
                reason="Failed to parse LLM response",
                confidence=0.0,
            )

        try:
            result = json.loads(json_match.group())

            chart_type = result.get("chart_type", "table").lower()
            echarts_config = result.get("echarts_config", {})
            reason = result.get("reason", "")
            confidence = float(result.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))

            return ChartConfig(
                chart_type=chart_type,
                echarts_config=echarts_config,
                reason=reason,
                confidence=confidence,
            )

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            return ChartConfig(
                chart_type="table",
                echarts_config={},
                reason=f"JSON parse error: {str(e)}",
                confidence=0.0,
            )

    # ============================================
    # CodeAct 模式 (E2B 沙箱执行)
    # ============================================

    async def generate_and_execute_code(
        self,
        data: list[dict],
        query: str,
    ) -> CodeActResult:
        """
        CodeAct 模式: 生成 Python 代码并在沙箱执行

        Args:
            data: 查询结果数据
            query: 原始查询

        Returns:
            CodeAct 执行结果
        """
        if not data:
            return CodeActResult(
                success=False,
                python_code="",
                stdout="",
                stderr="",
                error="No data available",
                chart_data=None,
                executor="none",
            )

        # 1. 生成 Python 代码
        code = await self._generate_python_code(data, query)

        # 2. 验证代码安全性
        is_safe, reason = validate_code(code, strict_mode=False)
        if not is_safe:
            return CodeActResult(
                success=False,
                python_code=code,
                stdout="",
                stderr="",
                error=f"Code validation failed: {reason}",
                chart_data=None,
                executor="validator",
            )

        # 3. 在沙箱中执行
        exec_result = await run_python_safe(code)

        # 4. 解析结果
        chart_data = None
        if exec_result.get("success") and exec_result.get("stdout"):
            try:
                # 尝试从 stdout 解析 JSON
                stdout = exec_result["stdout"].strip()
                # 找到最后一个 JSON 对象
                json_match = re.search(r'\{[\s\S]*\}(?![\s\S]*\{)', stdout)
                if json_match:
                    chart_data = json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        return CodeActResult(
            success=exec_result.get("success", False),
            python_code=code,
            stdout=exec_result.get("stdout", ""),
            stderr=exec_result.get("stderr", ""),
            error=exec_result.get("error"),
            chart_data=chart_data,
            executor=exec_result.get("executor", "unknown"),
        )

    async def _generate_python_code(self, data: list[dict], query: str) -> str:
        """生成 Python 代码"""
        # 构建数据摘要
        data_summary = self._build_data_summary(data)
        columns_desc = "\n".join([
            f"  - {col['name']} ({col['type']}): 样本值 {col['sample_values']}"
            for col in data_summary["columns"]
        ])

        # 限制数据量传给 LLM
        sample_data = data[:20] if len(data) > 20 else data

        user_prompt = f"""用户查询: {query}

数据摘要:
- 总行数: {len(data)}
- 列信息:
{columns_desc}

样本数据 (前 {len(sample_data)} 行):
{json.dumps(sample_data, ensure_ascii=False, indent=2)}

请生成完整的 Python 代码来处理数据并生成 ECharts 配置。
代码中的 DATA 变量将被替换为完整数据 ({len(data)} 行)。
"""

        messages = [
            {"role": "system", "content": CODEACT_GENERATION_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = await self.llm.chat(messages)

            # 提取代码块
            code_match = re.search(r'```python\s*([\s\S]*?)\s*```', response)
            if code_match:
                code = code_match.group(1)
            else:
                # 尝试提取所有代码
                code = response.strip()

            # 替换数据占位符
            data_json = json.dumps(data, ensure_ascii=False)
            if "__DATA_PLACEHOLDER__" in code:
                code = code.replace("__DATA_PLACEHOLDER__", data_json)
            elif "DATA = " in code:
                # 替换 DATA 赋值行
                code = re.sub(
                    r'DATA\s*=\s*\[[\s\S]*?\]',
                    f'DATA = {data_json}',
                    code,
                    count=1
                )
            else:
                # 在开头插入数据
                code = f"DATA = {data_json}\n\n{code}"

            return code

        except Exception as e:
            # 生成失败，返回基础代码
            return f"""
import json
import pandas as pd

DATA = {json.dumps(data, ensure_ascii=False)}

df = pd.DataFrame(DATA)

result = {{
    "chart_type": "table",
    "reason": "Code generation failed: {str(e)}",
    "echarts_config": {{}}
}}

print(json.dumps(result, ensure_ascii=False))
"""

    # ============================================
    # 配置校验
    # ============================================

    def validate_chart_config(self, config: dict, chart_type: str) -> ValidationResult:
        """校验 ECharts 配置有效性"""
        errors = []
        warnings = []

        if not config:
            if chart_type == "table":
                return ValidationResult(is_valid=True, errors=[], warnings=[])
            return ValidationResult(is_valid=False, errors=["Empty config"], warnings=[])

        # 检查必要字段
        if "series" not in config:
            errors.append("Missing 'series' field")

        # 检查轴配置
        if chart_type in ["bar", "line", "scatter", "area"]:
            if "xAxis" not in config:
                warnings.append(f"Missing 'xAxis' for {chart_type} chart")
            if "yAxis" not in config:
                warnings.append(f"Missing 'yAxis' for {chart_type} chart")

        # 检查 series 数据
        if "series" in config:
            series = config["series"]
            if not series:
                errors.append("Empty series array")
            else:
                for i, s in enumerate(series):
                    if "type" not in s:
                        errors.append(f"Series[{i}] missing 'type'")
                    if "data" not in s:
                        warnings.append(f"Series[{i}] missing 'data'")
                    elif not s.get("data"):
                        warnings.append(f"Series[{i}] has empty data")

        # 检查数据量
        if "series" in config and config["series"]:
            first_series = config["series"][0]
            data_len = len(first_series.get("data", []))
            if data_len > 1000:
                warnings.append(f"Large dataset ({data_len} points), consider aggregation")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    # ============================================
    # 主分析方法
    # ============================================

    async def analyze(
        self,
        data: list[dict],
        query: str,
        clean_data: bool = True,
        validate_output: bool = True,
        force_mode: Optional[Literal["echarts", "codeact"]] = None,
    ) -> dict:
        """
        完整的可视化分析

        Args:
            data: 查询结果
            query: 原始查询
            clean_data: 是否进行数据清洗
            validate_output: 是否校验输出
            force_mode: 强制指定模式

        Returns:
            可视化配置
        """
        result = {
            "chart_type": "table",
            "echarts_config": {},
            "reason": "",
            "confidence": 0.0,
            "cleaning_result": None,
            "validation": None,
            "mode": "echarts",
            "python_code": None,
            "sandbox_result": None,
        }

        # 1. 数据清洗
        if clean_data:
            cleaning_result = self.clean_data(data)
            result["cleaning_result"] = {
                "original_count": cleaning_result.original_count,
                "cleaned_count": cleaning_result.cleaned_count,
                "removed_nulls": cleaning_result.removed_nulls,
                "type_conversions": cleaning_result.type_conversions,
                "warnings": cleaning_result.warnings,
            }
            data = cleaning_result.cleaned_data

        # 2. 模式选择
        use_codeact = force_mode == "codeact" if force_mode else self._should_use_codeact(data, query)
        result["mode"] = "codeact" if use_codeact else "echarts"

        # 3. 生成配置或执行代码
        if use_codeact:
            # CodeAct 模式
            codeact_result = await self.generate_and_execute_code(data, query)
            result["python_code"] = codeact_result.python_code
            result["sandbox_result"] = {
                "success": codeact_result.success,
                "stdout": codeact_result.stdout,
                "stderr": codeact_result.stderr,
                "error": codeact_result.error,
                "executor": codeact_result.executor,
            }

            if codeact_result.success and codeact_result.chart_data:
                result["chart_type"] = codeact_result.chart_data.get("chart_type", "table")
                result["echarts_config"] = codeact_result.chart_data.get("echarts_config", {})
                result["reason"] = codeact_result.chart_data.get("reason", "Generated by CodeAct")
                result["confidence"] = 0.8
            else:
                # CodeAct 失败，回退到 ECharts 模式
                result["reason"] = f"CodeAct failed: {codeact_result.error}, falling back to ECharts"
                chart_config = await self.generate_chart_config(data, query)
                result["chart_type"] = chart_config.chart_type
                result["echarts_config"] = chart_config.echarts_config
                result["reason"] = chart_config.reason
                result["confidence"] = chart_config.confidence
        else:
            # ECharts 配置模式
            chart_config = await self.generate_chart_config(data, query)
            result["chart_type"] = chart_config.chart_type
            result["echarts_config"] = chart_config.echarts_config
            result["reason"] = chart_config.reason
            result["confidence"] = chart_config.confidence

        # 4. 校验配置
        if validate_output:
            validation = self.validate_chart_config(
                result["echarts_config"],
                result["chart_type"]
            )
            result["validation"] = {
                "is_valid": validation.is_valid,
                "errors": validation.errors,
                "warnings": validation.warnings,
            }

        return result

    # ============================================
    # 兼容旧接口
    # ============================================

    def recommend_chart(self, data: list[dict], query: str) -> dict:
        """同步版本 (兼容旧接口)"""
        import asyncio

        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, self.generate_chart_config(data, query))
                config = future.result()
        except RuntimeError:
            config = asyncio.run(self.generate_chart_config(data, query))

        return {
            "chart_type": config.chart_type,
            "reason": config.reason,
            "confidence": config.confidence,
            "echarts_config": config.echarts_config,
        }


# LangGraph 节点函数
async def viz_expert_node(state: dict) -> dict:
    """LangGraph 节点：Viz Expert"""
    expert = VizExpert(mode="auto")

    data = state.get("execution_result", {}).get("data", [])
    query = state.get("query", "")

    result = await expert.analyze(data, query)

    return {
        "visualization": result,
        "chart_config": {
            "chart_type": result.get("chart_type"),
            "title": query[:50],
            "config": result.get("echarts_config"),
        },
        "python_code": result.get("python_code"),
        "sandbox_result": result.get("sandbox_result"),
        "current_agent": "viz_expert",
        "next_agent": None,
    }


__all__ = [
    "VizExpert",
    "viz_expert_node",
    "DataCleaningResult",
    "ValidationResult",
    "ChartConfig",
    "CodeActResult",
    "ECHARTS_GENERATION_PROMPT",
    "CODEACT_GENERATION_PROMPT",
]
