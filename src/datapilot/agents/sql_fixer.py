# -*- coding: utf-8 -*-
"""
SQL Fixer - SQL 修复建议生成器

当 SQL 校验或执行失败时，提供具体的修复建议:
1. 分析错误类型
2. 基于错误模式匹配常见修复
3. 使用 LLM 生成智能建议
"""

import re
from dataclasses import dataclass
from typing import Optional
from difflib import get_close_matches

from ..llm.deepseek import get_deepseek_client


@dataclass
class FixSuggestion:
    """修复建议"""
    type: str  # 建议类型: column_fix, table_fix, syntax_fix, type_fix, group_by_fix
    description: str  # 描述
    original: str  # 原始内容
    suggested: str  # 建议内容
    confidence: float  # 置信度 0-1
    auto_fixable: bool  # 是否可自动修复


class SQLFixer:
    """
    SQL 修复建议生成器

    支持的错误类型:
    - 列不存在 -> 模糊匹配建议正确列名
    - 表不存在 -> 搜索相似表名
    - 语法错误 -> LLM 重写
    - 类型不匹配 -> 添加类型转换
    - 缺少 GROUP BY -> 自动补全
    """

    # 错误模式匹配 (所有模式使用小写，因为 analyze_error 会先将错误信息转为小写)
    ERROR_PATTERNS = {
        "column_not_found": [
            r"no such column[:\s]+(\w+)",
            r"unknown column '(\w+)'",
            r"column \"(\w+)\" does not exist",
            r"column (\w+) not found",
        ],
        "table_not_found": [
            r"no such table[:\s]+(\w+)",
            r"table '[\w.]*\.?(\w+)' doesn't exist",
            r"relation \"(\w+)\" does not exist",
            r"table (\w+) not found",
        ],
        "syntax_error": [
            r"syntax error",
            r"near \"(\w+)\"",
            r"unexpected token",
            r"parse error",
        ],
        "type_mismatch": [
            r"type mismatch",
            r"cannot compare",
            r"invalid input syntax for type",
            r"incompatible types",
        ],
        "group_by_missing": [
            r"not in group by",
            r"must appear in the group by clause",
            r"is not in aggregate function",
            r"aggregate function.*without group by",
        ],
        "ambiguous_column": [
            r"ambiguous column name",
            r"column reference \"(\w+)\" is ambiguous",
            r"column '(\w+)' in.*is ambiguous",
        ],
    }

    def __init__(self):
        self.llm_client = None  # 延迟初始化

    def _get_llm(self):
        """获取 LLM 客户端"""
        if self.llm_client is None:
            self.llm_client = get_llm_client()
        return self.llm_client

    def analyze_error(self, error: str) -> tuple[str, Optional[str]]:
        """
        分析错误类型

        Args:
            error: 错误信息

        Returns:
            (错误类型, 提取的实体)
        """
        error_lower = error.lower()

        for error_type, patterns in self.ERROR_PATTERNS.items():
            for pattern in patterns:
                match = re.search(pattern, error_lower)
                if match:
                    entity = match.group(1) if match.groups() else None
                    return error_type, entity

        return "unknown", None

    def generate_suggestions(
        self,
        sql: str,
        error: str,
        schema: str = "",
        available_columns: list[str] = None,
        available_tables: list[str] = None,
    ) -> list[FixSuggestion]:
        """
        生成修复建议

        Args:
            sql: 原始 SQL
            error: 错误信息
            schema: Schema 信息
            available_columns: 可用列名列表
            available_tables: 可用表名列表

        Returns:
            修复建议列表
        """
        suggestions = []
        error_type, entity = self.analyze_error(error)

        if error_type == "column_not_found" and entity:
            suggestions.extend(
                self._suggest_column_fix(entity, available_columns or [])
            )

        elif error_type == "table_not_found" and entity:
            suggestions.extend(
                self._suggest_table_fix(entity, available_tables or [])
            )

        elif error_type == "group_by_missing":
            suggestions.extend(
                self._suggest_group_by_fix(sql)
            )

        elif error_type == "ambiguous_column" and entity:
            suggestions.extend(
                self._suggest_ambiguous_fix(entity, sql)
            )

        elif error_type == "type_mismatch":
            suggestions.extend(
                self._suggest_type_fix(sql, error)
            )

        elif error_type == "syntax_error":
            suggestions.extend(
                self._suggest_syntax_fix(sql, error)
            )

        # 如果没有找到具体建议，添加通用建议
        if not suggestions:
            suggestions.append(FixSuggestion(
                type="general",
                description="检查 SQL 语法和表/列名是否正确",
                original=sql,
                suggested=sql,
                confidence=0.3,
                auto_fixable=False,
            ))

        return suggestions

    def _suggest_column_fix(
        self,
        wrong_column: str,
        available_columns: list[str],
    ) -> list[FixSuggestion]:
        """建议列名修复"""
        suggestions = []

        if not available_columns:
            return suggestions

        # 模糊匹配
        matches = get_close_matches(
            wrong_column.lower(),
            [c.lower() for c in available_columns],
            n=3,
            cutoff=0.6,
        )

        for match in matches:
            # 找到原始大小写的列名
            original_case = next(
                (c for c in available_columns if c.lower() == match),
                match
            )

            suggestions.append(FixSuggestion(
                type="column_fix",
                description=f"将 '{wrong_column}' 替换为 '{original_case}'",
                original=wrong_column,
                suggested=original_case,
                confidence=0.8 if match == wrong_column.lower() else 0.6,
                auto_fixable=True,
            ))

        return suggestions

    def _suggest_table_fix(
        self,
        wrong_table: str,
        available_tables: list[str],
    ) -> list[FixSuggestion]:
        """建议表名修复"""
        suggestions = []

        if not available_tables:
            return suggestions

        # 模糊匹配
        matches = get_close_matches(
            wrong_table.lower(),
            [t.lower() for t in available_tables],
            n=3,
            cutoff=0.6,
        )

        for match in matches:
            original_case = next(
                (t for t in available_tables if t.lower() == match),
                match
            )

            suggestions.append(FixSuggestion(
                type="table_fix",
                description=f"将 '{wrong_table}' 替换为 '{original_case}'",
                original=wrong_table,
                suggested=original_case,
                confidence=0.8 if match == wrong_table.lower() else 0.6,
                auto_fixable=True,
            ))

        return suggestions

    def _suggest_group_by_fix(self, sql: str) -> list[FixSuggestion]:
        """建议 GROUP BY 修复"""
        suggestions = []

        # 提取 SELECT 中的非聚合列
        select_match = re.search(
            r'select\s+(.+?)\s+from',
            sql,
            re.IGNORECASE | re.DOTALL
        )

        if not select_match:
            return suggestions

        select_clause = select_match.group(1)

        # 简单解析列 (不处理复杂表达式)
        columns = []
        for part in select_clause.split(','):
            part = part.strip()
            # 跳过聚合函数
            if re.search(r'\b(count|sum|avg|max|min)\s*\(', part, re.IGNORECASE):
                continue
            # 提取列名或别名
            alias_match = re.search(r'(\w+)\s*$', part)
            if alias_match:
                columns.append(alias_match.group(1))

        if columns:
            group_by_clause = ", ".join(columns)

            # 检查是否已有 GROUP BY
            if re.search(r'\bgroup\s+by\b', sql, re.IGNORECASE):
                # 已有 GROUP BY，建议添加列
                suggestions.append(FixSuggestion(
                    type="group_by_fix",
                    description=f"在 GROUP BY 中添加: {group_by_clause}",
                    original="",
                    suggested=group_by_clause,
                    confidence=0.7,
                    auto_fixable=False,
                ))
            else:
                # 没有 GROUP BY，建议添加
                suggestions.append(FixSuggestion(
                    type="group_by_fix",
                    description=f"添加 GROUP BY {group_by_clause}",
                    original=sql,
                    suggested=f"{sql.rstrip(';')} GROUP BY {group_by_clause}",
                    confidence=0.7,
                    auto_fixable=True,
                ))

        return suggestions

    def _suggest_ambiguous_fix(self, column: str, sql: str) -> list[FixSuggestion]:
        """建议歧义列修复"""
        suggestions = []

        # 提取 FROM 子句中的表名
        from_match = re.search(
            r'from\s+(\w+)(?:\s+(?:as\s+)?(\w+))?',
            sql,
            re.IGNORECASE
        )

        if from_match:
            table_name = from_match.group(2) or from_match.group(1)
            suggestions.append(FixSuggestion(
                type="ambiguous_fix",
                description=f"使用表别名限定列: {table_name}.{column}",
                original=column,
                suggested=f"{table_name}.{column}",
                confidence=0.6,
                auto_fixable=False,
            ))

        return suggestions

    def _suggest_type_fix(self, sql: str, error: str) -> list[FixSuggestion]:
        """建议类型转换修复"""
        suggestions = []

        # 常见类型转换建议
        suggestions.append(FixSuggestion(
            type="type_fix",
            description="尝试使用 CAST 进行类型转换",
            original="",
            suggested="CAST(column AS target_type)",
            confidence=0.5,
            auto_fixable=False,
        ))

        return suggestions

    def _suggest_syntax_fix(self, sql: str, error: str) -> list[FixSuggestion]:
        """建议语法修复"""
        suggestions = []

        # 常见语法问题检查
        sql_lower = sql.lower()

        # 检查括号匹配
        if sql.count('(') != sql.count(')'):
            suggestions.append(FixSuggestion(
                type="syntax_fix",
                description="括号不匹配，请检查括号配对",
                original=sql,
                suggested=sql,
                confidence=0.8,
                auto_fixable=False,
            ))

        # 检查引号匹配
        single_quotes = sql.count("'") - sql.count("\\'")
        if single_quotes % 2 != 0:
            suggestions.append(FixSuggestion(
                type="syntax_fix",
                description="单引号不匹配，请检查字符串引号",
                original=sql,
                suggested=sql,
                confidence=0.8,
                auto_fixable=False,
            ))

        # 检查常见拼写错误
        typos = {
            "selct": "SELECT",
            "slect": "SELECT",
            "form": "FROM",
            "wher": "WHERE",
            "gruop": "GROUP",
            "oder": "ORDER",
            "limt": "LIMIT",
        }

        for typo, correct in typos.items():
            if typo in sql_lower:
                suggestions.append(FixSuggestion(
                    type="syntax_fix",
                    description=f"拼写错误: '{typo}' 应为 '{correct}'",
                    original=typo,
                    suggested=correct,
                    confidence=0.9,
                    auto_fixable=True,
                ))

        return suggestions

    def auto_fix(
        self,
        sql: str,
        error: str,
        schema: str = "",
        available_columns: list[str] = None,
        available_tables: list[str] = None,
    ) -> Optional[str]:
        """
        尝试自动修复 SQL

        Args:
            sql: 原始 SQL
            error: 错误信息
            schema: Schema 信息
            available_columns: 可用列名列表
            available_tables: 可用表名列表

        Returns:
            修复后的 SQL，如果无法修复则返回 None
        """
        suggestions = self.generate_suggestions(
            sql, error, schema, available_columns, available_tables
        )

        # 找到可自动修复且置信度最高的建议
        auto_fixable = [s for s in suggestions if s.auto_fixable and s.confidence >= 0.7]

        if not auto_fixable:
            return None

        # 按置信度排序
        auto_fixable.sort(key=lambda x: x.confidence, reverse=True)

        # 应用修复
        fixed_sql = sql
        for suggestion in auto_fixable:
            if suggestion.type == "group_by_fix":
                # GROUP BY 修复直接使用 suggested
                fixed_sql = suggestion.suggested
            elif suggestion.original and suggestion.suggested:
                # 替换修复
                fixed_sql = re.sub(
                    rf'\b{re.escape(suggestion.original)}\b',
                    suggestion.suggested,
                    fixed_sql,
                    flags=re.IGNORECASE,
                )

        return fixed_sql if fixed_sql != sql else None

    async def generate_llm_fix(
        self,
        sql: str,
        error: str,
        schema: str,
    ) -> Optional[str]:
        """
        使用 LLM 生成修复建议

        Args:
            sql: 原始 SQL
            error: 错误信息
            schema: Schema 信息

        Returns:
            修复后的 SQL
        """
        llm = self._get_llm()

        prompt = f"""你是一个 SQL 专家。请修复以下 SQL 语句中的错误。

原始 SQL:
```sql
{sql}
```

错误信息:
{error}

数据库 Schema:
```sql
{schema[:2000]}
```

请直接返回修复后的 SQL，不要包含任何解释。只返回 SQL 语句本身。
"""

        try:
            response = await llm.chat(prompt)
            fixed_sql = response.strip()

            # 清理可能的 markdown 代码块
            if fixed_sql.startswith("```"):
                lines = fixed_sql.split("\n")
                fixed_sql = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

            return fixed_sql.strip()
        except Exception as e:
            print(f"LLM fix failed: {e}")
            return None


# 便捷函数
def create_sql_fixer() -> SQLFixer:
    """创建 SQL 修复器实例"""
    return SQLFixer()


__all__ = ["SQLFixer", "FixSuggestion", "create_sql_fixer"]
