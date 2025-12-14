# -*- coding: utf-8 -*-
"""
SQL 安全校验器
检测 SQL 注入和危险操作
"""

import re
from typing import Optional
from .parser import SQLParser


class SQLValidator:
    """
    SQL 安全校验器

    功能:
    1. SQL 注入检测
    2. 危险操作检测
    3. 只读查询验证
    4. 语法检查
    """

    # 危险关键字 (写操作)
    DANGEROUS_KEYWORDS = [
        "drop", "truncate", "delete", "update", "insert",
        "alter", "create", "grant", "revoke", "exec", "execute",
        "merge", "replace", "call", "load", "import",
    ]

    # SQL 注入模式
    INJECTION_PATTERNS = [
        r";\s*--",                    # 注释注入
        r";\s*(drop|delete|update)",  # 多语句注入
        r"union\s+(all\s+)?select",   # UNION 注入
        r"or\s+1\s*=\s*1",            # 永真条件
        r"or\s+'[^']*'\s*=\s*'[^']*'",# 字符串永真
        r"'\s*or\s*'",                # 字符串注入
        r"--\s*$",                    # 行尾注释
        r"/\*[\s\S]*?\*/",            # 块注释
        r";\s*\w+\s*\(",              # 函数调用注入
        r"benchmark\s*\(",            # 时间盲注
        r"sleep\s*\(",                # 时间盲注
        r"waitfor\s+delay",           # MSSQL 时间盲注
        r"pg_sleep\s*\(",             # PostgreSQL 时间盲注
        r"load_file\s*\(",            # 文件读取
        r"into\s+outfile",            # 文件写入
        r"into\s+dumpfile",           # 文件写入
    ]

    # 危险函数
    DANGEROUS_FUNCTIONS = [
        "load_file", "into_outfile", "into_dumpfile",
        "benchmark", "sleep", "pg_sleep", "waitfor",
        "exec", "execute", "xp_cmdshell", "sp_executesql",
    ]

    def __init__(self, dialect: str = "sqlite"):
        self.dialect = dialect
        self.parser = SQLParser(dialect)

    def validate(self, sql: str) -> dict:
        """
        完整验证 SQL

        Returns:
            {
                "valid": bool,
                "is_readonly": bool,
                "issues": list[str],
                "warnings": list[str],
            }
        """
        issues = []
        warnings = []

        sql_clean = sql.strip()
        if not sql_clean:
            return {
                "valid": False,
                "is_readonly": False,
                "issues": ["SQL is empty"],
                "warnings": [],
            }

        sql_lower = sql_clean.lower()

        # 1. 语法检查
        if not self.parser.is_valid(sql):
            issues.append("SQL syntax error")

        # 2. 只读检查
        is_readonly = self._check_readonly(sql_lower)
        if not is_readonly:
            issues.append("Only SELECT queries are allowed")

        # 3. 危险关键字检查
        dangerous = self._check_dangerous_keywords(sql_lower)
        if dangerous:
            issues.extend(dangerous)

        # 4. SQL 注入检查
        injection = self._check_injection(sql_lower)
        if injection:
            issues.extend(injection)

        # 5. 危险函数检查
        dangerous_funcs = self._check_dangerous_functions(sql_lower)
        if dangerous_funcs:
            issues.extend(dangerous_funcs)

        # 6. 警告检查
        warnings = self._check_warnings(sql_lower)

        return {
            "valid": len(issues) == 0,
            "is_readonly": is_readonly,
            "issues": issues,
            "warnings": warnings,
        }

    def _check_readonly(self, sql_lower: str) -> bool:
        """检查是否为只读查询"""
        # 移除注释
        sql_clean = re.sub(r'--.*$', '', sql_lower, flags=re.MULTILINE)
        sql_clean = re.sub(r'/\*.*?\*/', '', sql_clean, flags=re.DOTALL)
        sql_clean = sql_clean.strip()

        # 检查是否以 SELECT 开头
        return sql_clean.startswith("select") or sql_clean.startswith("with")

    def _check_dangerous_keywords(self, sql_lower: str) -> list[str]:
        """检查危险关键字"""
        issues = []
        for keyword in self.DANGEROUS_KEYWORDS:
            pattern = rf"\b{keyword}\b"
            if re.search(pattern, sql_lower):
                issues.append(f"Dangerous keyword detected: {keyword.upper()}")
        return issues

    def _check_injection(self, sql_lower: str) -> list[str]:
        """检查 SQL 注入"""
        issues = []
        for pattern in self.INJECTION_PATTERNS:
            if re.search(pattern, sql_lower, re.IGNORECASE):
                issues.append("Potential SQL injection detected")
                break
        return issues

    def _check_dangerous_functions(self, sql_lower: str) -> list[str]:
        """检查危险函数"""
        issues = []
        for func in self.DANGEROUS_FUNCTIONS:
            pattern = rf"\b{func}\s*\("
            if re.search(pattern, sql_lower):
                issues.append(f"Dangerous function detected: {func}")
        return issues

    def _check_warnings(self, sql_lower: str) -> list[str]:
        """检查警告"""
        warnings = []

        # SELECT *
        if re.search(r"select\s+\*", sql_lower):
            warnings.append("Using SELECT * is not recommended")

        # 无 LIMIT
        if "limit" not in sql_lower and "top" not in sql_lower:
            warnings.append("No LIMIT clause - may return large result set")

        # 无 WHERE
        if "where" not in sql_lower:
            warnings.append("No WHERE clause - may scan entire table")

        # LIKE 以 % 开头
        if re.search(r"like\s+['\"]%", sql_lower):
            warnings.append("LIKE pattern starts with % - cannot use index")

        return warnings

    def is_safe(self, sql: str) -> bool:
        """快速检查 SQL 是否安全"""
        result = self.validate(sql)
        return result["valid"]

    def sanitize(self, sql: str) -> str:
        """
        清理 SQL (移除危险部分)

        注意: 这只是基本清理，不能完全防止注入
        """
        # 移除注释
        sql = re.sub(r'--.*$', '', sql, flags=re.MULTILINE)
        sql = re.sub(r'/\*.*?\*/', '', sql, flags=re.DOTALL)

        # 移除多余分号
        sql = sql.rstrip(';').strip()

        return sql


# 便捷函数
def validate_sql(sql: str, dialect: str = "sqlite") -> dict:
    """验证 SQL"""
    validator = SQLValidator(dialect)
    return validator.validate(sql)


def is_safe_sql(sql: str, dialect: str = "sqlite") -> bool:
    """检查 SQL 是否安全"""
    validator = SQLValidator(dialect)
    return validator.is_safe(sql)


__all__ = [
    "SQLValidator",
    "validate_sql",
    "is_safe_sql",
]
