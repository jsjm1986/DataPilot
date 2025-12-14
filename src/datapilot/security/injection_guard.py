# -*- coding: utf-8 -*-
"""
SQL Injection Guard
Detect and prevent SQL injection attacks
"""

import re
from typing import Optional


class InjectionGuard:
    """SQL Injection Detection and Prevention"""

    # Dangerous SQL patterns
    DANGEROUS_PATTERNS = [
        # Multi-statement injection
        r";\s*(?:drop|delete|truncate|update|insert|alter|create|grant|revoke)",
        # Comment injection
        r"--\s*$",
        r"/\*.*\*/",
        # UNION injection
        r"\bunion\s+(?:all\s+)?select\b",
        # Boolean-based injection
        r"\bor\s+['\"]?\d+['\"]?\s*=\s*['\"]?\d+['\"]?",
        r"\band\s+['\"]?\d+['\"]?\s*=\s*['\"]?\d+['\"]?",
        r"\bor\s+['\"]?[a-z]+['\"]?\s*=\s*['\"]?[a-z]+['\"]?",
        # Time-based injection
        r"\bsleep\s*\(",
        r"\bbenchmark\s*\(",
        r"\bwaitfor\s+delay\b",
        # Stacked queries
        r";\s*(?:exec|execute)\s*\(",
        # Information schema access
        r"\binformation_schema\b",
        r"\bsys\.\w+\b",
        # File operations
        r"\bload_file\s*\(",
        r"\binto\s+(?:out|dump)file\b",
        # Hex encoding bypass
        r"0x[0-9a-f]+",
        # Char encoding bypass
        r"\bchar\s*\(\s*\d+",
    ]

    # Dangerous keywords (case-insensitive)
    DANGEROUS_KEYWORDS = [
        "drop", "truncate", "delete", "update", "insert",
        "alter", "create", "grant", "revoke", "exec", "execute",
        "xp_", "sp_", "shutdown", "kill",
    ]

    def __init__(self):
        self._compiled_patterns = [
            re.compile(p, re.IGNORECASE | re.DOTALL)
            for p in self.DANGEROUS_PATTERNS
        ]

    def check(self, sql: str) -> dict:
        """
        Check SQL for injection attempts

        Args:
            sql: SQL string to check

        Returns:
            dict with 'safe' boolean and 'issues' list
        """
        issues = []
        sql_lower = sql.lower()

        # Check patterns
        for i, pattern in enumerate(self._compiled_patterns):
            if pattern.search(sql):
                issues.append(f"Dangerous pattern detected: {self.DANGEROUS_PATTERNS[i][:30]}...")

        # Check keywords in non-SELECT context
        if not sql_lower.strip().startswith("select"):
            for keyword in self.DANGEROUS_KEYWORDS:
                if re.search(rf"\b{keyword}\b", sql_lower):
                    issues.append(f"Dangerous keyword in non-SELECT: {keyword.upper()}")

        # Check for multiple statements
        statements = [s.strip() for s in sql.split(";") if s.strip()]
        if len(statements) > 1:
            issues.append("Multiple SQL statements detected")

        # Check for suspicious string patterns
        if re.search(r"'\s*\+\s*'", sql) or re.search(r"'\s*\|\|\s*'", sql):
            issues.append("String concatenation detected")

        return {
            "safe": len(issues) == 0,
            "sql": sql,
            "issues": issues,
        }

    def sanitize_input(self, user_input: str) -> str:
        """
        Sanitize user input to prevent injection

        Args:
            user_input: Raw user input

        Returns:
            Sanitized string
        """
        # Remove dangerous characters
        sanitized = user_input.replace("'", "''")  # Escape single quotes
        sanitized = sanitized.replace(";", "")  # Remove semicolons
        sanitized = sanitized.replace("--", "")  # Remove comment markers
        sanitized = re.sub(r"/\*.*?\*/", "", sanitized)  # Remove block comments

        return sanitized


def check_sql_injection(sql: str) -> dict:
    """Convenience function for SQL injection check.

    Returns:
        bool: True if injection risk detected, False if considered safe.
    """
    guard = InjectionGuard()
    result = guard.check(sql)
    return not result["safe"]


__all__ = ["InjectionGuard", "check_sql_injection"]
