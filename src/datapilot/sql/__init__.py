# -*- coding: utf-8 -*-
"""SQL Processing Module"""

from .parser import (
    SQLParser,
    parse_sql,
    analyze_sql,
    format_sql,
)

from .validator import (
    SQLValidator,
    validate_sql,
    is_safe_sql,
)

from .dialect import (
    SQLDialectConverter,
    DialectDifferences,
    convert_sql,
    get_dialect_converter,
)

from .cost_analyzer import (
    CostAnalyzer,
    analyze_sql_cost,
    get_explain_plan,
)

__all__ = [
    # Parser
    "SQLParser",
    "parse_sql",
    "analyze_sql",
    "format_sql",
    # Validator
    "SQLValidator",
    "validate_sql",
    "is_safe_sql",
    # Dialect
    "SQLDialectConverter",
    "DialectDifferences",
    "convert_sql",
    "get_dialect_converter",
    # Cost Analyzer
    "CostAnalyzer",
    "analyze_sql_cost",
    "get_explain_plan",
]
