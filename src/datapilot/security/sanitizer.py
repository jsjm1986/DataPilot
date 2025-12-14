# -*- coding: utf-8 -*-
"""
PII Sanitizer
Detect and mask personally identifiable information
"""

import re
from typing import Any, Optional


class PIISanitizer:
    """PII Detection and Masking"""

    # PII patterns
    PATTERNS = {
        "phone": {
            "pattern": r"1[3-9]\d{9}",
            "mask": "***********",
            "description": "Chinese mobile phone",
        },
        "email": {
            "pattern": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
            "mask": "***@***.***",
            "description": "Email address",
        },
        "id_card": {
            "pattern": r"\d{17}[\dXx]",
            "mask": "******************",
            "description": "Chinese ID card",
        },
        "bank_card": {
            "pattern": r"\d{16,19}",
            "mask": "****************",
            "description": "Bank card number",
        },
        "ip_address": {
            "pattern": r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}",
            "mask": "***.***.***.***",
            "description": "IP address",
        },
    }

    def __init__(self, enabled_types: Optional[list[str]] = None):
        """
        Initialize sanitizer

        Args:
            enabled_types: List of PII types to detect, None for all
        """
        self.enabled_types = enabled_types or list(self.PATTERNS.keys())
        self._compiled = {
            name: re.compile(info["pattern"])
            for name, info in self.PATTERNS.items()
            if name in self.enabled_types
        }

    def detect(self, text: str) -> list[dict]:
        """
        Detect PII in text

        Args:
            text: Text to scan

        Returns:
            List of detected PII items
        """
        detected = []

        for pii_type, pattern in self._compiled.items():
            matches = pattern.findall(text)
            for match in matches:
                detected.append({
                    "type": pii_type,
                    "value": match,
                    "description": self.PATTERNS[pii_type]["description"],
                })

        return detected

    def mask(self, text: str) -> str:
        """
        Mask PII in text

        Args:
            text: Text to sanitize

        Returns:
            Text with PII masked
        """
        result = text

        for pii_type, pattern in self._compiled.items():
            mask = self.PATTERNS[pii_type]["mask"]
            result = pattern.sub(mask, result)

        return result

    def sanitize_dict(self, data: dict) -> dict:
        """
        Sanitize PII in dictionary values

        Args:
            data: Dictionary to sanitize

        Returns:
            Sanitized dictionary
        """
        result = {}

        for key, value in data.items():
            if isinstance(value, str):
                result[key] = self.mask(value)
            elif isinstance(value, dict):
                result[key] = self.sanitize_dict(value)
            elif isinstance(value, list):
                result[key] = [
                    self.sanitize_dict(item) if isinstance(item, dict)
                    else self.mask(item) if isinstance(item, str)
                    else item
                    for item in value
                ]
            else:
                result[key] = value

        return result

    def sanitize_results(self, results: list[dict]) -> list[dict]:
        """
        Sanitize PII in query results

        Args:
            results: List of result dictionaries

        Returns:
            Sanitized results
        """
        return [self.sanitize_dict(row) for row in results]


def sanitize_output(data: Any, pii_types: Optional[list[str]] = None) -> Any:
    """
    Convenience function for sanitizing output

    Args:
        data: Data to sanitize
        pii_types: PII types to mask

    Returns:
        Sanitized data
    """
    sanitizer = PIISanitizer(pii_types)

    if isinstance(data, str):
        return sanitizer.mask(data)
    elif isinstance(data, dict):
        return sanitizer.sanitize_dict(data)
    elif isinstance(data, list):
        return sanitizer.sanitize_results(data)
    else:
        return data


__all__ = ["PIISanitizer", "sanitize_output"]
