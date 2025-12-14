# -*- coding: utf-8 -*-
"""
Schema Graph - 基于外键关系的图剪枝

实现基于表关系图的智能 Schema 剪枝算法:
1. 解析 DDL 提取外键关系
2. 构建表关系图 (邻接表)
3. BFS 从种子表扩展关联表
"""

import re
from dataclasses import dataclass, field
from collections import defaultdict, deque
from typing import Optional


@dataclass
class TableNode:
    """表节点"""
    name: str
    columns: list[str] = field(default_factory=list)
    comment: str = ""
    primary_key: Optional[str] = None


@dataclass
class FKEdge:
    """外键边"""
    from_table: str
    from_column: str
    to_table: str
    to_column: str


class SchemaGraph:
    """
    Schema 关系图

    用于基于外键关系进行智能表剪枝
    """

    def __init__(self):
        self.nodes: dict[str, TableNode] = {}
        self.edges: list[FKEdge] = []
        # 邻接表: table_name -> [(related_table, edge)]
        self.adjacency: dict[str, list[tuple[str, FKEdge]]] = defaultdict(list)

    def add_table(self, name: str, columns: list[str] = None, comment: str = "", primary_key: str = None):
        """添加表节点"""
        self.nodes[name] = TableNode(
            name=name,
            columns=columns or [],
            comment=comment,
            primary_key=primary_key,
        )

    def add_foreign_key(self, from_table: str, from_column: str, to_table: str, to_column: str):
        """添加外键关系"""
        edge = FKEdge(
            from_table=from_table,
            from_column=from_column,
            to_table=to_table,
            to_column=to_column,
        )
        self.edges.append(edge)

        # 双向添加到邻接表 (外键关系是双向可达的)
        self.adjacency[from_table].append((to_table, edge))
        self.adjacency[to_table].append((from_table, edge))

    def build_from_ddl(self, ddl: str):
        """
        从 DDL 语句构建关系图

        支持解析:
        - CREATE TABLE 语句
        - FOREIGN KEY 约束
        - REFERENCES 子句
        """
        # 解析 CREATE TABLE 语句
        table_pattern = r'CREATE\s+TABLE\s+[`"\[]?(\w+)[`"\]]?\s*\((.*?)\)(?:\s*;)?'

        for match in re.finditer(table_pattern, ddl, re.IGNORECASE | re.DOTALL):
            table_name = match.group(1)
            table_body = match.group(2)

            # 解析列
            columns = self._parse_columns(table_body)

            # 解析主键
            primary_key = self._parse_primary_key(table_body)

            # 添加表节点
            self.add_table(
                name=table_name,
                columns=columns,
                primary_key=primary_key,
            )

            # 解析外键
            fk_relations = self._parse_foreign_keys(table_name, table_body)
            for fk in fk_relations:
                self.add_foreign_key(**fk)

    def build_from_tables(self, tables: list[dict]):
        """
        从表信息列表构建关系图

        Args:
            tables: 表信息列表，每个元素包含 name, columns, ddl 等字段
        """
        # 第一遍: 添加所有表节点
        for table in tables:
            columns = [c.get("name", "") for c in table.get("columns", [])]
            self.add_table(
                name=table["name"],
                columns=columns,
                comment=table.get("comment", ""),
            )

        # 第二遍: 解析外键关系
        for table in tables:
            ddl = table.get("ddl", "")
            if ddl:
                fk_relations = self._parse_foreign_keys(table["name"], ddl)
                for fk in fk_relations:
                    # 只添加目标表存在的外键
                    if fk["to_table"] in self.nodes:
                        self.add_foreign_key(**fk)

            # 也从列定义中解析外键
            for col in table.get("columns", []):
                if col.get("references"):
                    ref = col["references"]
                    if ref.get("table") in self.nodes:
                        self.add_foreign_key(
                            from_table=table["name"],
                            from_column=col["name"],
                            to_table=ref["table"],
                            to_column=ref.get("column", "id"),
                        )

    def _parse_columns(self, table_body: str) -> list[str]:
        """解析列名"""
        columns = []

        # 按逗号分割，但要处理括号内的逗号
        parts = self._split_by_comma(table_body)

        for part in parts:
            part = part.strip()
            if not part:
                continue

            # 跳过约束定义
            if any(kw in part.upper() for kw in ['PRIMARY KEY', 'FOREIGN KEY', 'UNIQUE', 'CHECK', 'INDEX', 'CONSTRAINT']):
                continue

            # 提取列名 (第一个词)
            col_match = re.match(r'[`"\[]?(\w+)[`"\]]?', part)
            if col_match:
                columns.append(col_match.group(1))

        return columns

    def _parse_primary_key(self, table_body: str) -> Optional[str]:
        """解析主键"""
        # PRIMARY KEY (column)
        pk_match = re.search(r'PRIMARY\s+KEY\s*\(\s*[`"\[]?(\w+)[`"\]]?\s*\)', table_body, re.IGNORECASE)
        if pk_match:
            return pk_match.group(1)

        # column_name ... PRIMARY KEY
        pk_inline = re.search(r'[`"\[]?(\w+)[`"\]]?\s+\w+.*?PRIMARY\s+KEY', table_body, re.IGNORECASE)
        if pk_inline:
            return pk_inline.group(1)

        return None

    def _parse_foreign_keys(self, table_name: str, table_body: str) -> list[dict]:
        """解析外键关系"""
        fk_relations = []

        # FOREIGN KEY (column) REFERENCES table(column)
        fk_pattern = r'FOREIGN\s+KEY\s*\(\s*[`"\[]?(\w+)[`"\]]?\s*\)\s*REFERENCES\s+[`"\[]?(\w+)[`"\]]?\s*\(\s*[`"\[]?(\w+)[`"\]]?\s*\)'

        for match in re.finditer(fk_pattern, table_body, re.IGNORECASE):
            fk_relations.append({
                "from_table": table_name,
                "from_column": match.group(1),
                "to_table": match.group(2),
                "to_column": match.group(3),
            })

        # column_name ... REFERENCES table(column)
        ref_pattern = r'[`"\[]?(\w+)[`"\]]?\s+\w+.*?REFERENCES\s+[`"\[]?(\w+)[`"\]]?\s*\(\s*[`"\[]?(\w+)[`"\]]?\s*\)'

        for match in re.finditer(ref_pattern, table_body, re.IGNORECASE):
            # 避免重复添加
            fk = {
                "from_table": table_name,
                "from_column": match.group(1),
                "to_table": match.group(2),
                "to_column": match.group(3),
            }
            if fk not in fk_relations:
                fk_relations.append(fk)

        return fk_relations

    def _split_by_comma(self, text: str) -> list[str]:
        """按逗号分割，但忽略括号内的逗号"""
        parts = []
        current = []
        depth = 0

        for char in text:
            if char == '(':
                depth += 1
                current.append(char)
            elif char == ')':
                depth -= 1
                current.append(char)
            elif char == ',' and depth == 0:
                parts.append(''.join(current))
                current = []
            else:
                current.append(char)

        if current:
            parts.append(''.join(current))

        return parts

    def get_related_tables(
        self,
        seed_tables: list[str],
        max_hops: int = 2,
        include_seeds: bool = True,
    ) -> list[str]:
        """
        基于种子表进行 BFS 扩展，获取关联表

        Args:
            seed_tables: 种子表列表 (通常是 Rerank 返回的高分表)
            max_hops: 最大跳数 (默认 2 跳)
            include_seeds: 是否包含种子表

        Returns:
            关联表列表 (按距离排序)
        """
        if not seed_tables:
            return []

        # BFS
        visited = set()
        result = []
        queue = deque()

        # 初始化队列
        for table in seed_tables:
            if table in self.nodes:
                queue.append((table, 0))
                visited.add(table)
                if include_seeds:
                    result.append(table)

        while queue:
            current_table, current_hop = queue.popleft()

            if current_hop >= max_hops:
                continue

            # 遍历邻居
            for neighbor, edge in self.adjacency.get(current_table, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    result.append(neighbor)
                    queue.append((neighbor, current_hop + 1))

        return result

    def get_join_path(self, table1: str, table2: str) -> list[FKEdge]:
        """
        获取两个表之间的连接路径

        Args:
            table1: 起始表
            table2: 目标表

        Returns:
            外键边列表，表示连接路径
        """
        if table1 not in self.nodes or table2 not in self.nodes:
            return []

        if table1 == table2:
            return []

        # BFS 找最短路径
        visited = {table1}
        queue = deque([(table1, [])])

        while queue:
            current, path = queue.popleft()

            for neighbor, edge in self.adjacency.get(current, []):
                if neighbor == table2:
                    return path + [edge]

                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [edge]))

        return []  # 无路径

    def get_table_distance(self, table1: str, table2: str) -> int:
        """
        获取两个表之间的距离 (跳数)

        Returns:
            距离，-1 表示不可达
        """
        path = self.get_join_path(table1, table2)
        return len(path) if path else (-1 if table1 != table2 else 0)

    def to_dict(self) -> dict:
        """导出为字典"""
        return {
            "tables": [
                {
                    "name": node.name,
                    "columns": node.columns,
                    "comment": node.comment,
                    "primary_key": node.primary_key,
                }
                for node in self.nodes.values()
            ],
            "foreign_keys": [
                {
                    "from_table": edge.from_table,
                    "from_column": edge.from_column,
                    "to_table": edge.to_table,
                    "to_column": edge.to_column,
                }
                for edge in self.edges
            ],
        }

    def __repr__(self) -> str:
        return f"SchemaGraph(tables={len(self.nodes)}, foreign_keys={len(self.edges)})"


# 便捷函数
def build_schema_graph(tables: list[dict]) -> SchemaGraph:
    """从表信息列表构建 Schema 图"""
    graph = SchemaGraph()
    graph.build_from_tables(tables)
    return graph


def build_schema_graph_from_ddl(ddl: str) -> SchemaGraph:
    """从 DDL 构建 Schema 图"""
    graph = SchemaGraph()
    graph.build_from_ddl(ddl)
    return graph


__all__ = [
    "SchemaGraph",
    "TableNode",
    "FKEdge",
    "build_schema_graph",
    "build_schema_graph_from_ddl",
]
