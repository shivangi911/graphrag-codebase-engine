"""Typed graph primitives."""

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class NodeType(StrEnum):
    FILE = "File"
    CLASS = "Class"
    FUNCTION = "Function"
    VARIABLE = "Variable"
    IMPORT = "Import"
    EXTERNAL = "External"


class Relation(StrEnum):
    DEFINES = "DEFINES"
    CALLS = "CALLS"
    INHERITS_FROM = "INHERITS_FROM"
    IMPORTS = "IMPORTS"
    USES_TYPE = "USES_TYPE"


@dataclass
class Node:
    id: str
    type: NodeType
    name: str
    code: str = ""
    file: str = ""
    start_line: int = 0
    end_line: int = 0
    docstring: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Edge:
    source: str
    target: str
    relation: Relation

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ParsedFile:
    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
