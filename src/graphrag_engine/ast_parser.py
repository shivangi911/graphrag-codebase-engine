"""Tree-sitter based AST visitor that emits code-knowledge-graph nodes and edges."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import tree_sitter_python as tspython
from tree_sitter import Language, Parser
from tree_sitter import Node as TSNode

from graphrag_engine.models import Edge, Node, NodeType, ParsedFile, Relation

EXTERNAL_PREFIX = "ext::"


@dataclass
class _Scope:
    node_id: str
    qualname: str
    kind: NodeType


class CodeASTParser:
    """Parses Python source into File/Class/Function nodes and relation edges."""

    def __init__(self) -> None:
        self.parser = Parser(Language(tspython.language()))

    # -- public API -----------------------------------------------------
    def parse_source_code(self, file_path: str, source_code: str) -> ParsedFile:
        src = source_code.encode("utf8")
        tree = self.parser.parse(src)
        result = ParsedFile()
        result.nodes.append(
            Node(
                id=file_path,
                type=NodeType.FILE,
                name=file_path,
                file=file_path,
                start_line=1,
                end_line=src.count(b"\n") + 1,
            )
        )
        root_scope = _Scope(file_path, "", NodeType.FILE)
        self._visit(tree.root_node, src, file_path, root_scope, result)
        return result

    def parse_to_dict(self, file_path: str, source_code: str) -> dict[str, Any]:
        parsed = self.parse_source_code(file_path, source_code)
        return {
            "nodes": [n.to_dict() for n in parsed.nodes],
            "edges": [e.to_dict() for e in parsed.edges],
        }

    # -- helpers --------------------------------------------------------
    @staticmethod
    def _text(node: TSNode, src: bytes) -> str:
        return src[node.start_byte : node.end_byte].decode("utf8", errors="replace")

    def _docstring(self, node: TSNode, src: bytes) -> str:
        body = node.child_by_field_name("body")
        if body is None or not body.children:
            return ""
        first = body.children[0]
        if first.type == "expression_statement" and first.children:
            inner = first.children[0]
            if inner.type == "string":
                return self._text(inner, src).strip("\"' \n")
        return ""

    def _ext(self, result: ParsedFile, name: str) -> str:
        ext_id = f"{EXTERNAL_PREFIX}{name}"
        if not any(n.id == ext_id for n in result.nodes):
            result.nodes.append(Node(id=ext_id, type=NodeType.EXTERNAL, name=name))
        return ext_id

    def _identifiers(self, node: TSNode, src: bytes) -> list[str]:
        names: list[str] = []
        stack = [node]
        while stack:
            cur = stack.pop()
            if cur.type == "identifier":
                names.append(self._text(cur, src))
            else:
                stack.extend(cur.children)
        return names

    def _add_type_uses(self, func: TSNode, src: bytes, owner: str, result: ParsedFile) -> None:
        type_nodes: list[TSNode] = []
        params = func.child_by_field_name("parameters")
        if params is not None:
            for p in params.children:
                if p.type in ("typed_parameter", "typed_default_parameter"):
                    t = p.child_by_field_name("type")
                    if t is not None:
                        type_nodes.append(t)
        ret = func.child_by_field_name("return_type")
        if ret is not None:
            type_nodes.append(ret)
        seen: set[str] = set()
        for t in type_nodes:
            for name in self._identifiers(t, src):
                if name not in seen:
                    seen.add(name)
                    result.edges.append(Edge(owner, f"{EXTERNAL_PREFIX}{name}", Relation.USES_TYPE))

    # -- traversal ------------------------------------------------------
    def _visit(
        self, node: TSNode, src: bytes, file_path: str, scope: _Scope, result: ParsedFile
    ) -> None:
        child_scope = scope

        if node.type == "class_definition":
            name_node = node.child_by_field_name("name")
            if name_node is not None:
                name = self._text(name_node, src)
                qual = f"{scope.qualname}.{name}" if scope.qualname else name
                node_id = f"{file_path}::{qual}"
                result.nodes.append(
                    Node(
                        id=node_id,
                        type=NodeType.CLASS,
                        name=name,
                        code=self._text(node, src),
                        file=file_path,
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        docstring=self._docstring(node, src),
                    )
                )
                result.edges.append(Edge(scope.node_id, node_id, Relation.DEFINES))
                bases = node.child_by_field_name("superclasses")
                if bases is not None:
                    for b in bases.children:
                        if b.type in ("identifier", "attribute"):
                            result.edges.append(
                                Edge(
                                    node_id,
                                    f"{EXTERNAL_PREFIX}{self._text(b, src)}",
                                    Relation.INHERITS_FROM,
                                )
                            )
                child_scope = _Scope(node_id, qual, NodeType.CLASS)

        elif node.type == "function_definition":
            name_node = node.child_by_field_name("name")
            if name_node is not None:
                name = self._text(name_node, src)
                qual = f"{scope.qualname}.{name}" if scope.qualname else name
                node_id = f"{file_path}::{qual}"
                result.nodes.append(
                    Node(
                        id=node_id,
                        type=NodeType.FUNCTION,
                        name=name,
                        code=self._text(node, src),
                        file=file_path,
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        docstring=self._docstring(node, src),
                    )
                )
                result.edges.append(Edge(scope.node_id, node_id, Relation.DEFINES))
                self._add_type_uses(node, src, node_id, result)
                child_scope = _Scope(node_id, qual, NodeType.FUNCTION)

        elif node.type == "call":
            func_node = node.child_by_field_name("function")
            if func_node is not None and func_node.type in ("identifier", "attribute"):
                callee = self._text(func_node, src)
                result.edges.append(
                    Edge(scope.node_id, f"{EXTERNAL_PREFIX}{callee}", Relation.CALLS)
                )

        elif node.type == "import_statement":
            for child in node.children:
                if child.type == "dotted_name":
                    mod = self._text(child, src)
                elif child.type == "aliased_import":
                    inner = child.child_by_field_name("name")
                    mod = self._text(inner, src) if inner is not None else ""
                else:
                    continue
                if mod:
                    result.edges.append(
                        Edge(file_path, self._ext(result, f"module:{mod}"), Relation.IMPORTS)
                    )

        elif node.type == "import_from_statement":
            mod_node = node.child_by_field_name("module_name")
            if mod_node is not None:
                mod = self._text(mod_node, src)
                result.edges.append(
                    Edge(file_path, self._ext(result, f"module:{mod}"), Relation.IMPORTS)
                )

        for child in node.children:
            self._visit(child, src, file_path, child_scope, result)
