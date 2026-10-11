from graphrag_engine.ast_parser import CodeASTParser
from graphrag_engine.models import NodeType, Relation

SRC = '''
import os
from pkg.util import helper

class Base:
    pass

class Child(Base):
    """Child doc."""
    def run(self, x: int) -> Result:
        helper(x)
        return self.other()

    def other(self):
        def inner():
            return os.getcwd()
        return inner()

def top():
    return Child().run(1)
'''


def parse():
    return CodeASTParser().parse_source_code("m.py", SRC)


def ids(parsed, t):
    return {n.id for n in parsed.nodes if n.type == t}


def rels(parsed, relation):
    return {(e.source, e.target) for e in parsed.edges if e.relation == relation}


def test_nodes_are_qualified():
    p = parse()
    assert ids(p, NodeType.CLASS) == {"m.py::Base", "m.py::Child"}
    assert ids(p, NodeType.FUNCTION) == {
        "m.py::Child.run",
        "m.py::Child.other",
        "m.py::Child.other.inner",
        "m.py::top",
    }


def test_defines_hierarchy():
    d = rels(parse(), Relation.DEFINES)
    assert ("m.py", "m.py::Child") in d
    assert ("m.py::Child", "m.py::Child.run") in d
    assert ("m.py::Child.other", "m.py::Child.other.inner") in d


def test_calls_attributed_to_enclosing_function():
    c = rels(parse(), Relation.CALLS)
    assert ("m.py::Child.run", "ext::helper") in c
    assert ("m.py::Child.run", "ext::self.other") in c
    assert ("m.py::Child.other.inner", "ext::os.getcwd") in c
    assert ("m.py::top", "ext::Child") in c


def test_inheritance_imports_and_types():
    p = parse()
    assert ("m.py::Child", "ext::Base") in rels(p, Relation.INHERITS_FROM)
    imp = rels(p, Relation.IMPORTS)
    assert ("m.py", "ext::module:os") in imp
    assert ("m.py", "ext::module:pkg.util") in imp
    uses = rels(p, Relation.USES_TYPE)
    assert ("m.py::Child.run", "ext::int") in uses
    assert ("m.py::Child.run", "ext::Result") in uses


def test_docstring_and_lines():
    child = next(n for n in parse().nodes if n.id == "m.py::Child")
    assert child.docstring == "Child doc."
    assert child.start_line < child.end_line


def test_syntax_error_does_not_raise():
    p = CodeASTParser().parse_source_code(
        "bad.py", "def f(:\n  pass\nclass A:\n  def g(self): pass"
    )
    assert any(n.id == "bad.py::A" for n in p.nodes)


def test_non_ascii_source_uses_byte_offsets_correctly():
    p = CodeASTParser().parse_source_code("u.py", 's = "héllo wörld"\ndef f():\n    return 1\n')
    f = next(n for n in p.nodes if n.id == "u.py::f")
    assert f.code.startswith("def f()")


def test_parse_to_dict_roundtrip_shape():
    d = CodeASTParser().parse_to_dict("a.py", "def f(): pass")
    assert {"nodes", "edges"} == set(d)
    assert d["nodes"][0]["type"] == "File"
