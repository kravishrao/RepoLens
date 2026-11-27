from repolens.graph.code_graph import CodeGraph
from repolens.ingestion.code_unit import (
    CallReference,
    CodeUnit,
    ImportReference,
)


def make_unit(
    id,
    kind,
    name,
    qualified_name,
    parent_id=None,
    imports=(),
    calls=(),
):
    return CodeUnit(
        id=id,
        kind=kind,
        name=name,
        qualified_name=qualified_name,
        file_path="src/example.py",
        start_line=1,
        end_line=10,
        parent_id=parent_id,
        imports=imports,
        calls=calls,
    )


def sample_units():
    return [
        make_unit(
            "module:main",
            "module",
            "main",
            "main",
        ),
        make_unit(
            "function:main.login",
            "function",
            "login",
            "main.login",
            parent_id="module:main",
            calls=(CallReference("authenticate"),),
        ),
        make_unit(
            "module:auth",
            "module",
            "auth",
            "services.auth",
        ),
        make_unit(
            "function:auth.authenticate",
            "function",
            "authenticate",
            "services.auth.authenticate",
            parent_id="module:auth",
        ),
    ]


def test_creates_all_code_element_nodes():
    graph = CodeGraph().build(sample_units())

    assert graph.graph.number_of_nodes() == 4

    assert graph.get_node("module:main")["type"] == "module"
    assert graph.get_node("function:main.login")["type"] == "function"


def test_creates_contains_relationships():
    graph = CodeGraph().build(sample_units())

    assert graph.graph.has_edge(
        "module:main",
        "function:main.login",
        key="CONTAINS",
    )


def test_creates_import_relationships():
    units = sample_units()

    main = units[0]

    units[0] = CodeUnit(
        id=main.id,
        kind=main.kind,
        name=main.name,
        qualified_name=main.qualified_name,
        file_path=main.file_path,
        start_line=main.start_line,
        end_line=main.end_line,
        imports=(ImportReference("services.auth"),),
    )

    graph = CodeGraph().build(units)

    assert graph.graph.has_edge(
        "module:main",
        "module:auth",
        key="IMPORTS",
    )


def test_creates_calls_relationship():
    graph = CodeGraph().build(sample_units())

    assert graph.graph.has_edge(
        "function:main.login",
        "function:auth.authenticate",
        key="CALLS",
    )


def test_duplicate_code_units_do_not_create_duplicate_nodes():
    units = sample_units()

    graph = CodeGraph().build(units + units)

    assert graph.graph.number_of_nodes() == 4


def test_duplicate_relationships_are_avoided():
    units = sample_units()

    main = units[0]

    units[0] = CodeUnit(
        id=main.id,
        kind=main.kind,
        name=main.name,
        qualified_name=main.qualified_name,
        file_path=main.file_path,
        start_line=main.start_line,
        end_line=main.end_line,
        imports=(
            ImportReference("services.auth"),
            ImportReference("services.auth"),
        ),
    )

    graph = CodeGraph().build(units)

    assert (
        graph.graph.number_of_edges(
            "module:main",
            "module:auth",
        )
        == 1
    )


def test_unresolved_calls_are_handled_gracefully():
    units = sample_units()

    login = units[1]

    units[1] = CodeUnit(
        id=login.id,
        kind=login.kind,
        name=login.name,
        qualified_name=login.qualified_name,
        file_path=login.file_path,
        start_line=login.start_line,
        end_line=login.end_line,
        parent_id=login.parent_id,
        calls=(CallReference("unknown_function"),),
    )

    graph = CodeGraph().build(units)

    assert graph.graph.number_of_nodes() == 4
    assert len(graph.unresolved_relationships) == 1


def test_graph_statistics():
    graph = CodeGraph().build(sample_units())

    stats = graph.get_stats()

    assert stats["nodes"] == 4
    assert stats["edges"] == 3
    assert stats["relationships"]["CONTAINS"] == 2
    assert stats["relationships"]["CALLS"] == 1
    assert stats["unresolved"] == 0


def test_creates_class_and_method_nodes():
    units = [
        make_unit("module:app", "module", "app", "app"),
        make_unit(
            "class:app.User",
            "class",
            "User",
            "app.User",
            parent_id="module:app",
        ),
        make_unit(
            "method:app.User.login",
            "method",
            "login",
            "app.User.login",
            parent_id="class:app.User",
        ),
    ]

    graph = CodeGraph().build(units)

    assert graph.graph.number_of_nodes() == 3
    assert graph.get_node("class:app.User")["type"] == "class"
    assert graph.get_node("method:app.User.login")["type"] == "method"
    assert graph.graph.has_edge(
        "class:app.User",
        "method:app.User.login",
        key="CONTAINS",
    )


def test_ambiguous_call_is_recorded():
    units = [
        make_unit("module:a", "module", "a", "a"),
        make_unit("module:b", "module", "b", "b"),
        make_unit(
            "function:a.process",
            "function",
            "process",
            "a.process",
            parent_id="module:a",
        ),
        make_unit(
            "function:b.process",
            "function",
            "process",
            "b.process",
            parent_id="module:b",
        ),
        make_unit(
            "function:main",
            "function",
            "main",
            "main",
            calls=(CallReference("process"),),
        ),
    ]

    graph = CodeGraph().build(units)

    assert not graph.graph.has_edge(
        "function:main",
        "function:a.process",
        key="CALLS",
    )
    assert not graph.graph.has_edge(
        "function:main",
        "function:b.process",
        key="CALLS",
    )
    assert any(
        item["reason"] == "Ambiguous call target."
        for item in graph.unresolved_relationships
    )


def test_ambiguous_import_is_recorded():
    units = [
        make_unit("module:main", "module", "main", "main"),
        make_unit("module:one", "module", "common", "common"),
        make_unit("module:two", "module", "common", "common"),
    ]

    units[0] = CodeUnit(
        id="module:main",
        kind="module",
        name="main",
        qualified_name="main",
        file_path="src/example.py",
        start_line=1,
        end_line=10,
        imports=(ImportReference("common"),),
    )

    graph = CodeGraph().build(units)

    assert not graph.graph.has_edge(
        "module:main",
        "module:one",
        key="IMPORTS",
    )
    assert not graph.graph.has_edge(
        "module:main",
        "module:two",
        key="IMPORTS",
    )
    assert any(
        item["reason"] == "Ambiguous imported module."
        for item in graph.unresolved_relationships
    )


def test_missing_parent_is_recorded():
    units = [
        make_unit(
            "function:orphan",
            "function",
            "orphan",
            "orphan",
            parent_id="module:missing",
        )
    ]

    graph = CodeGraph().build(units)

    assert graph.graph.number_of_nodes() == 1
    assert len(graph.unresolved_relationships) == 1
    assert graph.unresolved_relationships[0]["type"] == "CONTAINS"


def test_conflicting_code_unit_ids_raise_error():
    first = make_unit(
        "function:duplicate",
        "function",
        "first",
        "first",
    )
    second = make_unit(
        "function:duplicate",
        "function",
        "second",
        "second",
    )

    import pytest

    with pytest.raises(ValueError, match="Conflicting CodeUnit ID"):
        CodeGraph().build([first, second])


def test_multiple_relationship_types_between_same_nodes():
    units = [
        make_unit("module:a", "module", "a", "a"),
        make_unit(
            "function:a.run",
            "function",
            "run",
            "a.run",
            parent_id="module:a",
            calls=(CallReference("a.run"),),
        ),
    ]

    graph = CodeGraph().build(units)

    assert graph.graph.has_edge(
        "module:a",
        "function:a.run",
        key="CONTAINS",
    )
    assert graph.graph.has_edge(
        "function:a.run",
        "function:a.run",
        key="CALLS",
    )


def test_invalid_input_does_not_clear_existing_graph():
    graph = CodeGraph().build(sample_units())

    original_node_count = graph.graph.number_of_nodes()

    import pytest

    with pytest.raises(TypeError, match="Expected CodeUnit instances"):
        graph.build([sample_units()[0], "invalid"])

    assert graph.graph.number_of_nodes() == original_node_count
    assert graph.get_node("module:main") is not None
