import pytest

from repolens.graph.code_graph import CodeGraph
from repolens.graph.graph_query import GraphQuery
from repolens.ingestion.code_unit import (
    CallReference,
    CodeUnit,
    ImportReference,
)


def make_unit(
    unit_id,
    kind,
    name,
    qualified_name,
    parent_id=None,
    imports=(),
    calls=(),
    metadata=None,
):
    return CodeUnit(
        id=unit_id,
        kind=kind,
        name=name,
        qualified_name=qualified_name,
        file_path=f"src/{name}.py",
        start_line=1,
        end_line=20,
        parent_id=parent_id,
        imports=imports,
        calls=calls,
        metadata=metadata or {},
    )


def build_sample_graph():
    units = [
        make_unit(
            "module:main",
            "module",
            "main",
            "main",
            imports=(ImportReference("services.auth"),),
        ),
        make_unit(
            "function:main.login",
            "function",
            "login",
            "main.login",
            parent_id="module:main",
            calls=(CallReference("authenticate"),),
            metadata={"complexity": 3},
        ),
        make_unit(
            "function:main.logout",
            "function",
            "logout",
            "main.logout",
            parent_id="module:main",
            calls=(CallReference("cleanup"),),
        ),
        make_unit(
            "module:auth",
            "module",
            "auth",
            "services.auth",
            imports=(ImportReference("services.database"),),
        ),
        make_unit(
            "function:auth.authenticate",
            "function",
            "authenticate",
            "services.auth.authenticate",
            parent_id="module:auth",
            calls=(CallReference("validate"),),
        ),
        make_unit(
            "function:auth.validate",
            "function",
            "validate",
            "services.auth.validate",
            parent_id="module:auth",
        ),
        make_unit(
            "module:database",
            "module",
            "database",
            "services.database",
        ),
        make_unit(
            "function:main.cleanup",
            "function",
            "cleanup",
            "main.cleanup",
        ),
    ]

    graph = CodeGraph().build(units)
    return graph, GraphQuery(graph)


def test_get_callers():
    _, query = build_sample_graph()

    callers = query.get_callers("function:auth.authenticate")

    assert len(callers) == 1
    assert callers[0]["code_unit_id"] == "function:main.login"


def test_get_callees():
    _, query = build_sample_graph()

    callees = query.get_callees("function:main.login")

    assert len(callees) == 1
    assert callees[0]["code_unit_id"] == "function:auth.authenticate"


def test_call_queries_return_source_metadata():
    _, query = build_sample_graph()

    callees = query.get_callees("function:main.login")

    result = callees[0]
    assert result["type"] == "function"
    assert result["name"] == "authenticate"
    assert result["qualified_name"] == "services.auth.authenticate"
    assert result["file_path"] == "src/authenticate.py"
    assert result["start_line"] == 1
    assert result["end_line"] == 20


def test_callers_of_missing_node_return_empty_list():
    _, query = build_sample_graph()

    assert query.get_callers("missing") == []


def test_callees_of_missing_node_return_empty_list():
    _, query = build_sample_graph()

    assert query.get_callees("missing") == []


def test_direct_module_dependencies():
    _, query = build_sample_graph()

    dependencies = query.get_module_dependencies("module:main")

    assert [item["code_unit_id"] for item in dependencies] == ["module:auth"]


def test_transitive_module_dependencies():
    _, query = build_sample_graph()

    dependencies = query.get_module_dependencies(
        "module:main",
        transitive=True,
        max_depth=2,
    )

    assert {item["code_unit_id"] for item in dependencies} == {
        "module:auth",
        "module:database",
    }


def test_module_dependency_lookup_rejects_non_module():
    _, query = build_sample_graph()

    assert query.get_module_dependencies("function:main.login") == []


def test_traverse_contains_relationship():
    _, query = build_sample_graph()

    results = query.traverse(
        "module:main",
        relationships={"CONTAINS"},
        max_depth=1,
    )

    assert {item["code_unit_id"] for item in results} == {
        "function:main.login",
        "function:main.logout",
    }


def test_traverse_imports_relationship():
    _, query = build_sample_graph()

    results = query.traverse(
        "module:main",
        relationships={"IMPORTS"},
        max_depth=1,
    )

    assert [item["code_unit_id"] for item in results] == ["module:auth"]


def test_traverse_calls_relationship():
    _, query = build_sample_graph()

    results = query.traverse(
        "function:main.login",
        relationships={"CALLS"},
        max_depth=1,
    )

    assert [item["code_unit_id"] for item in results] == ["function:auth.authenticate"]


def test_traversal_respects_max_depth():
    _, query = build_sample_graph()

    results = query.traverse(
        "module:main",
        relationships={"IMPORTS"},
        max_depth=1,
    )

    assert len(results) == 1
    assert results[0]["depth"] == 1


def test_traversal_returns_depth_and_relationship():
    _, query = build_sample_graph()

    results = query.traverse(
        "module:main",
        relationships={"IMPORTS"},
        max_depth=2,
    )

    database = next(
        item for item in results if item["code_unit_id"] == "module:database"
    )

    assert database["depth"] == 2
    assert database["relationship"] == "IMPORTS"
    assert database["parent_id"] == "module:auth"


def test_bfs_and_dfs_return_same_reachable_nodes():
    _, query = build_sample_graph()

    bfs = query.traverse(
        "module:main",
        relationships={"IMPORTS"},
        max_depth=2,
        strategy="bfs",
    )

    dfs = query.traverse(
        "module:main",
        relationships={"IMPORTS"},
        max_depth=2,
        strategy="dfs",
    )

    assert {item["code_unit_id"] for item in bfs} == {
        item["code_unit_id"] for item in dfs
    }


def test_traversal_supports_incoming_relationships():
    _, query = build_sample_graph()

    results = query.traverse(
        "function:auth.authenticate",
        relationships={"CALLS"},
        direction="incoming",
        max_depth=1,
    )

    assert [item["code_unit_id"] for item in results] == ["function:main.login"]


def test_traversal_supports_both_directions():
    _, query = build_sample_graph()

    results = query.traverse(
        "function:auth.authenticate",
        relationships={"CALLS"},
        direction="both",
        max_depth=1,
    )

    assert {item["code_unit_id"] for item in results} == {
        "function:main.login",
        "function:auth.validate",
    }


def test_missing_start_node_returns_empty_list():
    _, query = build_sample_graph()

    assert query.traverse("missing") == []


def test_zero_depth_returns_empty_list():
    _, query = build_sample_graph()

    assert (
        query.traverse(
            "module:main",
            max_depth=0,
        )
        == []
    )


def test_negative_depth_is_rejected():
    _, query = build_sample_graph()

    with pytest.raises(ValueError):
        query.traverse("module:main", max_depth=-1)


def test_invalid_relationship_is_rejected():
    _, query = build_sample_graph()

    with pytest.raises(ValueError):
        query.traverse(
            "module:main",
            relationships={"UNKNOWN"},
        )


def test_invalid_direction_is_rejected():
    _, query = build_sample_graph()

    with pytest.raises(ValueError):
        query.traverse(
            "module:main",
            direction="sideways",
        )


def test_invalid_strategy_is_rejected():
    _, query = build_sample_graph()

    with pytest.raises(ValueError):
        query.traverse(
            "module:main",
            strategy="random",
        )


def test_unresolved_relationships_are_returned():
    units = [
        make_unit(
            "module:main",
            "module",
            "main",
            "main",
            imports=(ImportReference("missing.module"),),
        ),
        make_unit(
            "function:main.login",
            "function",
            "login",
            "main.login",
            parent_id="module:main",
            calls=(CallReference("missing_function"),),
        ),
    ]

    graph = CodeGraph().build(units)
    query = GraphQuery(graph)

    unresolved = query.get_unresolved_relationships()

    assert len(unresolved) == 2
    assert {item["type"] for item in unresolved} == {"IMPORTS", "CALLS"}


def test_unresolved_relationships_can_be_filtered():
    units = [
        make_unit(
            "module:main",
            "module",
            "main",
            "main",
            imports=(ImportReference("missing.module"),),
        ),
    ]

    graph = CodeGraph().build(units)
    query = GraphQuery(graph)

    unresolved = query.get_unresolved_relationships(
        code_unit_id="module:main",
        relationship="IMPORTS",
    )

    assert len(unresolved) == 1
    assert unresolved[0]["target"] == "missing.module"


def test_query_requires_code_graph():
    with pytest.raises(TypeError):
        GraphQuery("not a graph")


def test_metadata_is_copied_into_query_results():
    _, query = build_sample_graph()

    result = query.get_callers("function:auth.authenticate")[0]

    assert result["code_unit_id"] == "function:main.login"
    assert result["metadata"] == {"complexity": 3}
