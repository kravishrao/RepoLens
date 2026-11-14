import pytest

from repolens.graph.code_graph import CodeGraph
from repolens.graph.context_expander import ContextExpander
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
        start_line=2,
        end_line=20,
        parent_id=parent_id,
        imports=imports,
        calls=calls,
        metadata=metadata or {},
    )


def build_context_graph():
    units = [
        make_unit(
            "module:main",
            "module",
            "main",
            "main",
            imports=(ImportReference("services.auth"),),
            metadata={"layer": "application"},
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
            "function:main.logout",
            "function",
            "logout",
            "main.logout",
            parent_id="module:main",
        ),
    ]

    graph = CodeGraph().build(units)
    return graph, units, ContextExpander(graph)


def test_initial_code_units_are_preserved():
    _, units, expander = build_context_graph()

    result = expander.expand([units[1]], max_depth=0)

    assert result == [units[1]]


def test_expands_direct_relationships():
    _, units, expander = build_context_graph()

    result = expander.expand(
        [units[1]],
        max_depth=1,
    )

    result_ids = {unit.id for unit in result}

    assert "function:main.login" in result_ids
    assert "module:main" in result_ids
    assert "function:auth.authenticate" in result_ids


def test_expands_multi_hop_relationships():
    _, units, expander = build_context_graph()

    result = expander.expand(
        [units[1]],
        max_depth=3,
    )

    result_ids = {unit.id for unit in result}

    assert "function:auth.authenticate" in result_ids
    assert "function:auth.validate" in result_ids
    assert "module:auth" in result_ids
    assert "module:database" in result_ids


def test_expansion_respects_max_depth():
    _, units, expander = build_context_graph()

    result = expander.expand(
        [units[1]],
        max_depth=1,
    )

    result_ids = {unit.id for unit in result}

    assert "function:auth.validate" not in result_ids
    assert "module:database" not in result_ids


def test_expansion_respects_expansion_limit():
    _, units, expander = build_context_graph()

    result = expander.expand(
        [units[1]],
        max_depth=3,
        max_expanded_units=2,
    )

    # One initial CodeUnit plus at most two expanded CodeUnits.
    assert len(result) == 3
    assert result[0] == units[1]


def test_initial_units_do_not_count_towards_expansion_limit():
    _, units, expander = build_context_graph()

    result = expander.expand(
        [units[1], units[4]],
        max_depth=2,
        max_expanded_units=1,
    )

    assert result[0] == units[1]
    assert result[1] == units[4]
    assert len(result) == 3


def test_duplicate_initial_units_are_removed():
    _, units, expander = build_context_graph()

    result = expander.expand(
        [units[1], units[1]],
        max_depth=0,
    )

    assert len(result) == 1
    assert result[0].id == units[1].id


def test_duplicate_expanded_units_are_removed():
    _, units, expander = build_context_graph()

    result = expander.expand(
        [units[1], units[3]],
        max_depth=2,
    )

    result_ids = [unit.id for unit in result]

    assert len(result_ids) == len(set(result_ids))


def test_import_relationship_expansion():
    _, units, expander = build_context_graph()

    result = expander.expand(
        [units[0]],
        max_depth=2,
        relationships={"IMPORTS"},
    )

    result_ids = {unit.id for unit in result}

    assert "module:auth" in result_ids
    assert "module:database" in result_ids
    assert "function:main.login" not in result_ids


def test_contains_relationship_expansion():
    _, units, expander = build_context_graph()

    result = expander.expand(
        [units[0]],
        max_depth=1,
        relationships={"CONTAINS"},
    )

    result_ids = {unit.id for unit in result}

    assert "function:main.login" in result_ids
    assert "function:main.logout" in result_ids


def test_calls_relationship_expansion():
    _, units, expander = build_context_graph()

    result = expander.expand(
        [units[1]],
        max_depth=2,
        relationships={"CALLS"},
    )

    result_ids = {unit.id for unit in result}

    assert "function:auth.authenticate" in result_ids
    assert "function:auth.validate" in result_ids
    assert "module:main" not in result_ids


def test_source_metadata_is_preserved():
    _, units, expander = build_context_graph()

    result = expander.expand(
        [units[1]],
        max_depth=2,
    )

    authenticate = next(
        unit for unit in result if unit.id == "function:auth.authenticate"
    )

    assert authenticate.file_path == "src/authenticate.py"
    assert authenticate.qualified_name == "services.auth.authenticate"
    assert authenticate.start_line == 2
    assert authenticate.end_line == 20


def test_original_code_unit_objects_are_returned():
    graph, units, expander = build_context_graph()

    result = expander.expand([units[1]], max_depth=1)

    expanded = next(unit for unit in result if unit.id == "function:auth.authenticate")

    assert expanded is graph.get_code_unit(expanded.id)


def test_missing_graph_node_is_preserved_without_expansion():
    _, _, expander = build_context_graph()

    external_unit = make_unit(
        "function:external",
        "function",
        "external",
        "external",
    )

    result = expander.expand([external_unit], max_depth=2)

    assert result == [external_unit]


def test_invalid_initial_input_is_rejected():
    _, _, expander = build_context_graph()

    with pytest.raises(TypeError, match="Expected CodeUnit"):
        expander.expand(["invalid"])


def test_negative_depth_is_rejected():
    _, units, expander = build_context_graph()

    with pytest.raises(ValueError, match="max_depth"):
        expander.expand([units[1]], max_depth=-1)


def test_negative_expansion_limit_is_rejected():
    _, units, expander = build_context_graph()

    with pytest.raises(ValueError, match="max_expanded_units"):
        expander.expand(
            [units[1]],
            max_expanded_units=-1,
        )


def test_invalid_relationship_is_rejected():
    _, units, expander = build_context_graph()

    with pytest.raises(ValueError, match="Unsupported relationships"):
        expander.expand(
            [units[1]],
            relationships={"UNKNOWN"},
        )


def test_expander_requires_code_graph():
    with pytest.raises(TypeError, match="Expected a CodeGraph"):
        ContextExpander("invalid")
