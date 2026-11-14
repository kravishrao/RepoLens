from collections.abc import Iterable

from repolens.graph.code_graph import CodeGraph
from repolens.graph.graph_query import GraphQuery
from repolens.ingestion.code_unit import CodeUnit


class ContextExpander:
    """
    Expands initially retrieved CodeUnits using repository graph relationships.

    Initial CodeUnits are always preserved. Expansion limits apply only
    to additional CodeUnits discovered through graph traversal.
    """

    SUPPORTED_RELATIONSHIPS = frozenset({"CONTAINS", "IMPORTS", "CALLS"})

    def __init__(self, code_graph: CodeGraph):
        if not isinstance(code_graph, CodeGraph):
            raise TypeError("Expected a CodeGraph instance.")

        self._code_graph = code_graph
        self._graph_query = GraphQuery(code_graph)

    def expand(
        self,
        initial_code_units: Iterable[CodeUnit],
        max_depth: int = 1,
        max_expanded_units: int = 50,
        relationships: Iterable[str] | None = None,
    ) -> list[CodeUnit]:
        """
        Expand initial CodeUnits with related repository elements.

        Initial CodeUnits are preserved and do not count towards
        max_expanded_units.

        Returns deduplicated original CodeUnit objects.
        """

        if (
            not isinstance(max_depth, int)
            or isinstance(max_depth, bool)
            or max_depth < 0
        ):
            raise ValueError("max_depth must be a non-negative integer.")

        if (
            not isinstance(max_expanded_units, int)
            or isinstance(max_expanded_units, bool)
            or max_expanded_units < 0
        ):
            raise ValueError("max_expanded_units must be a non-negative integer.")

        if relationships is None:
            allowed_relationships = self.SUPPORTED_RELATIONSHIPS
        else:
            allowed_relationships = set(relationships)

        invalid = allowed_relationships - self.SUPPORTED_RELATIONSHIPS
        if invalid:
            raise ValueError(f"Unsupported relationships: {sorted(invalid)}")

        initial_units = list(initial_code_units)

        for unit in initial_units:
            if not isinstance(unit, CodeUnit):
                raise TypeError("Expected CodeUnit instances.")

        results: list[CodeUnit] = []
        included_ids: set[str] = set()

        # Preserve initial units and remove duplicate IDs.
        for unit in initial_units:
            if unit.id not in included_ids:
                results.append(unit)
                included_ids.add(unit.id)

        if max_depth == 0 or max_expanded_units == 0:
            return results

        # IMPORTANT: Freeze the roots before expansion.
        # Newly discovered units must not restart traversal.
        traversal_roots = list(results)

        expanded_count = 0

        for initial_unit in traversal_roots:
            if expanded_count >= max_expanded_units:
                break

            if initial_unit.id not in self._code_graph.graph:
                continue

            related_nodes = self._graph_query.traverse(
                start_id=initial_unit.id,
                relationships=allowed_relationships,
                direction="both",
                max_depth=max_depth,
                strategy="bfs",
            )

            for node in related_nodes:
                if expanded_count >= max_expanded_units:
                    break

                related_id = node["code_unit_id"]

                if related_id in included_ids:
                    continue

                related_unit = self._code_graph.get_code_unit(related_id)

                if related_unit is None:
                    continue

                results.append(related_unit)
                included_ids.add(related_id)
                expanded_count += 1

        return results
