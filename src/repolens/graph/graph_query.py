from collections import deque
from collections.abc import Iterable
from typing import ClassVar, Literal

from repolens.graph.code_graph import CodeGraph

Relationship = Literal["CONTAINS", "IMPORTS", "CALLS"]
Direction = Literal["outgoing", "incoming", "both"]
TraversalStrategy = Literal["bfs", "dfs"]


class GraphQuery:
    """Provides traversal and relationship queries over a CodeGraph."""

    SUPPORTED_RELATIONSHIPS: ClassVar[frozenset[str]] = frozenset(
        {"CONTAINS", "IMPORTS", "CALLS"}
    )

    def __init__(self, code_graph: CodeGraph):
        if not isinstance(code_graph, CodeGraph):
            raise TypeError("Expected a CodeGraph instance.")

        self._graph = code_graph.graph
        self._unresolved = code_graph.unresolved_relationships

    @staticmethod
    def _node_result(code_unit_id: str, data: dict) -> dict:
        """Create a query result containing CodeUnit information."""
        return {
            "code_unit_id": code_unit_id,
            "type": data.get("type"),
            "name": data.get("name"),
            "qualified_name": data.get("qualified_name"),
            "file_path": data.get("file_path"),
            "start_line": data.get("start_line"),
            "end_line": data.get("end_line"),
            "metadata": dict(data.get("metadata", {})),
        }

    def _get_neighbors(
        self,
        code_unit_id: str,
        relationships: Iterable[str] | None = None,
        direction: Direction = "outgoing",
    ) -> list[tuple[str, str]]:
        """Return neighboring node IDs and relationship types."""
        if direction not in {"outgoing", "incoming", "both"}:
            raise ValueError("Direction must be 'outgoing', 'incoming', or 'both'.")

        allowed = (
            self.SUPPORTED_RELATIONSHIPS
            if relationships is None
            else set(relationships)
        )

        invalid = allowed - self.SUPPORTED_RELATIONSHIPS
        if invalid:
            raise ValueError(f"Unsupported relationships: {sorted(invalid)}")

        neighbors = []

        if direction in {"outgoing", "both"}:
            for _, target, key, data in self._graph.out_edges(
                code_unit_id, keys=True, data=True
            ):
                relationship = data.get("type", key)
                if relationship in allowed:
                    neighbors.append((target, relationship))

        if direction in {"incoming", "both"}:
            for source, _, key, data in self._graph.in_edges(
                code_unit_id, keys=True, data=True
            ):
                relationship = data.get("type", key)
                if relationship in allowed:
                    neighbors.append((source, relationship))

        return sorted(set(neighbors))

    def get_callers(self, code_unit_id: str) -> list[dict]:
        """Find functions or methods that call the specified CodeUnit."""
        if code_unit_id not in self._graph:
            return []

        callers = []

        for source, _, key, data in self._graph.in_edges(
            code_unit_id, keys=True, data=True
        ):
            if data.get("type", key) == "CALLS":
                callers.append(self._node_result(source, self._graph.nodes[source]))

        return sorted(callers, key=lambda item: item["code_unit_id"])

    def get_callees(self, code_unit_id: str) -> list[dict]:
        """Find functions or methods called by the specified CodeUnit."""
        if code_unit_id not in self._graph:
            return []

        callees = []

        for _, target, key, data in self._graph.out_edges(
            code_unit_id, keys=True, data=True
        ):
            if data.get("type", key) == "CALLS":
                callees.append(self._node_result(target, self._graph.nodes[target]))

        return sorted(callees, key=lambda item: item["code_unit_id"])

    def get_module_dependencies(
        self,
        module_id: str,
        transitive: bool = False,
        max_depth: int = 1,
    ) -> list[dict]:
        """
        Find modules imported by a module.

        By default, only direct dependencies are returned.
        Set transitive=True to follow dependency chains.
        """
        if module_id not in self._graph:
            return []

        if self._graph.nodes[module_id].get("type") != "module":
            return []

        if max_depth < 1:
            raise ValueError("max_depth must be at least 1.")

        if not transitive:
            results = []

            for _, target, key, data in self._graph.out_edges(
                module_id, keys=True, data=True
            ):
                if (
                    data.get("type", key) == "IMPORTS"
                    and self._graph.nodes[target].get("type") == "module"
                ):
                    results.append(self._node_result(target, self._graph.nodes[target]))

            return sorted(results, key=lambda item: item["code_unit_id"])

        results = self.traverse(
            module_id,
            relationships={"IMPORTS"},
            direction="outgoing",
            max_depth=max_depth,
            strategy="bfs",
        )

        return [result for result in results if result["type"] == "module"]

    def traverse(
        self,
        start_id: str,
        relationships: Iterable[str] | None = None,
        direction: Direction = "outgoing",
        max_depth: int = 1,
        strategy: TraversalStrategy = "bfs",
    ) -> list[dict]:
        """
        Traverse graph relationships from a starting CodeUnit.

        The starting node is excluded from results.
        Each discovered node is returned once, with its traversal depth
        and the relationship used to reach it.
        """
        if max_depth < 0:
            raise ValueError("max_depth cannot be negative.")

        if strategy not in {"bfs", "dfs"}:
            raise ValueError("Strategy must be 'bfs' or 'dfs'.")

        if direction not in {"outgoing", "incoming", "both"}:
            raise ValueError("Direction must be 'outgoing', 'incoming', or 'both'.")

        allowed = (
            self.SUPPORTED_RELATIONSHIPS
            if relationships is None
            else set(relationships)
        )

        invalid = allowed - self.SUPPORTED_RELATIONSHIPS
        if invalid:
            raise ValueError(f"Unsupported relationships: {sorted(invalid)}")

        if start_id not in self._graph or max_depth == 0:
            return []

        # BFS uses a queue; DFS uses a stack.
        frontier = deque([(start_id, 0)])
        visited = {start_id}
        results = []

        while frontier:
            current_id, current_depth = (
                frontier.popleft() if strategy == "bfs" else frontier.pop()
            )

            if current_depth >= max_depth:
                continue

            neighbors = self._get_neighbors(
                current_id,
                relationships=allowed,
                direction=direction,
            )

            # Reverse insertion for DFS so traversal remains deterministic.
            if strategy == "dfs":
                neighbors = list(reversed(neighbors))

            for neighbor_id, relationship in neighbors:
                if neighbor_id in visited:
                    continue

                visited.add(neighbor_id)
                depth = current_depth + 1

                result = self._node_result(
                    neighbor_id,
                    self._graph.nodes[neighbor_id],
                )
                result["depth"] = depth
                result["relationship"] = relationship
                result["parent_id"] = current_id
                results.append(result)

                frontier.append((neighbor_id, depth))

        return results

    def get_unresolved_relationships(
        self,
        code_unit_id: str | None = None,
        relationship: str | None = None,
    ) -> list[dict]:
        """Retrieve unresolved relationships, optionally filtered by source."""
        if (
            relationship is not None
            and relationship not in self.SUPPORTED_RELATIONSHIPS
        ):
            raise ValueError(f"Unsupported relationship: {relationship}")

        return [
            dict(item)
            for item in self._unresolved
            if (code_unit_id is None or item["source"] == code_unit_id)
            and (relationship is None or item["type"] == relationship)
        ]
