from collections import defaultdict
from collections.abc import Iterable

import networkx as nx

from repolens.ingestion.code_unit import CodeUnit


class CodeGraph:
    """
    Builds a directed, typed graph of repository code elements.

    Nodes:
        module, class, function, method

    Relationships:
        CONTAINS, IMPORTS, CALLS
    """

    def __init__(self):
        self._graph = nx.MultiDiGraph()
        self._units: dict[str, CodeUnit] = {}
        self.unresolved_relationships: list[dict] = []

    @property
    def graph(self) -> nx.MultiDiGraph:
        """Expose the graph for downstream components."""
        return self._graph

    def build(self, code_units: Iterable[CodeUnit]):
        """Build the graph from extracted CodeUnits."""

        # Validate all input before modifying the current graph.
        units = list(code_units)
        validated_units: dict[str, CodeUnit] = {}

        for unit in units:
            if not isinstance(unit, CodeUnit):
                raise TypeError("Expected CodeUnit instances.")

            if unit.id in validated_units:
                if validated_units[unit.id] != unit:
                    raise ValueError(f"Conflicting CodeUnit ID: {unit.id}")
                continue

            validated_units[unit.id] = unit

        # Input is valid. Start building the new graph.
        self._graph.clear()
        self._units = validated_units
        self.unresolved_relationships.clear()

        # Phase 1: Create unique nodes.
        for unit in self._units.values():
            self._graph.add_node(
                unit.id,
                code_unit_id=unit.id,
                type=unit.kind,
                name=unit.name,
                qualified_name=unit.qualified_name,
                file_path=unit.file_path,
                start_line=unit.start_line,
                end_line=unit.end_line,
                metadata=dict(unit.metadata),
            )

        # Build lookup indexes.
        modules = defaultdict(list)
        qualified_names = defaultdict(list)
        simple_names = defaultdict(list)

        for unit in self._units.values():
            qualified_names[unit.qualified_name].append(unit)
            simple_names[unit.name].append(unit)

            if unit.kind == "module":
                modules[self._normalize_module(unit.qualified_name)].append(unit)

        # Phase 2: Structural relationships.
        for unit in self._units.values():
            if unit.parent_id:
                parent = self._units.get(unit.parent_id)

                if parent is None:
                    self._record_unresolved(
                        "CONTAINS",
                        unit.id,
                        unit.parent_id,
                        "Parent CodeUnit not found.",
                    )
                else:
                    self._add_relationship(parent.id, unit.id, "CONTAINS")

        # Phase 3: Import relationships.
        for unit in self._units.values():
            if unit.kind != "module":
                continue

            for reference in unit.imports:
                target_name = self._normalize_module(reference.module)
                matches = modules.get(target_name, [])

                if len(matches) == 1:
                    self._add_relationship(unit.id, matches[0].id, "IMPORTS")
                elif not matches:
                    self._record_unresolved(
                        "IMPORTS",
                        unit.id,
                        reference.module,
                        "Imported module not found.",
                    )
                else:
                    self._record_unresolved(
                        "IMPORTS",
                        unit.id,
                        reference.module,
                        "Ambiguous imported module.",
                    )

        # Phase 4: Function and method calls.
        for unit in self._units.values():
            if unit.kind not in {"function", "method"}:
                continue

            for reference in unit.calls:
                target = reference.target.strip()
                matches = qualified_names.get(target, [])

                if not matches and "." not in target:
                    matches = simple_names.get(target, [])

                matches = [
                    match for match in matches if match.kind in {"function", "method"}
                ]

                if len(matches) == 1:
                    self._add_relationship(unit.id, matches[0].id, "CALLS")
                elif not matches:
                    self._record_unresolved(
                        "CALLS",
                        unit.id,
                        target,
                        "Call target not found.",
                    )
                else:
                    self._record_unresolved(
                        "CALLS",
                        unit.id,
                        target,
                        "Ambiguous call target.",
                    )

        return self

    @staticmethod
    def _normalize_module(module_name: str) -> str:
        return module_name.strip().replace("/", ".").removesuffix(".py")

    def _add_relationship(
        self,
        source: str,
        target: str,
        relationship: str,
    ):
        """Add an edge without duplicating the same relationship."""

        if not self._graph.has_edge(
            source,
            target,
            key=relationship,
        ):
            self._graph.add_edge(
                source,
                target,
                key=relationship,
                type=relationship,
            )

    def _record_unresolved(
        self,
        relationship: str,
        source: str,
        target: str,
        reason: str,
    ):
        self.unresolved_relationships.append(
            {
                "type": relationship,
                "source": source,
                "target": target,
                "reason": reason,
            }
        )

    def get_node(self, code_unit_id: str):
        """Retrieve a node by its CodeUnit identifier."""
        if code_unit_id not in self._graph:
            return None

        return self._graph.nodes[code_unit_id]

    def get_relationships(
        self,
        code_unit_id: str,
        relationship: str | None = None,
    ):
        """Retrieve outgoing relationships from a node."""

        if code_unit_id not in self._graph:
            return []

        results = []

        for _, target, _key, data in self._graph.out_edges(
            code_unit_id,
            keys=True,
            data=True,
        ):
            if relationship is None or data["type"] == relationship:
                results.append(
                    {
                        "source": code_unit_id,
                        "target": target,
                        "relationship": data["type"],
                    }
                )

        return results

    def get_stats(self):
        """Return basic graph statistics."""
        relationship_counts = defaultdict(int)

        for _, _, _, data in self._graph.edges(
            keys=True,
            data=True,
        ):
            relationship_counts[data["type"]] += 1

        return {
            "nodes": self._graph.number_of_nodes(),
            "edges": self._graph.number_of_edges(),
            "relationships": dict(relationship_counts),
            "unresolved": len(self.unresolved_relationships),
        }
