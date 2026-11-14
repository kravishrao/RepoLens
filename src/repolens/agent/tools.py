from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from repolens.graph.code_graph import CodeGraph
from repolens.graph.graph_query import GraphQuery
from repolens.ingestion.code_unit import CodeUnit
from repolens.retrieval.bm25_retriever import BM25Retriever
from repolens.retrieval.semantic_retriever import SemanticSearchResult


class SemanticRetriever(Protocol):
    """Interface expected from an optional semantic retrieval component."""

    def query(
        self,
        query_text: str,
        *,
        top_k: int = 10,
    ) -> Iterable[SemanticSearchResult | CodeUnit]:
        """Return CodeUnits or ranked semantic search results."""
        ...


class InvestigationTools:
    """Provides repository investigation tools for the LangGraph workflow."""

    def __init__(
        self,
        code_units: Iterable[CodeUnit],
        code_graph: CodeGraph,
        semantic_retriever: SemanticRetriever | None = None,
    ) -> None:
        units = list(code_units)

        if not all(isinstance(unit, CodeUnit) for unit in units):
            raise TypeError("Expected CodeUnit instances.")

        if not isinstance(code_graph, CodeGraph):
            raise TypeError("Expected a CodeGraph instance.")

        graph_units = {unit.id: unit for unit in code_graph.get_code_units()}
        supplied_units = {unit.id: unit for unit in units}

        if graph_units != supplied_units:
            raise ValueError(
                "CodeGraph and investigation tools must use "
                "the same CodeUnit collection."
            )

        self._code_units = supplied_units
        self._code_graph = code_graph
        self._graph_query = GraphQuery(code_graph)
        self._keyword_retriever = BM25Retriever(units)
        self._semantic_retriever = semantic_retriever

    @property
    def semantic_available(self) -> bool:
        """Whether a semantic retriever has been configured."""
        return self._semantic_retriever is not None

    def keyword_search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[dict[str, object]]:
        """Search repository CodeUnits using BM25 keyword retrieval."""
        results = self._keyword_retriever.query(
            query,
            top_k=top_k,
        )

        return [
            self._make_evidence(
                unit=result.code_unit,
                evidence_id=f"keyword:{result.code_unit_id}",
                source="keyword_search",
                content=result.code_unit.source_code or "",
                score=result.score,
                metadata={
                    "rank": result.rank,
                    "file_path": result.file_path,
                    "start_line": result.start_line,
                    "end_line": result.end_line,
                },
            )
            for result in results
        ]

    def semantic_search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[dict[str, object]]:
        """Search repository CodeUnits using semantic embeddings."""
        if self._semantic_retriever is None:
            return []

        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError("top_k must be a positive integer.")

        results = self._semantic_retriever.query(
            query,
            top_k=top_k,
        )

        evidence: list[dict[str, object]] = []

        for result in results:
            if isinstance(result, SemanticSearchResult):
                unit = result.code_unit
                score = result.score
                rank = result.rank
            elif isinstance(result, CodeUnit):
                # Backward compatibility with simple injected retrievers.
                unit = result
                score = None
                rank = len(evidence) + 1
            else:
                raise TypeError(
                    "Semantic retriever must return CodeUnit instances "
                    "or SemanticSearchResult instances."
                )

            if unit.id not in self._code_units:
                continue

            if unit.id in {str(item["code_unit_id"]) for item in evidence}:
                continue

            evidence.append(
                self._make_evidence(
                    unit=unit,
                    evidence_id=f"semantic:{unit.id}",
                    source="semantic_search",
                    content=unit.source_code or "",
                    score=score,
                    metadata={
                        "rank": rank,
                        "file_path": unit.file_path,
                        "start_line": unit.start_line,
                        "end_line": unit.end_line,
                    },
                )
            )

            if len(evidence) >= top_k:
                break

        return evidence

    def graph_traversal(
        self,
        code_unit_ids: Iterable[str],
        *,
        relationships: Iterable[str] | None = None,
        direction: str = "both",
        max_depth: int = 2,
        top_k: int = 20,
    ) -> list[dict[str, object]]:
        """Discover related CodeUnits through repository graph traversal."""
        evidence = []
        included_ids: set[str] = set()

        for start_id in code_unit_ids:
            if start_id not in self._code_units:
                continue

            results = self._graph_query.traverse(
                start_id=start_id,
                relationships=relationships,
                direction=direction,
                max_depth=max_depth,
                strategy="bfs",
            )

            for result in results:
                related_id = result["code_unit_id"]

                if related_id in included_ids:
                    continue

                unit = self._code_graph.get_code_unit(related_id)

                if unit is None:
                    continue

                included_ids.add(related_id)

                evidence.append(
                    self._make_evidence(
                        unit=unit,
                        evidence_id=f"graph:{related_id}",
                        source="graph_traversal",
                        content=unit.source_code or "",
                        metadata={
                            "relationship": result["relationship"],
                            "parent_id": result["parent_id"],
                            "depth": result["depth"],
                            "file_path": unit.file_path,
                            "start_line": unit.start_line,
                            "end_line": unit.end_line,
                        },
                    )
                )

                if len(evidence) >= top_k:
                    return evidence

        return evidence

    def inspect_source(
        self,
        code_unit_ids: Iterable[str],
    ) -> list[dict[str, object]]:
        """Retrieve source code and docstrings for selected CodeUnits."""
        evidence = []

        for unit_id in dict.fromkeys(code_unit_ids):
            unit = self._code_graph.get_code_unit(unit_id)

            if unit is None:
                continue

            content = unit.source_code or ""

            evidence.append(
                self._make_evidence(
                    unit=unit,
                    evidence_id=f"source:{unit.id}",
                    source="source_inspection",
                    content=content,
                    metadata={
                        "docstring": unit.docstring,
                        "file_path": unit.file_path,
                        "start_line": unit.start_line,
                        "end_line": unit.end_line,
                        "qualified_name": unit.qualified_name,
                        "kind": unit.kind,
                    },
                )
            )

        return evidence

    @staticmethod
    def _make_evidence(
        *,
        unit: CodeUnit,
        evidence_id: str,
        source: str,
        content: str,
        metadata: dict[str, object],
        score: float | None = None,
    ) -> dict[str, object]:
        """Create a consistent evidence record for investigation results."""
        record: dict[str, object] = {
            "evidence_id": evidence_id,
            "source": source,
            "code_unit_id": unit.id,
            "name": unit.name,
            "qualified_name": unit.qualified_name,
            "kind": unit.kind,
            "repository": unit.repository,
            "file_path": unit.file_path,
            "start_line": unit.start_line,
            "end_line": unit.end_line,
            "content": content,
            "metadata": metadata,
        }

        if score is not None:
            record["score"] = score

        return record
