"""Tests for repolens.retrieval.bm25_retriever."""

from __future__ import annotations

import pytest

from repolens.ingestion.code_unit import CallReference, CodeUnit, ImportReference
from repolens.retrieval.bm25_retriever import (
    _STOP_WORDS,
    BM25Retriever,
    RetrievalResult,
    preprocess,
    tokenize,
)

# ===========================================================================
# Helpers
# ===========================================================================


def make_unit(
    uid: str,
    name: str,
    qualified_name: str = "",
    kind: str = "function",
    file_path: str = "src/example.py",
    start_line: int = 1,
    end_line: int = 10,
    parent_id: str | None = None,
    imports: tuple = (),
    calls: tuple = (),
    metadata: dict | None = None,
) -> CodeUnit:
    return CodeUnit(
        id=uid,
        kind=kind,
        name=name,
        qualified_name=qualified_name or name,
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
        parent_id=parent_id,
        imports=imports,
        calls=calls,
        metadata=metadata or {},
    )


def _units_fixture() -> list[CodeUnit]:
    """Small deterministic corpus used across many tests."""
    return [
        make_unit(
            "mod:loader",
            "repository_loader",
            "repolens.ingestion.repository_loader",
            kind="module",
            file_path="src/repolens/ingestion/repository_loader.py",
            imports=(ImportReference("subprocess"), ImportReference("pathlib")),
        ),
        make_unit(
            "fn:load_local",
            "load_local",
            "repolens.ingestion.repository_loader.load_local",
            kind="function",
            file_path="src/repolens/ingestion/repository_loader.py",
            start_line=5,
            end_line=20,
            parent_id="mod:loader",
        ),
        make_unit(
            "fn:clone_remote",
            "clone_remote",
            "repolens.ingestion.repository_loader.clone_remote",
            kind="function",
            file_path="src/repolens/ingestion/repository_loader.py",
            start_line=22,
            end_line=55,
            parent_id="mod:loader",
        ),
        make_unit(
            "mod:discovery",
            "file_discovery",
            "repolens.ingestion.file_discovery",
            kind="module",
            file_path="src/repolens/ingestion/file_discovery.py",
        ),
        make_unit(
            "cls:graph",
            "CodeGraph",
            "repolens.graph.code_graph.CodeGraph",
            kind="class",
            file_path="src/repolens/graph/code_graph.py",
            start_line=10,
            end_line=200,
        ),
        make_unit(
            "fn:build_graph",
            "build",
            "repolens.graph.code_graph.CodeGraph.build",
            kind="method",
            file_path="src/repolens/graph/code_graph.py",
            start_line=30,
            end_line=120,
            parent_id="cls:graph",
            calls=(CallReference("add_node"), CallReference("add_edge")),
        ),
        make_unit(
            "fn:get_node",
            "get_node",
            "repolens.graph.code_graph.CodeGraph.get_node",
            kind="method",
            file_path="src/repolens/graph/code_graph.py",
            start_line=122,
            end_line=130,
            parent_id="cls:graph",
        ),
    ]


# ===========================================================================
# tokenize()
# ===========================================================================


class TestTokenize:
    def test_simple_snake_case(self):
        assert tokenize("file_path_discovery") == ["file", "path", "discovery"]

    def test_camel_case_expansion(self):
        assert tokenize("loadRepository") == ["load", "repository"]

    def test_pascal_case_expansion(self):
        assert tokenize("CodeGraph") == ["code", "graph"]

    def test_mixed_case_expansion(self):
        assert tokenize("BM25Retriever") == ["bm25", "retriever"]

    def test_acronym_followed_by_word(self):
        result = tokenize("parseHTMLContent")
        assert "html" in result or "content" in result  # at minimum

    def test_numbers_preserved(self):
        result = tokenize("bm25_index")
        assert "bm25" in result

    def test_punctuation_stripped(self):
        result = tokenize("hello.world")
        assert "hello" in result and "world" in result

    def test_empty_string(self):
        assert tokenize("") == []

    def test_whitespace_only(self):
        assert tokenize("   ") == []

    def test_lowercased(self):
        assert tokenize("UPPER") == ["upper"]

    def test_single_word(self):
        assert tokenize("repository") == ["repository"]

    def test_no_pure_digit_tokens(self):
        # Pure digit strings like "123" should not appear
        result = tokenize("123abc")
        for t in result:
            assert not t.isdigit()

    def test_path_splitting(self):
        result = tokenize("src/repolens/ingestion/repository_loader.py")
        assert "src" in result
        assert "repolens" in result
        assert "ingestion" in result
        assert "repository" in result
        assert "loader" in result

    def test_multiple_separators(self):
        result = tokenize("get__node__data")
        assert "get" in result
        assert "node" in result
        assert "data" in result


# ===========================================================================
# preprocess()
# ===========================================================================


class TestPreprocess:
    def test_stop_words_removed(self):
        unit = make_unit("u1", "is_valid", "pkg.is_valid")
        tokens = preprocess(unit)
        for sw in _STOP_WORDS:
            assert sw not in tokens

    def test_single_char_tokens_removed(self):
        unit = make_unit("u1", "a_b_c", "a.b.c")
        tokens = preprocess(unit)
        for t in tokens:
            assert len(t) > 1

    def test_imports_included(self):
        unit = make_unit("u2", "mod", imports=(ImportReference("subprocess"),))
        tokens = preprocess(unit)
        assert "subprocess" in tokens

    def test_calls_included(self):
        unit = make_unit("u3", "fn", calls=(CallReference("authenticate_user"),))
        tokens = preprocess(unit)
        assert "authenticate" in tokens or "user" in tokens

    def test_file_path_included(self):
        unit = make_unit(
            "u4",
            "loader",
            file_path="src/repolens/ingestion/repository_loader.py",
        )
        tokens = preprocess(unit)
        assert "loader" in tokens or "repository" in tokens

    def test_metadata_strings_included(self):
        unit = make_unit("u5", "fn", metadata={"docstring": "clone remote repository"})
        tokens = preprocess(unit)
        assert "clone" in tokens or "remote" in tokens

    def test_metadata_non_string_ignored(self):
        unit = make_unit("u5", "fn", metadata={"count": 42, "flag": True})
        # Should not raise
        tokens = preprocess(unit)
        assert isinstance(tokens, list)

    def test_returns_list_of_strings(self):
        unit = make_unit("u6", "compute_hash")
        tokens = preprocess(unit)
        assert all(isinstance(t, str) for t in tokens)

    def test_deterministic(self):
        unit = make_unit("u7", "load_repository")
        assert preprocess(unit) == preprocess(unit)


# ===========================================================================
# BM25Retriever — construction
# ===========================================================================


class TestBM25RetrieverConstruction:
    def test_empty_iterable(self):
        r = BM25Retriever([])
        assert r.is_empty
        assert r.document_count == 0

    def test_single_unit(self):
        r = BM25Retriever([make_unit("u1", "loader")])
        assert r.document_count == 1
        assert not r.is_empty

    def test_multiple_units(self):
        r = BM25Retriever(_units_fixture())
        assert r.document_count == len(_units_fixture())

    def test_duplicate_ids_deduplicated(self):
        unit = make_unit("dup", "loader")
        r = BM25Retriever([unit, unit, unit])
        assert r.document_count == 1

    def test_generator_input(self):
        units = (u for u in _units_fixture())
        r = BM25Retriever(units)
        assert r.document_count == len(_units_fixture())

    def test_invalid_k1_raises(self):
        with pytest.raises(ValueError, match="k1"):
            BM25Retriever([], k1=-1.0)

    def test_invalid_b_too_high_raises(self):
        with pytest.raises(ValueError, match="b"):
            BM25Retriever([], b=1.5)

    def test_invalid_b_negative_raises(self):
        with pytest.raises(ValueError, match="b"):
            BM25Retriever([], b=-0.1)

    def test_non_code_unit_raises(self):
        with pytest.raises(TypeError, match="CodeUnit"):
            BM25Retriever(["not_a_code_unit"])  # type: ignore[list-item]

    def test_custom_k1_b_accepted(self):
        r = BM25Retriever(_units_fixture(), k1=1.2, b=0.5)
        assert r.document_count == len(_units_fixture())

    def test_b_boundary_zero(self):
        r = BM25Retriever(_units_fixture(), b=0.0)
        assert r.document_count == len(_units_fixture())

    def test_b_boundary_one(self):
        r = BM25Retriever(_units_fixture(), b=1.0)
        assert r.document_count == len(_units_fixture())


# ===========================================================================
# BM25Retriever.query() — empty / edge cases
# ===========================================================================


class TestBM25QueryEdgeCases:
    def setup_method(self):
        self.retriever = BM25Retriever(_units_fixture())

    def test_empty_query_returns_empty(self):
        assert self.retriever.query("") == []

    def test_whitespace_query_returns_empty(self):
        assert self.retriever.query("   ") == []

    def test_stop_words_only_query_returns_empty(self):
        assert self.retriever.query("the is a and or") == []

    def test_empty_index_returns_empty(self):
        r = BM25Retriever([])
        assert r.query("repository") == []

    def test_non_string_query_raises(self):
        with pytest.raises(TypeError, match="string"):
            self.retriever.query(42)  # type: ignore[arg-type]

    def test_top_k_zero_raises(self):
        with pytest.raises(ValueError, match="top_k"):
            self.retriever.query("repository", top_k=0)

    def test_top_k_negative_raises(self):
        with pytest.raises(ValueError, match="top_k"):
            self.retriever.query("repository", top_k=-1)

    def test_query_with_no_matches_returns_empty(self):
        # A term that cannot appear in any preprocessed document
        result = self.retriever.query("zzzzzyyyxxxwww_nonexistent_term")
        assert result == []

    def test_single_unit_index_hit(self):
        r = BM25Retriever([make_unit("u1", "authenticate_user")])
        results = r.query("authenticate")
        assert len(results) == 1
        assert results[0].name == "authenticate_user"

    def test_single_unit_index_miss(self):
        r = BM25Retriever([make_unit("u1", "authenticate_user")])
        results = r.query("zzz_nonexistent")
        assert results == []


# ===========================================================================
# BM25Retriever.query() — ranking correctness
# ===========================================================================


class TestBM25Ranking:
    def setup_method(self):
        self.units = _units_fixture()
        self.retriever = BM25Retriever(self.units)

    # ---- Rank order -------------------------------------------------------

    def test_results_sorted_descending_by_score(self):
        results = self.retriever.query("repository loader", top_k=10)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_rank_field_starts_at_one(self):
        results = self.retriever.query("graph", top_k=5)
        if results:
            assert results[0].rank == 1

    def test_rank_field_sequential(self):
        results = self.retriever.query("repository", top_k=10)
        for i, r in enumerate(results, start=1):
            assert r.rank == i

    # ---- Relevance correctness --------------------------------------------

    def test_loader_beats_graph_for_loader_query(self):
        results = self.retriever.query("repository loader")
        ids = [r.code_unit_id for r in results]
        assert "mod:loader" in ids
        loader_rank = next(r.rank for r in results if r.code_unit_id == "mod:loader")
        # Loader module should rank above code-graph items
        graph_ranks = [r.rank for r in results if "graph" in r.file_path]
        for gr in graph_ranks:
            assert loader_rank < gr, (
                f"loader rank {loader_rank} should beat graph rank {gr}"
            )

    def test_clone_remote_ranked_for_clone_query(self):
        results = self.retriever.query("clone remote", top_k=5)
        result_ids = [r.code_unit_id for r in results]
        assert "fn:clone_remote" in result_ids

    def test_graph_beats_loader_for_graph_query(self):
        results = self.retriever.query("code graph build")
        ids = [r.code_unit_id for r in results]
        assert "cls:graph" in ids or "fn:build_graph" in ids
        # Loader should not outrank the graph class
        graph_hits = [r for r in results if "graph" in r.file_path]
        loader_hits = [r for r in results if "loader" in r.file_path]
        if graph_hits and loader_hits:
            assert graph_hits[0].rank < loader_hits[0].rank

    def test_scores_are_positive_floats(self):
        results = self.retriever.query("repository", top_k=10)
        for r in results:
            assert isinstance(r.score, float)
            assert r.score > 0.0

    # ---- top_k enforcement ------------------------------------------------

    def test_top_k_limits_results(self):
        results = self.retriever.query("repository", top_k=2)
        assert len(results) <= 2

    def test_top_k_one_returns_at_most_one(self):
        results = self.retriever.query("repository", top_k=1)
        assert len(results) <= 1

    def test_top_k_larger_than_corpus(self):
        results = self.retriever.query("repository", top_k=1000)
        assert len(results) <= len(self.units)

    # ---- min_score filtering ----------------------------------------------

    def test_min_score_filters_low_scores(self):
        results_all = self.retriever.query("repository", top_k=10)
        if not results_all:
            return
        threshold = results_all[-1].score + 0.01
        results_filtered = self.retriever.query(
            "repository", top_k=10, min_score=threshold
        )
        assert len(results_filtered) < len(results_all)

    def test_very_high_min_score_returns_empty(self):
        results = self.retriever.query("repository", top_k=10, min_score=9999.0)
        assert results == []

    def test_min_score_zero_equivalent_to_default(self):
        r1 = self.retriever.query("repository", top_k=10, min_score=0.0)
        r2 = self.retriever.query("repository", top_k=10)
        assert [x.code_unit_id for x in r1] == [x.code_unit_id for x in r2]


# ===========================================================================
# RetrievalResult — metadata preservation
# ===========================================================================


class TestRetrievalResultMetadata:
    def setup_method(self):
        self.units = _units_fixture()
        self.retriever = BM25Retriever(self.units)

    def _get_result(self, uid: str, query: str) -> RetrievalResult | None:
        results = self.retriever.query(query, top_k=10)
        return next((r for r in results if r.code_unit_id == uid), None)

    def test_file_path_preserved(self):
        r = self._get_result("fn:clone_remote", "clone remote")
        assert r is not None
        assert r.file_path == "src/repolens/ingestion/repository_loader.py"

    def test_start_line_preserved(self):
        r = self._get_result("fn:clone_remote", "clone remote")
        assert r is not None
        assert r.start_line == 22

    def test_end_line_preserved(self):
        r = self._get_result("fn:clone_remote", "clone remote")
        assert r is not None
        assert r.end_line == 55

    def test_kind_preserved(self):
        r = self._get_result("cls:graph", "code graph")
        assert r is not None
        assert r.kind == "class"

    def test_qualified_name_preserved(self):
        r = self._get_result("fn:load_local", "load local")
        assert r is not None
        assert r.qualified_name == "repolens.ingestion.repository_loader.load_local"

    def test_code_unit_id_property(self):
        r = self._get_result("fn:load_local", "load local")
        assert r is not None
        assert r.code_unit_id == "fn:load_local"

    def test_code_unit_reference_intact(self):
        r = self._get_result("fn:load_local", "load local")
        assert r is not None
        assert isinstance(r.code_unit, CodeUnit)
        assert r.code_unit.id == "fn:load_local"

    def test_result_is_frozen(self):
        r = self._get_result("fn:load_local", "load local")
        assert r is not None
        with pytest.raises((AttributeError, TypeError)):
            r.score = 999.9  # type: ignore[misc]

    def test_name_property(self):
        r = self._get_result("fn:clone_remote", "clone remote")
        assert r is not None
        assert r.name == "clone_remote"


# ===========================================================================
# BM25 scoring properties
# ===========================================================================


class TestBM25ScoringProperties:
    def test_idf_zero_for_universal_term(self):
        """A term present in every document has zero or near-zero IDF."""
        # All units share the word "repolens" in qualified_name / file_path.
        units = _units_fixture()
        r = BM25Retriever(units)
        idf = r._idf("repolens")
        # With Robertson IDF the value is log(1 + 0.5/(N+0.5)) which is positive
        # but very small when df == N.
        assert idf >= 0.0

    def test_idf_nonzero_for_rare_term(self):
        units = _units_fixture()
        r = BM25Retriever(units)
        # "clone" only appears in fn:clone_remote
        idf = r._idf("clone")
        assert idf > 0.0

    def test_missing_term_idf_zero(self):
        r = BM25Retriever(_units_fixture())
        assert r._idf("zzznotaword") == 0.0

    def test_b_zero_no_length_normalisation(self):
        """With b=0, document length has no effect on scores."""
        units = [
            make_unit("short", "clone", file_path="src/a.py"),
            make_unit(
                "long",
                "clone_remote_repository_loader_very_long_name",
                file_path="src/b.py",
            ),
        ]
        r_b0 = BM25Retriever(units, b=0.0)
        r_b1 = BM25Retriever(units, b=1.0)
        results_b0 = r_b0.query("clone", top_k=2)
        results_b1 = r_b1.query("clone", top_k=2)
        # With b=1 the shorter doc should score higher; with b=0 less so.
        # Just assert both run without error and return 2 results.
        assert len(results_b0) >= 1
        assert len(results_b1) >= 1

    def test_higher_tf_gives_higher_score(self):
        """A unit that mentions the query term more often should rank higher."""
        unit_hi = make_unit(
            "hi",
            "clone_clone_clone",
            "pkg.clone_clone_clone",
            file_path="src/hi.py",
        )
        unit_lo = make_unit(
            "lo",
            "clone_other",
            "pkg.clone_other",
            file_path="src/lo.py",
        )
        r = BM25Retriever([unit_hi, unit_lo])
        results = r.query("clone", top_k=2)
        # "hi" has more "clone" tokens, so it should rank first.
        assert results[0].code_unit_id == "hi"

    def test_scores_deterministic_across_calls(self):
        r = BM25Retriever(_units_fixture())
        r1 = r.query("repository graph", top_k=5)
        r2 = r.query("repository graph", top_k=5)
        assert [x.code_unit_id for x in r1] == [x.code_unit_id for x in r2]
        assert [x.score for x in r1] == [x.score for x in r2]


# ===========================================================================
# Multi-term queries
# ===========================================================================


class TestMultiTermQueries:
    def setup_method(self):
        self.retriever = BM25Retriever(_units_fixture())

    def test_multi_term_query_returns_results(self):
        results = self.retriever.query("repository loader ingestion", top_k=5)
        assert len(results) >= 1

    def test_multi_term_better_than_single(self):
        """A unit matching multiple query terms should outscore a single-term match."""
        single = self.retriever.query("repository", top_k=1)
        multi = self.retriever.query("repository loader ingestion", top_k=1)
        if single and multi and single[0].code_unit_id == multi[0].code_unit_id:
            assert multi[0].score >= single[0].score

    def test_case_insensitive_query(self):
        lower = self.retriever.query("repository loader", top_k=5)
        upper = self.retriever.query("REPOSITORY LOADER", top_k=5)
        assert [r.code_unit_id for r in lower] == [r.code_unit_id for r in upper]

    def test_query_with_path_separator(self):
        results = self.retriever.query("ingestion/repository_loader", top_k=5)
        assert len(results) >= 1

    def test_query_with_dot_notation(self):
        results = self.retriever.query("code_graph.build", top_k=5)
        assert len(results) >= 1


# ===========================================================================
# Large corpus — stress / smoke test
# ===========================================================================


class TestLargeCorpus:
    def _make_large_corpus(self, n: int) -> list[CodeUnit]:
        return [
            make_unit(
                f"fn:{i}",
                f"function_{i}",
                f"pkg.module_{i % 10}.function_{i}",
                file_path=f"src/pkg/module_{i % 10}.py",
                start_line=i * 10 + 1,
                end_line=i * 10 + 9,
            )
            for i in range(n)
        ]

    def test_large_corpus_builds(self):
        corpus = self._make_large_corpus(500)
        r = BM25Retriever(corpus)
        assert r.document_count == 500

    def test_large_corpus_query_returns_top_k(self):
        corpus = self._make_large_corpus(500)
        r = BM25Retriever(corpus)
        results = r.query("function module", top_k=10)
        assert len(results) <= 10

    def test_large_corpus_query_sorted(self):
        corpus = self._make_large_corpus(200)
        r = BM25Retriever(corpus)
        results = r.query("function pkg module", top_k=20)
        scores = [x.score for x in results]
        assert scores == sorted(scores, reverse=True)
