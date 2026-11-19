from __future__ import annotations

from pathlib import Path

from repolens.evaluation.documentation import (
    generate_ablation_documentation,
    get_baseline_limitations,
)
from repolens.evaluation.models import (
    BaselineType,
    BenchmarkQuery,
    ExperimentConfig,
)
from repolens.evaluation.runner import (
    ExperimentRunner,
    get_git_commit_hash,
    get_system_provenance,
)


def test_system_provenance() -> None:
    """Verify capturing execution environment provenance."""
    prov = get_system_provenance(seed=123)
    assert "platform" in prov
    assert "python_version" in prov
    assert prov["random_seed"] == 123
    assert "git_commit" in prov


def test_git_commit_hash() -> None:
    """Verify git commit retrieval."""
    commit = get_git_commit_hash()
    assert isinstance(commit, str)
    assert len(commit) > 0


def test_get_baseline_limitations() -> None:
    """Verify retrieving limitations for each baseline."""
    lim_b0 = get_baseline_limitations("B0: Naive RAG")
    assert any("Syntactic boundaries" in item for item in lim_b0)

    lim_b5 = get_baseline_limitations("B5: Full RepoLens")
    assert any("filesystem" in item.lower() for item in lim_b5)


def test_generate_ablation_documentation() -> None:
    """Verify complete documentation markdown generation."""
    doc = generate_ablation_documentation()
    assert "# RepoLens Baseline & Ablation Experimental Architecture" in doc
    assert "Naive RAG" in doc
    assert "Full RepoLens" in doc


def test_experiment_runner_save_and_load(tmp_path: Path) -> None:
    """Verify ExperimentRunner runs experiment, saves reproducible JSON, and loads it."""
    # Create mini repo
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    (repo_dir / "sample.py").write_text("def ping(): return 'pong'\n", encoding="utf-8")

    out_dir = tmp_path / "results"
    runner = ExperimentRunner(output_dir=out_dir)

    cfg = ExperimentConfig(
        baseline=BaselineType.B0_NAIVE_RAG,
        repository_name="MiniRepo",
        repository_path=repo_dir,
        top_k=2,
    )
    queries = [
        BenchmarkQuery(
            query_id="q1",
            question="Where is ping?",
            repository="MiniRepo",
            gold_symbols=["ping"],
        )
    ]

    run_res = runner.run_experiment(config=cfg, queries=queries)
    assert run_res.experiment_id is not None
    assert len(run_res.query_results) == 1
    assert run_res.summary_metrics["mean_hit_rate"] >= 0.0

    saved_file = runner.save_run_result(run_res)
    assert saved_file.exists()

    loaded = runner.load_run_result(saved_file)
    assert loaded["experiment_id"] == run_res.experiment_id
    assert loaded["config"]["baseline"] == BaselineType.B0_NAIVE_RAG.value
