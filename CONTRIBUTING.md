# Contributing to RepoLens

This guide defines the **branching model, Pull Request process, and review rules** for all four RepoLens team members. Follow it on every task.

---

## Team

| GitHub Handle | Role |
|---|---|
| [@kravishrao](https://github.com/kravishrao) | Project Lead / Admin |
| [@Manoj-kanur](https://github.com/Manoj-kanur) | Developer |
| [@BmHarshithKumar](https://github.com/BmHarshithKumar) | Developer |
| [@mohit2007malur-png](https://github.com/mohit2007malur-png) | Developer |

---

## Branch Model

```
main          ← stable release snapshots only (tagged)
  └── develop ← team integration branch  ← all PRs target here
        ├── feature/RL-XX-short-description
        ├── fix/RL-XX-short-description
        ├── docs/RL-XX-short-description
        └── ci/RL-XX-short-description
```

### Protected Branches

| Branch | Rule |
|--------|------|
| `main` | No direct pushes. No force-pushes. No deletion. PRs from `develop` only, by project lead. |
| `develop` | No direct pushes. No force-pushes. No deletion. All PRs require ≥ 1 approval + CI green. |

---

## Branch Naming Convention

```
<prefix>/RL-<issue-number>-<short-description>
```

| Prefix | Use for |
|--------|---------|
| `feature/` | New functionality tied to a Jira story |
| `fix/` | Bug fixes |
| `docs/` | Documentation-only changes |
| `ci/` | CI workflow or tooling changes |
| `refactor/` | Internal restructuring with no behaviour change |
| `chore/` | Dependency bumps, config, project maintenance |

**Examples:**

```bash
feature/RL-15-repository-loading
fix/RL-42-ssh-url-validation
docs/RL-50-contributing-guide
ci/RL-11-develop-only-triggers
```

> **Rule:** Never work directly on `develop` or `main`. Always create a branch.

---

## Daily Workflow (Step-by-Step)

### 1 — Start a task

```bash
# Always branch from develop
git checkout develop
git pull origin develop
git checkout -b feature/RL-XX-short-description
```

### 2 — Commit your changes

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <short description>

[optional body]
```

| Type | When to use |
|------|------------|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation |
| `ci` | CI / workflow |
| `refactor` | Refactoring |
| `test` | Tests only |
| `chore` | Tooling / config |

**Examples:**

```bash
git commit -m "feat(ingestion): implement Python source file discovery"
git commit -m "fix(graph): rename unused loop variable to satisfy B007"
git commit -m "ci: restrict CI triggers to develop branch only"
```

### 3 — Push and open a PR

```bash
git push origin feature/RL-XX-short-description
```

Then open a Pull Request on GitHub:
- **Base branch:** `develop`
- **Title:** matches your Jira issue title
- **Fill in the PR template completely** — especially the Jira link

### 4 — Request review

Assign at least one teammate as reviewer. Code owners (`.github/CODEOWNERS`) are added automatically based on which files you changed.

### 5 — Address review feedback

```bash
# Make requested changes, then:
git add .
git commit -m "fix(scope): address review feedback"
git push origin feature/RL-XX-short-description
```

> Every new commit re-triggers CI and notifies reviewers automatically.

### 6 — Merge

Once **all** of the following are true, the PR can be merged:

- [ ] CI passes: **Code Quality Checks** ✅ and **Test Suite** ✅
- [ ] At least **1 approving review** from a teammate
- [ ] All review **conversations are resolved**
- [ ] Branch is **up to date** with `develop`

Use **"Create a merge commit"** (not squash/rebase) to preserve history.

### 7 — Clean up

```bash
# Delete your local branch after merge
git checkout develop
git pull origin develop
git branch -d feature/RL-XX-short-description
```

The remote branch is deleted automatically on merge (GitHub setting).

---

## Pull Request Rules

### Required before opening a PR

- Branch name follows the naming convention above
- PR template is fully filled in
- Jira issue is linked (`Closes RL-XX`)
- Self-review completed (read your own diff)
- Local CI passes: `ruff format --check src tests && ruff check src tests && pytest tests/ -q`

### Required before merging

| Requirement | Enforced by |
|------------|-------------|
| ≥ 1 approving review | Branch protection |
| All conversations resolved | Branch protection |
| CI: Code Quality Checks passed | Required status check |
| CI: Test Suite passed | Required status check |
| No direct push to `develop` | Branch protection |
| No force-push | Branch protection |

### Review etiquette

- **Reviewer:** respond within **1 working day**
- **Author:** respond to comments within **1 working day** after re-push
- Use **"Request changes"** for blockers, **"Comment"** for suggestions
- Approve only when you are genuinely satisfied — not as a courtesy

---

## Jira ↔ GitHub Linking

Every PR **must** reference its Jira issue:

```markdown
Closes [RL-XX](https://repolens.atlassian.net/browse/RL-XX)
```

Include the Jira key in:
1. The branch name: `feature/RL-XX-...`
2. The PR body (via the template)
3. Commit messages where relevant: `feat(scope): description (RL-XX)`

---

## CI Checks

The CI workflow (`.github/workflows/ci.yml`) runs automatically on every PR targeting `develop` and on every push to `develop`.

| Job | What it checks |
|-----|---------------|
| **Code Quality Checks** | `ruff format --check src tests` + `ruff check src tests` |
| **Test Suite** | `pytest tests/ --tb=short --strict-markers -q` |

### Running CI locally

```bash
# Install dev tools
pip install -e ".[dev]"

# Format check
ruff format --check src tests

# Lint
ruff check src tests

# Tests
pytest tests/ -q
```

Fix all issues before pushing. A failing CI blocks the PR from merging.

---

## Branch Protection Summary

> **Note for @kravishrao:** Configure these rules in GitHub → Settings → Branches after merging this PR.

### `develop` branch rules

```
✅ Require a pull request before merging
✅ Required approving reviews: 1
✅ Dismiss stale reviews when new commits are pushed
✅ Require review from Code Owners
✅ Require status checks to pass before merging:
      - Code Quality Checks
      - Test Suite
✅ Require branches to be up to date before merging
✅ Require conversation resolution before merging
✅ Restrict who can push to matching branches (team only)
❌ Allow force pushes: OFF
❌ Allow deletions: OFF
```

### `main` branch rules

```
✅ Require a pull request before merging
✅ Required approving reviews: 1 (project lead only)
✅ Dismiss stale reviews when new commits are pushed
✅ Require status checks to pass
✅ Require conversation resolution before merging
❌ Allow force pushes: OFF
❌ Allow deletions: OFF
```

---

## Merge Strategy

| Situation | Strategy |
|-----------|----------|
| Feature / fix PR → `develop` | **Merge commit** (`--no-ff`) |
| `develop` → `main` (release) | **Merge commit** with tag |

Do **not** use squash or rebase merges — they rewrite history and make it harder to trace which commits belong to which Jira issue.

---

## Quick-Reference Cheatsheet

```bash
# Start work
git checkout develop && git pull origin develop
git checkout -b feature/RL-XX-description

# During work
git add . && git commit -m "feat(scope): description"

# Before pushing
ruff format --check src tests
ruff check src tests
pytest tests/ -q

# Push and open PR
git push origin feature/RL-XX-description
# → open PR on GitHub targeting develop
# → fill in PR template, link Jira issue
# → assign a reviewer

# After PR is merged
git checkout develop && git pull origin develop
git branch -d feature/RL-XX-description
```
