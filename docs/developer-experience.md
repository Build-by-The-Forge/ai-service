# Developer Experience

This document captures the local developer workflow, tooling, and quality gates for the `ai-service` repository.

## Local Environment Setup

1. Create or activate the virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Linux/macOS
   .venv\Scripts\activate    # Windows PowerShell
   ```

2. Install runtime and dev dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Verify the environment:
   ```bash
   python --version
   python -m pip show fastapi ruff black mypy pre-commit
   ```

## Makefile Targets

The repository exposes developer workflows through `Makefile` targets.

- `make lint` - run `ruff`, `black --check`, and `mypy app`
- `make format` - format Python code with `black`
- `make precommit` - install hooks and run them against all files
- `make test` - run unit, integration, and contract tests
- `make coverage` - run the test suite with coverage reporting
- `make up` / `make down` - start and stop Docker Compose services
- `make migrate` - apply Alembic migrations inside the container
- `make seed` - seed the database with sample data

> Windows note: If `make` is not installed on your machine, use the equivalent PowerShell commands with the repository virtual environment.
>
> ```powershell
> .\.venv\Scripts\activate.ps1
> .\.venv\Scripts\python.exe -m pre_commit install
> .\.venv\Scripts\python.exe -m pre_commit run --all-files
> .\.venv\Scripts\python.exe -m pytest tests/unit tests/integration tests/contracts -q
> .\.venv\Scripts\python.exe -m pytest tests/unit tests/integration tests/contracts --cov=app --cov-report=term-missing
> ```

## Pre-commit Hooks

The repository uses pre-commit to enforce the following checks on every commit:

- `ruff --fix`
- `black`
- `mypy` (using `mypy.ini`)

Install the hooks once with:

```bash
make precommit
```

Run the hooks manually at any time with:

```bash
.venv\Scripts\python.exe -m pre_commit run --all-files
```

## Quality Gates

The following quality checks should be run before opening a PR:

```bash
make lint
make test
make coverage
```

If `make coverage` is too slow during local development, run the targeted suites first:

```bash
.venv\Scripts\python.exe -m pytest tests/unit -q
.venv\Scripts\python.exe -m pytest tests/integration -q
```

## CI Workflow Alignment

The local workflow is intentionally aligned with CI. The same checks should be exercised both locally and in GitHub Actions:

- `make lint` runs `ruff`, `black --check`, and `mypy app`
- `make precommit` installs hooks and runs `pre_commit` across all files
- `make test` runs unit, integration, and contract tests
- `make coverage` runs the same tests with coverage reporting

On GitHub Actions, the pipeline runs these checks in the same order and adds security scanning and Docker build validation.

## Recommended Workflow

1. Pull the latest branch and update dependencies.
2. Run `make lint` and fix any issues.
3. Run `make test` to confirm behavior.
4. Run `make coverage` before merging if there are significant code changes.
5. Use `make precommit` to ensure commit-time checks are installed.

## Notes

- The project is intentionally configured to use pinned versions in `pyproject.toml`, `requirements.txt`, and `.pre-commit-config.yaml`.
- `mypy.ini` ignores missing imports for `jose` and `passlib` while the repo is still stabilizing.
