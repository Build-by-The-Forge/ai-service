up:
	docker compose up --build

down:
	docker compose down

logs:
	docker compose logs -f

migrate:
	docker compose exec ai-service \
	alembic upgrade head

PYTHON=.venv/Scripts/python.exe

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m black --check .
	$(PYTHON) -m mypy app

format:
	$(PYTHON) -m black .

precommit:
	$(PYTHON) -m pre_commit install
	$(PYTHON) -m pre_commit run --all-files

test:
	$(PYTHON) -m pytest tests/unit tests/integration tests/contracts -q

coverage:
	$(PYTHON) -m pytest tests/unit tests/integration tests/contracts --cov=app --cov-report=term-missing

ci:
	make lint
	make test
	make coverage

integration:
	pytest tests/integration -q
