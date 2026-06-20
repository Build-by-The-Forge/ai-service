up:
	docker compose up --build

down:
	docker compose down

logs:
	docker compose logs -f

migrate:
	docker compose exec ai-service \
	alembic upgrade head

integration:
	pytest tests/integration -q
