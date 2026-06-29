.PHONY: up down logs lint test migrate

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f

lint:
	black .
	flake8 .
	mypy .

test:
	pytest

migrate:
	alembic upgrade head
