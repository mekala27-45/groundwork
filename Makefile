.PHONY: dev test lint typecheck format check em-dash claims eval deploy-api deploy-web clean db-up db-down

dev:
	uv sync --all-packages --dev
	cd web && npm install

test:
	uv run pytest --cov --cov-report=term-missing

lint:
	uv run ruff check .

format:
	uv run ruff format .

typecheck:
	uv run mypy packages

em-dash:
	uv run python scripts/check_no_em_dash.py

claims:
	uv run python scripts/check_published_numbers.py

claims-write:
	uv run python scripts/check_published_numbers.py --write

eval:
	uv run python scripts/run_eval.py

check: lint typecheck em-dash test claims

db-up:
	service postgresql start || pg_ctlcluster 16 main start

db-down:
	pg_ctlcluster 16 main stop

migrate:
	cd packages/api && uv run alembic upgrade head

web-dev:
	cd web && npm run dev

web-build:
	cd web && npm run build

deploy-api:
	cd packages/api && fly deploy

deploy-web:
	@echo "No manual step needed: pushing to master with changes under web/ or .github/workflows/pages.yml triggers that workflow, which builds and publishes the static export to GitHub Pages on its own."

clean:
	find . -name "__pycache__" -not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .ruff_cache .mypy_cache .pytest_cache htmlcov .coverage
