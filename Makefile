# Compose v2 plugin if present, else legacy docker-compose.
COMPOSE ?= $(shell docker compose version >/dev/null 2>&1 && echo "docker compose" || echo "docker-compose")
GIT_SHA ?= $(shell git rev-parse --short HEAD 2>/dev/null || echo local)
export GIT_SHA

.PHONY: install lint format test test-unit test-slow test-int dq data train api run down logs ci ask eval-assistant

install:            ## Install locked deps (incl. dev) into .venv
	uv sync --frozen

lint:               ## Static checks
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .
	uv run ruff check --fix .

test-unit:          ## Fast unit tests (milliseconds each), no infrastructure
	uv run pytest --cov=app --cov=pipeline --cov=dq --cov=ml --cov=assistant --cov-report=term-missing:skip-covered

test-slow:          ## Real-data reconciliation tests (a few seconds, no infrastructure)
	uv run pytest -m slow -v

test-int:           ## Needs the stack running (`make run`)
	uv run pytest -m integration -v

test: test-unit test-slow

dq:                 ## Standalone DQ check against the raw CSVs
	uv run python -m dq.dq_check --input data/raw --output data/reports/dq_report.json

data:               ## DQ gate + pipeline, locally
	PYTHON="uv run python" DQ_FAIL_ON=fail sh scripts/prepare_data.sh

train: data         ## Retrain + evaluate the delay model -> models/ (artefact + model card)
	uv run python -m ml.train --db data/warehouse/supply_chain.duckdb --out models

ask:                ## Ask the assistant (needs API up + GEMINI_API_KEY): make ask Q="..."
	uv run python -m assistant "$(Q)"

eval-assistant:     ## Live assistant eval, 10 questions (about 1 cent on Flash-Lite)
	uv run python -m assistant.eval

api: data           ## Run the API locally without Docker
	uv run python -m app

run:                ## Full stack in Docker, waits until healthy
	$(COMPOSE) up --build -d
	@for i in $$(seq 1 60); do curl -sf localhost:8000/health >/dev/null && break; sleep 1; done
	@curl -s localhost:8000/health; echo

down:
	$(COMPOSE) down --volumes

logs:
	$(COMPOSE) logs -f api

ci: lint test-unit test-slow run test-int down   ## Mirror of the CI pipeline, locally
