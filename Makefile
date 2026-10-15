# Convenience targets (macOS / Linux). Windows users: run the commands from the README directly.
PYTHON ?= python

.PHONY: install bootstrap api ui test test-fast lint demo samples backup benchmark docker-up docker-down

install:
	$(PYTHON) -m pip install -r requirements-dev.txt

bootstrap:
	$(PYTHON) scripts/bootstrap.py

api:
	uvicorn railsentinel.api.main:app --port 8000

ui:
	streamlit run ui/Home.py

test:
	$(PYTHON) -m pytest

test-fast:
	$(PYTHON) -m pytest -m "not slow"

lint:
	$(PYTHON) -m pyflakes railsentinel ui scripts tests

demo:
	$(PYTHON) scripts/run_scenario.py fog_surge

samples:
	$(PYTHON) scripts/export_samples.py

backup:
	$(PYTHON) scripts/backup.py create

benchmark:
	$(PYTHON) scripts/benchmark.py

docker-up:
	docker compose up --build

docker-down:
	docker compose down
