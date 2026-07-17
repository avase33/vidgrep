.PHONY: up down test test-rust test-python test-go build-web ml-demo verify

up:
	docker compose up --build

down:
	docker compose down

test: test-rust test-python test-go

test-rust:
	cd processor-rust && cargo test

test-python:
	cd ml-pipeline-python && pip install -e ".[dev]" && pytest -q

test-go:
	cd gateway-go && go test ./...

build-web:
	cd frontend-ts && npm install && npm run build

# Offline: index a synthetic video and run text searches (no services needed).
ml-demo:
	cd ml-pipeline-python && python -m vidgrep_ml.cli demo

verify:
	python scripts/verify.py
