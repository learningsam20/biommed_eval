# Makefile — BioMed Hybrid Search (Pinecone-first)
.PHONY: help install backend-install frontend-install dev backend-dev frontend-dev restart build test eval-setup eval-run eval-judge eval-report index clean

help:
	@echo "Setup:"; echo "  make install          - backend + frontend deps"
	@echo "Development:"; echo "  make backend-dev      - FastAPI on BACKEND_PORT"
	@echo "  make frontend-dev     - Vite on FRONTEND_URL port"
	@echo "  make restart          - kill those ports and start both (scripts/restart.sh)"
	@echo "Indexing:"; echo "  make index            - build BM25 + Pinecone index"
	@echo "Evaluation:"; echo "  make eval-setup       - frozen 100-query set"
	@echo "  make eval-run         - run all configs"; echo "  make eval-judge       - LLM-judge (JUDGE_FRAMEWORK in .env)"
	@echo "  make eval-report      - markdown report + best config + README findings"
	@echo "Maintenance:"; echo "  make test | make clean"

install: backend-install frontend-install
backend-install:
	cd backend && pip install -r requirements.txt
frontend-install:
	cd frontend && npm install

dev:
	$(MAKE) backend-dev & $(MAKE) frontend-dev

backend-dev:
	cd backend && python -m app.main

frontend-dev:
	cd frontend && npm run dev

restart:
	./scripts/restart.sh

build:
	cd frontend && npm run build

index:
	cd backend && python -m scripts.build_index

eval-setup:
	cd backend && python -m scripts.create_eval_set --output data/eval_queries_100.json --seed 42 --size 100

eval-run:
	cd backend && python -m scripts.run_evaluation --config all --eval-set data/eval_queries_100.json --output results

eval-judge:
	cd backend && python -m scripts.run_evaluation --config lexical,hybrid_weighted,hybrid_expansion --judge --limit 20 --eval-set data/eval_queries_100.json --output results

eval-report:
	cd backend && python -m scripts.generate_report --results-dir results --output docs/evaluation_report.md

test:
	cd backend && python -m pytest tests/ -v

clean:
	rm -rf frontend/dist backend/app/__pycache__ backend/scripts/__pycache__ backend/.pytest_cache
