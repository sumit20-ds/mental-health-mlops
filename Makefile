PY ?= python
export PYTHONPATH := $(CURDIR)

.PHONY: install lint test train mlflow-ui serve ui up down train-docker logs drift-demo package deploy invoke teardown

install:            ## create venv + install everything
	$(PY) -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -r requirements-dev.txt

lint:
	ruff check .

test:
	pytest -q

train:              ## train 3 models, log to MLflow, register + promote champion, export ./models
	$(PY) -m src.train

mlflow-ui:          ## browse experiments & registry at http://localhost:5000 (local sqlite store)
	mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

serve:              ## FastAPI on :8000 (docs at /docs)
	MODEL_DIR=models PORT=8000 uvicorn serving.app:app --reload --port 8000

ui:                 ## Streamlit on :8501 (expects API on :8000)
	API_URL=http://localhost:8000 streamlit run ui/app.py

up:                 ## full stack in Docker: MLflow, API, UI, Prometheus, Grafana
	docker compose up --build -d
	@echo "UI http://localhost:8501 | API http://localhost:8000/docs | MLflow http://localhost:5000 | Prometheus :9090 | Grafana :3000"

down:
	docker compose down

train-docker:       ## train inside Docker against the compose MLflow server, then restart the API to load it
	docker compose up -d mlflow
	docker compose --profile train run --rm --build trainer
	docker compose restart api

logs:
	docker compose logs -f api

drift-demo:         ## push normal, then drifted traffic and print the drift verdict
	curl -s -X POST localhost:8000/monitoring/simulate -H 'Content-Type: application/json' -d '{"n":200,"drift":false}' && echo
	curl -s localhost:8000/monitoring/drift | $(PY) -c "import sys,json; print('normal ->', json.load(sys.stdin)['overall'])"
	curl -s -X POST localhost:8000/monitoring/simulate -H 'Content-Type: application/json' -d '{"n":400,"drift":true}' && echo
	curl -s localhost:8000/monitoring/drift | $(PY) -c "import sys,json; print('after drift ->', json.load(sys.stdin)['overall'])"

package:
	$(PY) deploy/sagemaker/package_model.py

# The next three need: ENDPOINT, IMAGE_URI, MODEL_DATA, ROLE_ARN in the environment (see README)
deploy:
	$(PY) deploy/sagemaker/deploy.py --endpoint-name $(ENDPOINT) --image-uri $(IMAGE_URI) --model-data $(MODEL_DATA) --role-arn $(ROLE_ARN)

invoke:
	$(PY) deploy/sagemaker/invoke.py --endpoint-name $(ENDPOINT)

teardown:
	$(PY) deploy/sagemaker/teardown.py --endpoint-name $(ENDPOINT)
