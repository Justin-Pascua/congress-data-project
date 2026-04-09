include .env

.PHONY: compose-up
compose-up:
	@docker compose up -d

.PHONY: compose-down
compose-down:
	@docker compose down -v

.PHONY: mlflow-server
mlflow-server:
	@mlflow server --backend-store-uri ${MLFLOW_TRACKING_URI} --port 5000

.PHONY: train
train:
	@python -m ml.main.train

.PHONY: eval
eval:
	@python -m ml.main.eval

.PHONY: model-api
model-api:
	@if [ "$(ENV)" = "container" ]; then \
		docker run -p 8000:8000 congress-data-project-model-api:1.0.0; \
	else \
		uvicorn api.main:app --reload; \
	fi
