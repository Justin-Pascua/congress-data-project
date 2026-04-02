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