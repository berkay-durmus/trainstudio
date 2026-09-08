# TrainStudio — convenience targets around docker compose.
# Everything here is a thin wrapper; `docker compose ...` works just as well.

COMPOSE ?= docker compose
SERVICE ?= trainstudio

.DEFAULT_GOAL := help
.PHONY: help build up down restart logs shell gpu-check ps clean prune \
        proxy-up dummy-data

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

build: ## Build the image
	$(COMPOSE) build

up: ## Start in the background
	$(COMPOSE) up -d
	@v() { grep -E "^$$1=" .env 2>/dev/null | tail -1 | cut -d= -f2- ; }; \
	echo "TrainStudio (this machine):  http://localhost:$$(v HOST_PORT)"; \
	case "$$(v COMPOSE_PROFILES)" in \
	  *proxy*) echo "Shared access (sign in):     http://$$(v PUBLIC_HOSTNAME):$$(v PROXY_PORT)  ·  user: $$(v BASIC_AUTH_USER)";; \
	  *) echo "Shared access:               off — set COMPOSE_PROFILES=proxy in .env";; \
	esac

down: ## Stop and remove the containers
	$(COMPOSE) down

restart: ## Restart the application
	$(COMPOSE) restart $(SERVICE)

logs: ## Follow the logs
	$(COMPOSE) logs -f $(SERVICE)

ps: ## Show container status
	$(COMPOSE) ps

shell: ## Open a shell inside the running container
	$(COMPOSE) exec $(SERVICE) bash

gpu-check: ## Verify that the GPU is visible from inside the container
	$(COMPOSE) run --rm $(SERVICE) python -c "\
import torch, core.hardware as hw; \
print('torch', torch.__version__, '· cuda', torch.version.cuda); \
print('available:', torch.cuda.is_available()); \
print(hw.detect().label)"

dummy-data: ## Generate synthetic datasets into the mounted data volume
	$(COMPOSE) run --rm $(SERVICE) \
		python scripts/make_dummy_dataset.py --out "$${TRAINSTUDIO_RUNS_DIR%/*}/_synthetic" --kinds cls seg

proxy-up: ## Start with the basic-auth reverse proxy in front
	$(COMPOSE) --profile proxy up -d
	@echo "TrainStudio is at http://localhost:$${PROXY_PORT:-8080} (basic auth)"

clean: ## Remove the containers and the named volumes (caches are lost)
	$(COMPOSE) down -v

prune: ## Remove dangling build layers
	docker image prune -f
