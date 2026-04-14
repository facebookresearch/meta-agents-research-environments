# GAIA2 CLI container build chain
#
# Usage:
#   make all              # build everything (gaia2-cli → all containers)
#   make gaia2-oc          # build OpenClaw image
#   make gaia2-hermes      # build just gaia2-hermes
#   make gaia2-oracle      # build just gaia2-oracle
#   make verify           # smoke-test all images
#   make verify-gaia2-oc   # smoke-test a single image
#   make test             # run gaia2-cli unit tests
#   make list             # list gaia2 images in podman store
#   make clean            # remove all built images

SHELL := /bin/bash

# ── Paths ──────────────────────────────────────────────────────────────
CLI_CONTAINERS := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))

# ── Image names ────────────────────────────────────────────────────────
GAIA2_CLI_IMAGE     := localhost/gaia2-cli:local
GAIA2_OC_IMAGE      := localhost/gaia2-oc:latest
GAIA2_HERMES_IMAGE  := localhost/gaia2-hermes:latest
GAIA2_ORACLE_IMAGE  := localhost/gaia2-oracle:latest

ALL_IMAGES := $(GAIA2_CLI_IMAGE) $(GAIA2_OC_IMAGE) \
              $(GAIA2_HERMES_IMAGE) $(GAIA2_ORACLE_IMAGE)

# ── Build config ───────────────────────────────────────────────────────
# Override PODMAN_BUILD / PODMAN_SAVE when your environment needs wrappers
# such as sudo or a custom launcher. BUILD_EXTRA_HOST is optional for custom
# host aliases during the build.
PODMAN_BUILD ?= podman build
PODMAN_SAVE  ?= podman save
BUILD_EXTRA_HOST ?=
HTTP_PROXY_VALUE  ?= $(or $(http_proxy),$(HTTP_PROXY))
HTTPS_PROXY_VALUE ?= $(or $(https_proxy),$(HTTPS_PROXY),$(HTTP_PROXY_VALUE))

# Common podman build flags
BUILD_FLAGS := --network=host $(BUILD_EXTRA_HOST)
PROXY_ARGS  := $(if $(HTTP_PROXY_VALUE),--build-arg http_proxy=$(HTTP_PROXY_VALUE) \
               --build-arg HTTP_PROXY=$(HTTP_PROXY_VALUE)) \
               $(if $(HTTPS_PROXY_VALUE),--build-arg https_proxy=$(HTTPS_PROXY_VALUE) \
               --build-arg HTTPS_PROXY=$(HTTPS_PROXY_VALUE))

GAIA2_CLI_ARGS := --build-arg GAIA2_CLI_IMAGE=localhost/gaia2-cli \
                 --build-arg GAIA2_CLI_VERSION=local

# ── Helper: build + copy to user store ─────────────────────────────────
define copy-to-user
	@echo "══════ Copying to user store ══════"
	$(PODMAN_SAVE) $(1) | podman load
	@echo ""
	@echo "════════════════════════════════════"
	@echo "✓ $(1) ready (user store)"
	@echo "════════════════════════════════════"
endef

# ── Targets ────────────────────────────────────────────────────────────

.PHONY: all gaia2-cli gaia2-oc gaia2-hermes gaia2-oracle \
        verify verify-gaia2-cli verify-gaia2-oc verify-gaia2-hermes verify-gaia2-oracle \
        test list stop clean help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

all: gaia2-oc gaia2-hermes gaia2-oracle ## Build all images

# ── Base image ─────────────────────────────────────────────────────────

gaia2-cli: ## Build the gaia2-cli base image
	@echo "══════ Building gaia2-cli ══════"
	cd $(CLI_CONTAINERS) && \
	$(PODMAN_BUILD) $(BUILD_FLAGS) $(PROXY_ARGS) \
		-f cli/Dockerfile \
		-t $(GAIA2_CLI_IMAGE) .
	$(call copy-to-user,$(GAIA2_CLI_IMAGE))

# ── Container images (all depend on gaia2-cli) ─────────────────────────

gaia2-oc: gaia2-cli ## Build gaia2-oc image
	@echo "══════ Building gaia2-oc ══════"
	cd $(CLI_CONTAINERS) && \
	$(PODMAN_BUILD) $(BUILD_FLAGS) $(PROXY_ARGS) $(GAIA2_CLI_ARGS) \
		-f containers/openclaw/Dockerfile \
		-t $(GAIA2_OC_IMAGE) .
	$(call copy-to-user,$(GAIA2_OC_IMAGE))

gaia2-hermes: gaia2-cli ## Build gaia2-hermes image
	@echo "══════ Building gaia2-hermes ══════"
	cd $(CLI_CONTAINERS) && \
	$(PODMAN_BUILD) $(BUILD_FLAGS) $(PROXY_ARGS) $(GAIA2_CLI_ARGS) \
		-f containers/hermes/Dockerfile \
		-t $(GAIA2_HERMES_IMAGE) .
	$(call copy-to-user,$(GAIA2_HERMES_IMAGE))

gaia2-oracle: gaia2-cli ## Build gaia2-oracle image
	@echo "══════ Building gaia2-oracle ══════"
	cd $(CLI_CONTAINERS) && \
	$(PODMAN_BUILD) $(BUILD_FLAGS) $(GAIA2_CLI_ARGS) \
		-f containers/oracle/Dockerfile \
		-t $(GAIA2_ORACLE_IMAGE) .
	$(call copy-to-user,$(GAIA2_ORACLE_IMAGE))

# ── Verify ─────────────────────────────────────────────────────────────

verify: verify-gaia2-cli verify-gaia2-oc verify-gaia2-hermes verify-gaia2-oracle ## Smoke-test all images

verify-gaia2-cli: ## Verify gaia2-cli base image
	@echo "══════ Verifying $(GAIA2_CLI_IMAGE) ══════"
	podman run --rm $(GAIA2_CLI_IMAGE) /usr/bin/bash -c ' \
		echo "--- GAIA2 CLI tools ---"; \
		for cmd in calendar contacts emails messages chats rent-a-flat city cabs shopping cloud-drive; do \
			command -v "$$cmd" > /dev/null 2>&1 || { echo "FAIL: $$cmd not found"; exit 1; }; \
			$$cmd --help > /dev/null 2>&1 || { echo "FAIL: $$cmd --help failed"; exit 1; }; \
			echo "  OK: $$cmd"; \
		done; \
		echo ""; \
		echo "--- Internal tools ---"; \
		for cmd in /usr/local/bin/gaia2-init /usr/local/bin/gaia2-eventd /usr/local/bin/gaia2-replay; do \
			command -v "$$cmd" > /dev/null 2>&1 || { echo "FAIL: $$cmd not found"; exit 1; }; \
			echo "  OK: $$cmd"; \
		done; \
		echo ""; \
		echo "All checks passed!" \
	'

verify-gaia2-oc: ## Verify gaia2-oc image
	@echo "══════ Verifying $(GAIA2_OC_IMAGE) ══════"
	podman run --rm $(GAIA2_OC_IMAGE) /usr/bin/bash -c ' \
		echo "--- GAIA2 CLI tools ---"; \
		for cmd in calendar contacts emails messages chats rent-a-flat city cabs shopping cloud-drive; do \
			command -v "$$cmd" > /dev/null 2>&1 || { echo "FAIL: $$cmd not found"; exit 1; }; \
			echo "  OK: $$cmd"; \
		done; \
		echo ""; \
		echo "--- Internal tools ---"; \
		for cmd in /usr/local/bin/gaia2-init /usr/local/bin/gaia2-eventd /usr/local/bin/gaia2-replay; do \
			command -v "$$cmd" > /dev/null 2>&1 || { echo "FAIL: $$cmd not found"; exit 1; }; \
			echo "  OK: $$cmd"; \
		done; \
		echo ""; \
		echo "All checks passed!" \
	'

verify-gaia2-hermes: ## Verify gaia2-hermes image
	@echo "══════ Verifying $(GAIA2_HERMES_IMAGE) ══════"
	podman run --rm $(GAIA2_HERMES_IMAGE) /usr/bin/bash -c ' \
		echo "--- GAIA2 CLI tools ---"; \
		for cmd in calendar contacts emails messages chats rent-a-flat city cabs shopping cloud-drive; do \
			command -v "$$cmd" > /dev/null 2>&1 || { echo "FAIL: $$cmd not found"; exit 1; }; \
			echo "  OK: $$cmd"; \
		done; \
		echo ""; \
		echo "All checks passed!" \
	'

verify-gaia2-oracle: ## Verify gaia2-oracle image
	@echo "══════ Verifying $(GAIA2_ORACLE_IMAGE) ══════"
	podman run --rm $(GAIA2_ORACLE_IMAGE) /usr/bin/bash -c ' \
		echo "--- GAIA2 CLI tools ---"; \
		for cmd in calendar contacts emails messages chats rent-a-flat city cabs shopping cloud-drive; do \
			command -v "$$cmd" > /dev/null 2>&1 || { echo "FAIL: $$cmd not found"; exit 1; }; \
			echo "  OK: $$cmd"; \
		done; \
		echo ""; \
		echo "All checks passed!" \
	'

# ── Test ───────────────────────────────────────────────────────────────

test: ## Run Gaia2 unit tests
	@echo "══════ Running Gaia2 unit tests ══════"
	cd $(CLI_CONTAINERS) && \
	PYTHONPATH=cli:runner:core:. python3 -m pytest cli/tests runner/tests -q && \
	PYTHONPATH=cli:runner:core:. python3 -m pytest \
		containers/openclaw/test_gaia2_adapter.py \
		containers/openclaw/test_openclaw_config.py \
		-q && \
	PYTHONPATH=cli:runner:core:. python3 -m pytest \
		containers/hermes/test_gaia2_adapter.py \
		containers/hermes/test_hermes_worker.py \
		-q && \
	node containers/openclaw/test_launch_request_utils.mjs

# ── Utilities ──────────────────────────────────────────────────────────

list: ## List gaia2 images in podman store
	@echo "── Images ──"
	@podman images --format "table {{.Repository}}:{{.Tag}}\t{{.Size}}\t{{.Created}}" | \
		grep -E 'gaia|REPOSITORY' || true

stop: ## Stop all containers and kill port listeners
	@echo "Stopping all containers..."
	-podman rm -af
	@echo "Killing port listeners..."
	-sudo fuser -k 18789/tcp 8090/tcp 2>/dev/null
	@echo "Done."

clean: ## Remove all built images
	@echo "Removing images..."
	-@for img in $(ALL_IMAGES); do podman rmi $$img 2>/dev/null; done
	@echo "Done."
