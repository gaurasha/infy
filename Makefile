# infy — a local inference engine in Rust.
#
# `make check` is what CI runs and what you run before every commit: build,
# tests, lint, format, and the two architecture rules from GUIDELINES.md
# (no domain imports a sibling; every domain is portable on its own).
#
# Every target is plain cargo underneath, so `cargo test -p infy-engine` and
# friends work too. The Makefile exists so the architecture checks are one
# command away and nobody has to remember the script paths.

CARGO ?= cargo
BIN   := infy

.DEFAULT_GOAL := help

##@ Everything

.PHONY: help
help: ## show this help
	@awk 'BEGIN {FS = ":.*##"; printf "\ninfy — make targets\n"} \
	  /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } \
	  /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)
	@printf "\n"

.PHONY: check
check: build test lint fmt-check check-arch check-portable ## everything CI runs

.PHONY: build
build: ## compile every crate
	$(CARGO) build --workspace --all-targets

.PHONY: release
release: ## optimised binary at target/release/infy
	$(CARGO) build --release -p $(BIN)

.PHONY: test
test: ## run every test
	$(CARGO) test --workspace

.PHONY: lint
lint: ## clippy, warnings are errors
	$(CARGO) clippy --workspace --all-targets -- -D warnings

.PHONY: fmt
fmt: ## format all code
	$(CARGO) fmt --all

.PHONY: fmt-check
fmt-check: ## fail if anything is unformatted
	$(CARGO) fmt --all -- --check

.PHONY: clean
clean: ## remove build output
	$(CARGO) clean

##@ Architecture rules (GUIDELINES.md)

.PHONY: check-arch
check-arch: ## fail if a domain depends on a sibling, or a leaf grows a dependency
	./scripts/check-arch.sh

.PHONY: check-portable
check-portable: ## fail if a domain's tests do not pass with only kernel + wire beside it
	./scripts/check-portable.sh

##@ Run

.PHONY: run
run: ## start the OpenAI-compatible server (pass ARGS="--model ...")
	$(CARGO) run -p $(BIN) -- serve $(ARGS)

.PHONY: chat
chat: ## interactive chat (pass ARGS="--model ...")
	$(CARGO) run -p $(BIN) -- chat $(ARGS)
