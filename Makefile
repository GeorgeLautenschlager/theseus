.PHONY: test debug release

test:
	poetry run pytest -q $(or $(TESTS),tests/) $(ARGS)

debug:
	poetry run pytest -s $(or $(TESTS),tests/) $(ARGS)

# Cut a release: verify, bump the version, commit, tag, and push (current branch + tag).
# Run it from an up-to-date main.
#   make release VERSION=0.4.3
#   make release VERSION=0.4.3 SKIP_TESTS=1   # skip the offline test gate
release:
	@test -n "$(VERSION)" || { echo "Usage: make release VERSION=X.Y.Z"; exit 1; }
	@echo "$(VERSION)" | grep -Eq '^[0-9]+\.[0-9]+\.[0-9]+$$' || { echo "VERSION must be X.Y.Z (got '$(VERSION)')"; exit 1; }
	@test -z "$$(git status --porcelain)" || { echo "Working tree is dirty -- commit or stash first."; exit 1; }
	@if git rev-parse -q --verify "refs/tags/v$(VERSION)" >/dev/null; then echo "Tag v$(VERSION) already exists."; exit 1; fi
	@if [ -z "$(SKIP_TESTS)" ]; then poetry run pytest -q tests/ --ignore=tests/test_fact_retention.py; else echo "Skipping tests (SKIP_TESTS set)."; fi
	poetry version $(VERSION)
	git add pyproject.toml
	git commit -m "Release $(VERSION)"
	git tag -a "v$(VERSION)" -m "Release $(VERSION)"
	git push origin HEAD "v$(VERSION)"
	@echo "Released v$(VERSION) and pushed (branch + tag)."
