.PHONY: test debug

test:
	poetry run pytest -q $(or $(TESTS),tests/) $(ARGS)

debug:
	poetry run pytest -s $(or $(TESTS),tests/) $(ARGS)
