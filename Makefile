.PHONY: test debug

test:
	python3 -m pytest -q $(or $(TESTS),tests/) $(ARGS)

debug:
	python3 -m pytest -s $(or $(TESTS),tests/) $(ARGS)
