CONFIG ?= configs/tiny.yaml
PY ?= python

.PHONY: install test smoke pipeline eval serve docker yambda

install:
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest

smoke:
	$(PY) -m encore pipeline --config configs/ci.yaml && $(PY) -m encore check --config configs/ci.yaml

pipeline:
	$(PY) -m encore pipeline --config $(CONFIG)

eval:
	$(PY) -m encore eval --config $(CONFIG)

serve:
	$(PY) -m encore serve --config $(CONFIG)

docker:
	docker build -t encore .

yambda:
	$(PY) -m encore download --config configs/yambda50m.yaml
	$(PY) -m encore pipeline --config configs/yambda50m.yaml
