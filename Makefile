# AI Infra 从零构建 · 常用命令
PY := .venv/bin/python

.PHONY: venv build check test execute site pdf all clean

venv:
	UV_CACHE_DIR="$(CURDIR)/.uv-cache" uv venv .venv --python 3.13
	UV_CACHE_DIR="$(CURDIR)/.uv-cache" uv pip install --python $(PY) -e . -r requirements.txt

build:
	$(PY) tools/build_notebooks.py

check:
	$(PY) tools/check_notebooks.py --all

test:
	$(PY) tools/test_lib.py

execute:
	$(PY) tools/execute_notebooks.py --all --timeout 300

site:
	$(PY) tools/build_site.py --out _site

pdf:
	$(PY) tools/build_site.py --out _site --pdf dist/ai-infra-course-v0.1.0.pdf

all: build check test execute

clean:
	rm -rf outputs/*.png outputs/_preview __pycache__ aiinfra/__pycache__ tools/__pycache__
