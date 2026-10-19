# ragkb 常用命令（Linux / macOS）
# 用法：make demo / make ask q="试用期最长可以多久？" / make eval / make test

PY ?= python3
Q  ?= 试用期最长可以多久？

.PHONY: help demo ingest ask eval eval-retrieval serve stats check test smoke sweep lint clean

help:
	@echo "make demo                       入库示例语料"
	@echo "make ingest path=data/corpus    入库指定路径"
	@echo "make ask q=\"你的问题\"            提问（含召回明细）"
	@echo "make eval                       评测 + CI 门禁"
	@echo "make eval-retrieval             只测检索（不调生成模型）"
	@echo "make serve                      启动 HTTP 服务与 Web UI"
	@echo "make stats                      语料 + 线上指标"
	@echo "make check                      体检"
	@echo "make test                       运行单元测试"
	@echo "make smoke                      端到端冒烟（离线）"
	@echo "make sweep                      拒答阈值扫描"
	@echo "make lint                        ruff 静态检查"

demo:
	$(PY) -m ragkb.cli demo

path ?= data/corpus
ingest:
	$(PY) -m ragkb.cli ingest $(path) --acl public

ask:
	$(PY) -m ragkb.cli ask "$(Q)" --show-context

eval:
	$(PY) -m ragkb.cli eval

eval-retrieval:
	$(PY) -m ragkb.cli eval --retrieval-only

serve:
	$(PY) -m ragkb.cli serve

stats:
	$(PY) -m ragkb.cli stats --online

check:
	$(PY) -m ragkb.cli check

test:
	$(PY) -m unittest discover -s tests -t . -v

smoke:
	$(PY) scripts/smoke.py

sweep:
	$(PY) scripts/sweep_refusal.py

lint:
	$(PY) -m ruff check ragkb tests scripts

clean:
	rm -rf data/_tests data/_smoke data/*.sqlite3 data/traces.jsonl data/feedback.jsonl eval/reports
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
