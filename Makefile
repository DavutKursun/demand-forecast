.PHONY: setup test data data-synthetic train eval app
PY ?= python
setup:
	$(PY) -m pip install -r requirements-dev.txt
test:
	$(PY) -m pytest -q
data:            ## download Kaggle data and aggregate to weekly
	$(PY) scripts/download_data.py && $(PY) scripts/prepare.py
data-synthetic:  ## no Kaggle account needed (demo only)
	$(PY) scripts/prepare.py --synthetic
train:
	$(PY) scripts/train.py
eval:
	$(PY) scripts/eval.py
app:
	streamlit run app.py
