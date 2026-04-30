.PHONY: install train api test clean

install:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt

train:
	.venv/bin/python -m backend.models.train

api:
	.venv/bin/uvicorn backend.api.main:app --reload --port 8000

test:
	.venv/bin/python -m pytest tests/ -v

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf artifacts/*.joblib artifacts/*.pkl
