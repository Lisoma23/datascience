.PHONY: install train api test lint dashboard clean

install:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt

train:
	.venv/bin/jupyter nbconvert --to notebook --execute notebooks/01_eda.ipynb --output 01_eda.ipynb
	.venv/bin/jupyter nbconvert --to notebook --execute notebooks/02_preprocessing.ipynb --output 02_preprocessing.ipynb
	.venv/bin/jupyter nbconvert --to notebook --execute notebooks/03_modeling.ipynb --output 03_modeling.ipynb

api:
	.venv/bin/uvicorn backend.api:app --reload --port 8000

test:
	.venv/bin/python -m pytest tests/ -v

lint:
	.venv/bin/ruff check backend/ tests/ frontend/

dashboard:
	.venv/bin/streamlit run frontend/streamlit_app.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf artifacts/*.joblib artifacts/*.json
