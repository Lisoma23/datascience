@echo off

if "%1"=="install" goto install
if "%1"=="train" goto train
if "%1"=="api" goto api
if "%1"=="test" goto test
if "%1"=="lint" goto lint
if "%1"=="dashboard" goto dashboard
if "%1"=="clean" goto clean

echo Commandes disponibles : install, train, api, test, lint, dashboard, clean
goto end

:install
python -m venv .venv
.venv\Scripts\pip install --upgrade pip
.venv\Scripts\pip install -r requirements.txt
goto end

:train
.venv\Scripts\jupyter nbconvert --to notebook --execute notebooks\01_eda.ipynb --output 01_eda.ipynb
.venv\Scripts\jupyter nbconvert --to notebook --execute notebooks\02_preprocessing.ipynb --output 02_preprocessing.ipynb
.venv\Scripts\jupyter nbconvert --to notebook --execute notebooks\03_modeling.ipynb --output 03_modeling.ipynb
goto end

:api
.venv\Scripts\uvicorn backend.api:app --reload --port 8000
goto end

:test
.venv\Scripts\python -m pytest tests/ -v
goto end

:lint
ruff check backend/ tests/ frontend/
goto end

:dashboard
.venv\Scripts\streamlit run frontend\streamlit_app.py
goto end

:clean
for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"
del /q artifacts\*.joblib artifacts\*.json 2>nul
goto end

:end
