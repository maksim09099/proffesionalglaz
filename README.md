# Iris Analysis Project (Windows + PyCharm)

Проект переделан с face-recognition на **анализ изображений глаза/радужки**.

## Почему не face recognition

Исходные данные описаны как изображения глаза (`.jpg`) + разметка в `.mat` (`fotoXX.mat`, `all_images.mat`). Это не соответствует задаче детекции/распознавания лица. Корректная базовая постановка в этом репозитории: **локализация/сегментация радужки**.

## Архитектура

- `src/iris_project/data_audit.py` — аудит структуры датасета, чтение `.mat`, проверка соответствия `.jpg` ↔ `.mat`, эвристика типа задачи.
- `src/iris_project/preprocessing.py` — загрузка изображений, нормализация, split train/val/test.
- `src/iris_project/model.py` — baseline-модель локализации радужки (HoughCircles).
- `src/iris_project/pipeline.py` — единый inference pipeline.
- `api/main.py` — FastAPI (`/health`, `/model-info`, `/predict`).
- `gui/app.py` — Tkinter GUI для локального анализа глаза.
- `notebooks/` — 5 notebook-файлов для аудита, подготовки, обучения, оценки и API-демо.
- `tests/` — тесты под новую eye/iris логику.

## Структура `.mat`

Используйте notebook `01_data_audit.ipynb` или функцию `audit_dataset`:

- читает каждый `.mat`;
- показывает ключи и формы массивов;
- определяет, есть ли маски/контуры/ID/labels;
- сопоставляет `fotoXX.jpg` и `fotoXX.mat`;
- отдельно учитывает `all_images.mat` как агрегированный файл.

## Запуск в PyCharm (Windows)

1. Создайте интерпретатор Python 3.10+.
2. Установите зависимости:
   ```bash
   pip install -r requirements.txt
   ```
3. Положите датасет в `data/`.

## API

```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

Endpoints:
- `GET /health`
- `GET /model-info`
- `POST /predict` (multipart file)

## GUI

```bash
python gui/app.py
```

## Notebooks

Откройте в PyCharm/Jupyter:
- `notebooks/01_data_audit.ipynb`
- `notebooks/02_preprocessing.ipynb`
- `notebooks/03_training.ipynb`
- `notebooks/04_evaluation.ipynb`
- `notebooks/05_demo_api_usage.ipynb`

Во всех ноутбуках первой ячейкой задается `PROJECT_ROOT` и `sys.path`.

## Тесты

```bash
pytest
```
