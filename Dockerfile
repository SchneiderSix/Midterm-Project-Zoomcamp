FROM python:3.12-slim AS builder

WORKDIR /app

COPY pyproject.toml poetry.lock ./

RUN apt-get update && apt-get install -y --no-install-recommends python3-pip

RUN pip install poetry

RUN rm -rf /app/.venv

RUN poetry config virtualenvs.create false

RUN poetry config installer.max-workers 4

RUN poetry self update

RUN poetry install --no-dev --no-interaction

COPY . .

FROM builder AS api

CMD ["python", "app.py"]