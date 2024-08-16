FROM python:3.10.0

ENV PYTHONUNBUFFERED 1

WORKDIR /app

COPY poetry.lock pyproject.toml ./
RUN pip install --upgrade pip && \
    pip install poetry && \
    poetry config virtualenvs.create false

ARG DEV=false
RUN if [ "$DEV" = "true" ] ; then poetry install --with dev ; else poetry install --only main ; fi

COPY ./app/ ./
COPY ./ml/model/ ./ml/model/

ENV PYTHONPATH "${PYTHONPATH}:/app"

LABEL com.datadoghq.tags.service="titanic"
LABEL com.datadoghq.tags.env="dev"
LABEL com.datadoghq.tags.version="1.0.0"

EXPOSE 8080
CMD ddtrace-run uvicorn main:app --host 0.0.0.0 --port 8080 --log-level trace
