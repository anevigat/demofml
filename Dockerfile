# syntax=docker/dockerfile:1.7

FROM python:3.14-slim-bookworm@sha256:86f975aca15cf04a40b399eebede9aea7c82eae084d1f1a0a6ef6bcaae871a30 AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

COPY LICENSE README.md pyproject.toml ./
COPY src/ src/

RUN python -m pip wheel --wheel-dir /wheels .

FROM python:3.14-slim-bookworm@sha256:86f975aca15cf04a40b399eebede9aea7c82eae084d1f1a0a6ef6bcaae871a30 AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN groupadd --gid 10001 demofml \
    && useradd --uid 10001 --gid demofml --create-home demofml

COPY --from=builder /wheels /wheels
RUN python -m pip install /wheels/*.whl \
    && rm -rf /wheels

USER 10001:10001
WORKDIR /home/demofml

ENTRYPOINT ["python", "-m", "demofml"]
