# syntax=docker/dockerfile:1.7

FROM python:3.12-slim-bookworm@sha256:d50fb7611f86d04a3b0471b46d7557818d88983fc3136726336b2a4c657aa30b AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

COPY LICENSE README.md pyproject.toml ./
COPY src/ src/

RUN python -m pip wheel --wheel-dir /wheels .

FROM python:3.12-slim-bookworm@sha256:d50fb7611f86d04a3b0471b46d7557818d88983fc3136726336b2a4c657aa30b AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN groupadd --gid 10001 demofml \
    && useradd --uid 10001 --gid demofml --create-home demofml

# LightGBM links against OpenMP at runtime; the slim base does not ship it.
RUN apt-get update \
    && apt-get install --no-install-recommends --yes libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /wheels /wheels
RUN python -m pip install /wheels/*.whl \
    && rm -rf /wheels

COPY configs/ /opt/demofml/configs/
COPY pyproject.toml /opt/demofml/pyproject.toml
COPY docs/research/campaign-2-prospective-factor-plan.md /opt/demofml/docs/research/campaign-2-prospective-factor-plan.md
COPY docs/research/campaign-2-prospective-factor-v2.md /opt/demofml/docs/research/campaign-2-prospective-factor-v2.md
# Sealed by campaign-3-lightgbm-causal-v2-envelope-v1: the acceptance gate
# verifies its digest, so the image must carry the document, not just the TOMLs.
COPY docs/research/campaign-3-lightgbm-causal-v2-hypothesis-v1.md /opt/demofml/docs/research/campaign-3-lightgbm-causal-v2-hypothesis-v1.md

USER 10001:10001
WORKDIR /home/demofml

ENTRYPOINT ["python", "-m", "demofml"]

FROM runtime AS mlflow

USER 0:0
RUN python -m pip install \
    "mlflow==3.15.1" \
    "psycopg[binary]==3.3.4" \
    && python -m pip install --no-deps --upgrade "cryptography==50.0.0" \
    && python -c "import cryptography, mlflow; assert mlflow.__version__ == '3.15.1'; assert cryptography.__version__ == '50.0.0'"
USER 10001:10001

ENTRYPOINT ["mlflow"]
