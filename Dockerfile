# One image runs the API, the UI, or both (scripts/start.sh).
# It follows the Hugging Face Docker Spaces conventions (non-root user with uid 1000),
# so the same file deploys there unchanged.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --user -r requirements.txt

COPY --chown=user . .

# Train the models and seed the demo at build time so the container starts in seconds.
RUN python scripts/bootstrap.py

EXPOSE 8000 8501
CMD ["bash", "scripts/start.sh"]
