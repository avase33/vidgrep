FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml ./
COPY vidgrep_ml ./vidgrep_ml
RUN pip install --no-cache-dir ".[server]"
ENV VIDGREP_CLIP=mock
EXPOSE 8000
CMD ["vidgrep-ml", "serve", "--host", "0.0.0.0", "--port", "8000"]
