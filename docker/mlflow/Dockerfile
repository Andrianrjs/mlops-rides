FROM ghcr.io/mlflow/mlflow:v3.18.0
# MLflow itself comes with the base image; add the S3 client (artifacts on RustFS) and the Postgres driver (backend store)
RUN pip install --no-cache-dir boto3 "psycopg[binary]"
