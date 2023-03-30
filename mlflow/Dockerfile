FROM python:3.10
WORKDIR /app

RUN pip install mlflow==2.0.1

EXPOSE 5000

CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", "--backend-store-uri", "sqlite:///mlflow.db"]