FROM python:3.10

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    openjdk-11-jre-headless \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for PySpark
ENV PYSPARK_PYTHON=python
ENV PYSPARK_DRIVER_PYTHON=python
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV SPARK_HOME=/spark

# Download and install Spark
RUN curl -O https://downloads.apache.org/spark/spark-3.2.3/spark-3.2.3-bin-hadoop3.2.tgz \
    && tar -xvf spark-3.2.3-bin-hadoop3.2.tgz \
    && mv spark-3.2.3-bin-hadoop3.2 /spark \
    && rm spark-3.2.3-bin-hadoop3.2.tgz

WORKDIR /app

COPY requirements.txt ./requirements.txt

RUN pip3 install --no-cache-dir -r requirements.txt

EXPOSE 8000

COPY . /app

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

#CMD python app.py
# the --port(8000) must match with the EXPOSE port above(8000)