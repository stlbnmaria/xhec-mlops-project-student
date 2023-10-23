# Set the base image.
FROM python:3.10-slim

# Expose necessary ports.
# Uvicorn [REST-API]
EXPOSE 8000

# Set the working directory.
WORKDIR /app

# Copy necessary files from the host.
COPY ./src ./src
COPY ./requirements.txt ./requirements.txt

# Install application dependencies.
RUN pip install -r requirements.txt

# Healthcheck.
HEALTHCHECK CMD curl --fail http://localhost:8000/_stcore/health

# Set up entrypoint to make the container executable.
ENTRYPOINT ["uvicorn", "src.web_service.main:app", "--host=0.0.0.0", "--port=8000"]
