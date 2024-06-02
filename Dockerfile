# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt requirements.txt

# Install any needed packages specified in requirements.txt
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/* && \
    python -m venv /venv && \
    /venv/bin/python -m pip install --no-cache-dir --upgrade pip && \
    /venv/bin/pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the rest of the application code into the container
COPY . /app

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Expose port 8080 for the FastAPI application
EXPOSE 8080

# Run the FastAPI application with uvicorn
CMD ["/venv/bin/uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--log-level=debug"]
