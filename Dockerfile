# Use the official Python 3.6 image as the base image
FROM python:3.9-slim

# Install poetry
RUN pip install poetry==1.4.2

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# Set the working directory inside the container
WORKDIR /app

# Copy poetry files
COPY pyproject.toml poetry.lock ./
RUN touch README.md

# Install any dependencies specified in the requirements file
RUN poetry install --without dev --no-root && rm -rf $POETRY_CACHE_DIR

# Copy rest of the project
COPY data_science_bank_churn ./data_science_bank_churn

# Install any dependencies specified in the requirements file
RUN poetry install --without dev

# Expose the ports for each API (adjust the port numbers as needed)
EXPOSE 8000

ENTRYPOINT ["poetry", "run", "python", "-m", "uvicorn", "--host", "0.0.0.0", "--port", "8000", "data_science_bank_churn.src.app:app"]
