# Use the official Python 3.6 image as the base image
FROM python:3.6-slim

# Set the working directory inside the container
WORKDIR /app

# Copy all files
COPY . .

# Install any dependencies specified in the requirements file
RUN pip install --no-cache-dir -r src/requirements.txt

# Expose the ports for each API (adjust the port numbers as needed)
EXPOSE 8000

ENTRYPOINT ["uvicorn", "--host", "0.0.0.0", "--port", "8000", "src.app:app"]
