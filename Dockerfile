# Use official Python 3.9 slim image
FROM python:3.9-slim

# Set a working directory inside the container
WORKDIR /app

# Copy local files to the container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir fastapi uvicorn transformers torch matplotlib numpy sentence_transformers pydantic pinecone-client

# Expose the FastAPI port
EXPOSE 8000

# Run FastAPI app with Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
