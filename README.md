# NLP Sentiment Analysis API with GPU Support

This repository contains a FastAPI-based sentiment analysis API that uses a BERT model for sentiment classification. The API is containerized with Docker and supports NVIDIA GPU acceleration.

## Prerequisites

- Docker installed on your system
- For GPU support: NVIDIA GPU with appropriate drivers
- NVIDIA Container Toolkit (nvidia-docker2) installed

## Building the Docker Image

To build the Docker image, run the following command from the project root:

```bash
docker build -t sentiment-analysis-api .
```

## Running the Container

### With GPU Support

To run the container with GPU support:

```bash
docker run --gpus all -p 9005:9005 sentiment-analysis-api
```

### Without GPU (CPU only)

If you don't have a GPU or don't want to use it:

```bash
docker run -p 9005:9005 sentiment-analysis-api
```

## API Endpoints

Once the container is running, the API will be available at `http://localhost:9005`

- `GET /health` - Health check endpoint
- `POST /predict` - Predict sentiment for a single text
- `POST /predict/batch` - Predict sentiment for multiple texts (up to 100)

## Example Usage

### Single Prediction

```bash
curl -X POST "http://localhost:9005/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "This movie was fantastic and I really enjoyed it!"}'
```

### Batch Prediction

```bash
curl -X POST "http://localhost:9005/predict/batch" \
     -H "Content-Type: application/json" \
     -d '{"texts": ["I loved this product", "The service was terrible"]}'
```

## Notes on GPU Usage

- The application will automatically detect if a GPU is available and use it
- You can verify GPU usage by checking the logs when the container starts
- The Dockerfile is configured with CUDA 12.1, which should be compatible with recent NVIDIA GPUs