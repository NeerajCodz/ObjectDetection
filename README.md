# 🚀 Object Detection with Transformer Models

This project provides a robust object detection system leveraging state-of-the-art transformer models, including **DETR (DEtection TRansformer)** and **YOLOS (You Only Look One-level Series)**. The system supports object detection and panoptic segmentation from uploaded images or image URLs. It features a user-friendly **Gradio** web interface for interactive use and a **FastAPI** endpoint for programmatic access.

Try the online demo on Hugging Face Spaces: [Object Detection Demo](https://huggingface.co/spaces/NeerajCodz/ObjectDetection).

## Models Supported

The application supports the following models, each tailored for specific detection or segmentation tasks:

- **DETR (DEtection TRansformer)**:
  - `facebook/detr-resnet-50`: Fast and accurate object detection with a ResNet-50 backbone.
  - `facebook/detr-resnet-101`: Higher accuracy object detection with a ResNet-101 backbone, slower than ResNet-50.
  - `facebook/detr-resnet-50-panoptic`: Panoptic segmentation with ResNet-50 (note: may have stability issues).
  - `facebook/detr-resnet-101-panoptic`: Panoptic segmentation with ResNet-101 (note: may have stability issues).
  
- **YOLOS (You Only Look One-level Series)**:
  - `hustvl/yolos-tiny`: Lightweight and fast, ideal for resource-constrained environments.
  - `hustvl/yolos-base`: Balances speed and accuracy for object detection.

## Features

- **Image Upload**: Upload images via the Gradio interface for object detection.
- **URL Input**: Provide image URLs for detection through the Gradio interface or API.
- **Model Selection**: Choose between DETR and YOLOS models for detection or panoptic segmentation.
- **Object Detection**: Highlights detected objects with bounding boxes and confidence scores.
- **Panoptic Segmentation**: Supports scene segmentation with colored masks (DETR panoptic models).
- **Image Properties**: Displays metadata like format, size, aspect ratio, file size, and color statistics.
- **API Access**: Programmatically process images via the FastAPI `/detect` endpoint.
- **Flexible Deployment**: Run locally, in Docker, or in cloud environments like Google Colab.

## How to Use

### 1. **Local Setup (Git Clone)**

Follow these steps to set up the application locally:

#### Prerequisites

- Python 3.8 or higher
- `pip` for installing dependencies
- Git for cloning the repository

#### Clone the Repository

```bash
git clone https://github.com/NeerajCodz/ObjectDetection
cd ObjectDetection
```

#### Install Dependencies

Install required packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```

#### Run the Application

Launch the Gradio interface:

```bash
python app.py
```

To enable the FastAPI server:

```bash
python app.py --enable-fastapi
```

#### Access the Application

- **Gradio**: Open the URL displayed in the console (typically `http://127.0.0.1:7860`).
- **FastAPI**: Navigate to `http://localhost:8000` for the API or Swagger UI (if enabled).

### 2. **Running with Docker**

Use Docker for a containerized setup.

#### Prerequisites

- Docker installed on your machine. Download from [Docker's official site](https://www.docker.com/get-started).

#### Pull the Docker Image

Pull the pre-built image from Docker Hub:

```bash
docker pull neerajcodz/objectdetection:latest
```

#### Run the Docker Container

Run the application on port 8080:

```bash
docker run -d -p 8080:80 neerajcodz/objectdetection:latest
```

Access the interface at `http://localhost:8080`.

#### Build and Run the Docker Image

To build the Docker image locally:

1. Ensure you have a `Dockerfile` in the repository root (example provided in the repository).
2. Build the image:

```bash
docker build -t objectdetection:local .
```

3. Run the container:

```bash
docker run -d -p 8080:80 objectdetection:local
```

Access the interface at `http://localhost:8080`.

### 3. **Demo**

Try the demo on Hugging Face Spaces:

[Object Detection Demo](https://huggingface.co/spaces/NeerajCodz/ObjectDetection)

## Command-Line Arguments

The `app.py` script supports the following command-line arguments:

- `--gradio-port <port>`: Specify the port for the Gradio UI (default: 7860).
  - Example: `python app.py --gradio-port 7870`
- `--enable-fastapi`: Enable the FastAPI server (disabled by default).
  - Example: `python app.py --enable-fastapi`
- `--fastapi-port <port>`: Specify the port for the FastAPI server (default: 8000).
  - Example: `python app.py --enable-fastapi --fastapi-port 8001`
- `--confidence-threshold <float-value)`: Confidence threshold for detection (Range: 0 - 1) (default: 0.5).
  - Example: `python app.py --confidence-threshold 0.75`

You can combine arguments:

```bash
python app.py --gradio-port 7870 --enable-fastapi --fastapi-port 8001 --confidence-threshold 0.75
```

Alternatively, set the `GRADIO_SERVER_PORT` environment variable:

```bash
export GRADIO_SERVER_PORT=7870
python app.py
```

## Using the API

**Note**: The FastAPI API is currently unstable and may require additional configuration for production use.

The `/detect` endpoint allows programmatic image processing.

### Running the FastAPI Server

Enable FastAPI when launching the script:

```bash
python app.py --enable-fastapi
```

Or run FastAPI separately with Uvicorn:

```bash
uvicorn objectdetection:app --host 0.0.0.0 --port 8000
```

Access the Swagger UI at `http://localhost:8000/docs` for interactive testing.

### Endpoint Details

- **Endpoint**: `POST /detect`
- **Parameters**:
  - `file`: (optional) Image file (must be `image/*` type).
  - `image_url`: (optional) URL of the image.
  - `model_name`: (optional) Model name (e.g., `facebook/detr-resnet-50`, `hustvl/yolos-tiny`).
- **Content-Type**: `multipart/form-data` for file uploads, `application/json` for URL inputs.

### Example Requests

#### Using `curl` with an Image URL

```bash
curl -X POST "http://localhost:8000/detect" \\
  -H "Content-Type: application/json" \\
  -d '{"image_url": "https://example.com/image.jpg", "model_name": "facebook/detr-resnet-50"}'
```

#### Using `curl` with an Image File

```bash
curl -X POST "http://localhost:8000/detect" \\
  -F "file=@/path/to/image.jpg" \\
  -F "model_name=facebook/detr-resnet-50"
```

### Response Format

The response includes a base64-encoded image with detections and detection details:

```json
{
  "image_url": "data:image/png;base64,...",
  "detected_objects": ["person", "car"],
  "confidence_scores": [0.95, 0.87],
  "unique_objects": ["person", "car"],
  "unique_confidence_scores": [0.95, 0.87]
}
```

### Notes

- Ensure only one of `file` or `image_url` is provided.
- The API may experience instability with panoptic models; use object detection models for reliability.
- Test the API using the Swagger UI for easier debugging.

## Development Setup

To contribute or modify the application:

1. Clone the repository:

```bash
git clone https://github.com/NeerajCodz/ObjectDetection
cd ObjectDetection
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the application:

```bash
python app.py
```

Or run FastAPI:

```bash
uvicorn objectdetection:app --host 0.0.0.0 --port 8000
```

4. Access at `http://localhost:7860` (Gradio) or `http://localhost:8000` (FastAPI).

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a feature or bugfix branch (`git checkout -b feature/your-feature`).
3. Commit changes (`git commit -m "Add your feature"`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request on the [GitHub repository](https://github.com/NeerajCodz/ObjectDetection).

Please include tests and documentation for new features. Report issues via GitHub Issues.

## Troubleshooting

- **Port Conflicts**: If port 7860 is in use, specify a different port with `--gradio-port` or set `GRADIO_SERVER_PORT`.
  - Example: `python app.py --gradio-port 7870`
- **Colab Asyncio Error**: If you encounter `RuntimeError: asyncio.run() cannot be called from a running event loop` in Colab, the application now uses `nest_asyncio` to handle this. Ensure `nest_asyncio` is installed (`pip install nest_asyncio`).
- **Panoptic Model Bugs**: Avoid `detr-resnet-*-panoptic` models until stability issues are resolved.
- **API Instability**: Test with smaller images and object detection models first.
- **FastAPI Not Starting**: Ensure `--enable-fastapi` is used, and check that the specified `--fastapi-port` (default: 8000) is available.

For further assistance, open an issue on the [GitHub repository](https://github.com/NeerajCodz/ObjectDetection).
