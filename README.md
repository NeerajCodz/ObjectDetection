# 🚀 Object Detection with Transformer Models

This project provides an object detection system using state-of-the-art transformer models, such as **DETR (DEtection TRansformer)** and **YOLOS (You Only Look One-level Series)**. The system can detect objects from uploaded images or image URLs, and it supports different models for detection and segmentation tasks. It includes a Gradio-based web interface and a FastAPI-based API for programmatic access.

You can try the demo online on Hugging Face: [Demo Link](https://huggingface.co/spaces/NeerajCodz/ObjectDetection).

## Models Supported

The following models are supported, as defined in the application:

- **DETR (DEtection TRansformer)**:
  - `facebook/detr-resnet-50`: DETR with ResNet-50 backbone for object detection. Fast and accurate for general use.
  - `facebook/detr-resnet-101`: DETR with ResNet-101 backbone for object detection. More accurate but slower than ResNet-50.
  - `facebook/detr-resnet-50-panoptic`(currently has bugs): DETR with ResNet-50 for panoptic segmentation. Detects objects and segments scenes.
  - `facebook/detr-resnet-101-panoptic`(currently has bugs): DETR with ResNet-101 for panoptic segmentation. High accuracy for complex scenes.
  
- **YOLOS (You Only Look One-level Series)**:
  - `hustvl/yolos-tiny`: YOLOS Tiny model. Lightweight and fast, ideal for resource-constrained environments.
  - `hustvl/yolos-base`: YOLOS Base model. Balances speed and accuracy for object detection.

## Features

- **Image Upload**: Upload images from your device for object detection via the Gradio interface.
- **URL Input**: Input an image URL for detection through the Gradio interface or API.
- **Model Selection**: Choose between DETR and YOLOS models for detection or panoptic segmentation.
- **Object Detection**: Detects objects and highlights them with bounding boxes and confidence scores.
- **Panoptic Segmentation**: Some models (e.g., DETR panoptic variants) support detailed scene segmentation with colored masks.
- **Image Properties**: Displays image metadata such as format, size, aspect ratio, file size, and color statistics.
- **API Access**: Use the FastAPI endpoint `/detect` to programmatically process images and retrieve detection results.

## How to Use

### 1. **Normal Git Clone Method**

Follow these steps to set up the application locally:

#### Prerequisites

- Python 3.8 or higher
- Install dependencies using `pip`

#### Clone the Repository

```bash
git clone https://github.com/NeerajCodz/ObjectDetection.git
cd ObjectDetection
```

#### Install Dependencies

Install the required dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

#### Run the Application

Start the FastAPI server using uvicorn:

```bash
uvicorn objectdetection:app --reload
```

Alternatively, launch the Gradio interface by running the main script:

```bash
python app.py
```

#### Access the Application

- For FastAPI: Open your browser and navigate to `http://localhost:8000` to use the API or view the Swagger UI.
- For Gradio: The Gradio interface URL will be displayed in the console (typically `http://127.0.0.1:7860`).

### 2. **Running with Docker**

If you prefer to use Docker to set up and run the application, follow these steps:

#### Prerequisites

- Docker installed on your machine. If you don’t have Docker, download and install it from [here](https://www.docker.com/get-started).

#### Build the Docker Image

First, clone the repository (if you haven't already):

```bash
git clone https://github.com/NeerajCodz/ObjectDetection.git
cd ObjectDetection
```

Now, build the Docker image:

```bash
docker build -t objectdetection:latest .
```

#### Run the Docker Container

Once the image is built, run the application using this command:

```bash
docker run -p 5000:5000 objectdetection:latest
```

This will start the application on port 5000. Open your browser and go to `http://localhost:5000` to access the FastAPI interface.

### 3. **Demo**

You can try the demo directly online through Hugging Face's Spaces:

[Object Detection Demo](https://huggingface.co/spaces/NeerajCodz/ObjectDetection)

## Using the API

You can interact with the application via the FastAPI `/detect` endpoint to send images and get detection results.

**Endpoint**: `/detect`

**POST**: `/detect`

**Parameters**:

- `file`: (optional) Image file (must be of type `image/*`).
- `image_url`: (optional) URL of the image.
- `model_name`: (optional) Choose from `facebook/detr-resnet-50`, `hustvl/yolos-tiny`, etc.

**Example Request Body**:

```json
{
  "image_url": "https://example.com/image.jpg",
  "model_name": "facebook/detr-resnet-50"
}
```

**Response**:

The response includes a base64-encoded image with detections, detected objects, confidence scores, and unique objects with their scores.

```json
{
  "image_url": "data:image/png;base64,...",
  "detected_objects": ["person", "car"],
  "confidence_scores": [0.95, 0.87],
  "unique_objects": ["person", "car"],
  "unique_confidence_scores": [0.95, 0.87]
}
```

## Development Setup

If you'd like to contribute or modify the application:

1. Clone the repository:

```bash
git clone https://github.com/NeerajCodz/ObjectDetection.git
cd ObjectDetection
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the FastAPI server or Gradio interface:

```bash
uvicorn objectdetection:app --reload
```

or

```bash
python app.py
```

4. Open your browser and navigate to `http://localhost:8000` (FastAPI) or the Gradio URL (typically `http://127.0.0.1:7860`).

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests for bug fixes or new features on the [GitHub repository](https://github.com/NeerajCodz/ObjectDetection).
