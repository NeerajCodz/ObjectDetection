import argparse
import base64
import logging
import os
import sys
import traceback
import threading
from collections import Counter
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import gradio as gr
import pandas as pd
import requests
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, ImageDraw, ImageStat
from transformers import (
    DetrForObjectDetection,
    DetrForSegmentation,
    DetrImageProcessor,
    YolosForObjectDetection,
    YolosImageProcessor,
)
import nest_asyncio

# ------------------------------
# Configuration
# ------------------------------

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Model and processing constants
CONFIDENCE_THRESHOLD: float = 0.5
VALID_MODELS: List[str] = [
    "facebook/detr-resnet-50",
    "facebook/detr-resnet-101",
    "facebook/detr-resnet-50-panoptic",
    "facebook/detr-resnet-101-panoptic",
    "hustvl/yolos-tiny",
    "hustvl/yolos-base",
]
MODEL_DESCRIPTIONS: Dict[str, str] = {
    "facebook/detr-resnet-50": (
        "DETR with ResNet-50 backbone for object detection. Fast and accurate for general use."
    ),
    "facebook/detr-resnet-101": (
        "DETR with ResNet-101 backbone for object detection. More accurate but slower than ResNet-50."
    ),
    "facebook/detr-resnet-50-panoptic": (
        "DETR with ResNet-50 for panoptic segmentation. Detects objects and segments scenes."
    ),
    "facebook/detr-resnet-101-panoptic": (
        "DETR with ResNet-101 for panoptic segmentation. High accuracy for complex scenes."
    ),
    "hustvl/yolos-tiny": (
        "YOLOS Tiny model. Lightweight and fast, ideal for resource-constrained environments."
    ),
    "hustvl/yolos-base": (
        "YOLOS Base model. Balances speed and accuracy for object detection."
    ),
}

# Port configuration
DEFAULT_GRADIO_PORT: int = 7860
DEFAULT_FASTAPI_PORT: int = 8000
PORT_RANGE: range = range(7860, 7870)  # Try ports 7860-7869
MAX_PORT_ATTEMPTS: int = 10

# Thread-safe storage for lazy-loaded models and processors
models: Dict[str, any] = {}
processors: Dict[str, any] = {}
model_lock = threading.Lock()

# ------------------------------
# Model Loading
# ------------------------------

def load_model_and_processor(model_name: str) -> Tuple[any, any]:
    """
    Load and cache the specified model and processor thread-safely.

    Args:
        model_name: Name of the model to load (must be in VALID_MODELS).

    Returns:
        Tuple containing the loaded model and processor.

    Raises:
        ValueError: If the model_name is invalid or loading fails.
    """
    with model_lock:
        if model_name not in models:
            logger.info(f"Loading model: {model_name}")
            try:
                if "yolos" in model_name:
                    models[model_name] = YolosForObjectDetection.from_pretrained(model_name)
                    processors[model_name] = YolosImageProcessor.from_pretrained(model_name)
                elif "panoptic" in model_name:
                    models[model_name] = DetrForSegmentation.from_pretrained(model_name)
                    processors[model_name] = DetrImageProcessor.from_pretrained(model_name)
                else:
                    models[model_name] = DetrForObjectDetection.from_pretrained(model_name)
                    processors[model_name] = DetrImageProcessor.from_pretrained(model_name)
                logger.debug(f"Model {model_name} loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {str(e)}")
                raise ValueError(f"Failed to load model: {str(e)}")
        return models[model_name], processors[model_name]

# ------------------------------
# Image Processing
# ------------------------------

def process(image: Image.Image, model_name: str) -> Tuple[Image.Image, List[str], List[float], List[str], List[float], Dict[str, str]]:
    """
    Process an image for object detection or panoptic segmentation.

    Args:
        image: PIL Image to process.
        model_name: Name of the model to use (must be in VALID_MODELS).

    Returns:
        Tuple containing:
        - Annotated image (PIL Image).
        - List of detected object names.
        - List of confidence scores for detected objects.
        - List of unique object names.
        - List of confidence scores for unique objects.
        - Dictionary of image properties (format, size, etc.).

    Raises:
        ValueError: If the model_name is invalid.
        RuntimeError: If processing fails due to model or image issues.
    """
    if model_name not in VALID_MODELS:
        raise ValueError(f"Invalid model: {model_name}. Choose from: {VALID_MODELS}")

    try:
        # Load model and processor
        model, processor = load_model_and_processor(model_name)
        logger.debug(f"Processing image with model: {model_name}")

        # Prepare image for processing
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)

        # Initialize drawing context
        draw = ImageDraw.Draw(image)
        object_names: List[str] = []
        confidence_scores: List[float] = []
        object_counter = Counter()
        target_sizes = torch.tensor([image.size[::-1]])

        # Process panoptic segmentation or object detection
        if "panoptic" in model_name:
            processed_sizes = torch.tensor([[inputs["pixel_values"].shape[2], inputs["pixel_values"].shape[3]]])
            results = processor.post_process_panoptic(outputs, target_sizes=target_sizes, processed_sizes=processed_sizes)[0]

            for segment in results["segments_info"]:
                label = segment["label_id"]
                label_name = model.config.id2label.get(label, "Unknown")
                score = segment.get("score", 1.0)

                # Apply segmentation mask if available
                if "masks" in results and segment["id"] < len(results["masks"]):
                    mask = results["masks"][segment["id"]].cpu().numpy()
                    if mask.shape[0] > 0 and mask.shape[1] > 0:
                        mask_image = Image.fromarray((mask * 255).astype("uint8"))
                        colored_mask = Image.new("RGBA", image.size, (0, 0, 0, 0))
                        mask_draw = ImageDraw.Draw(colored_mask)
                        r, g, b = (segment["id"] * 50) % 255, (segment["id"] * 100) % 255, (segment["id"] * 150) % 255
                        mask_draw.bitmap((0, 0), mask_image, fill=(r, g, b, 128))
                        image = Image.alpha_composite(image.convert("RGBA"), colored_mask).convert("RGB")
                        draw = ImageDraw.Draw(image)

                if score > CONFIDENCE_THRESHOLD:
                    object_names.append(label_name)
                    confidence_scores.append(float(score))
                    object_counter[label_name] = float(score)
        else:
            results = processor.post_process_object_detection(outputs, target_sizes=target_sizes)[0]

            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                if score > CONFIDENCE_THRESHOLD:
                    x, y, x2, y2 = box.tolist()
                    draw.rectangle([x, y, x2, y2], outline="#32CD32", width=2)
                    label_name = model.config.id2label.get(label.item(), "Unknown")
                    text = f"{label_name}: {score:.2f}"
                    text_bbox = draw.textbbox((0, 0), text)
                    text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
                    draw.text((x2 - text_width - 2, y - text_height - 2), text, fill="#32CD32")
                    object_names.append(label_name)
                    confidence_scores.append(float(score))
                    object_counter[label_name] = float(score)

        # Compile unique objects and confidences
        unique_objects = list(object_counter.keys())
        unique_confidences = [object_counter[obj] for obj in unique_objects]

        # Calculate image properties
        properties: Dict[str, str] = {
            "Format": image.format if hasattr(image, "format") and image.format else "Unknown",
            "Size": f"{image.width}x{image.height}",
            "Width": f"{image.width} px",
            "Height": f"{image.height} px",
            "Mode": image.mode,
            "Aspect Ratio": (
                f"{round(image.width / image.height, 2)}" if image.height != 0 else "Undefined"
            ),
            "File Size": "Unknown",
            "Mean (R,G,B)": "Unknown",
            "StdDev (R,G,B)": "Unknown",
        }

        # Compute file size
        try:
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            properties["File Size"] = f"{len(buffered.getvalue()) / 1024:.2f} KB"
        except Exception as e:
            logger.error(f"Error calculating file size: {str(e)}")

        # Compute color statistics
        try:
            stat = ImageStat.Stat(image)
            properties["Mean (R,G,B)"] = ", ".join(f"{m:.2f}" for m in stat.mean)
            properties["StdDev (R,G,B)"] = ", ".join(f"{s:.2f}" for s in stat.stddev)
        except Exception as e:
            logger.error(f"Error calculating color statistics: {str(e)}")

        return image, object_names, confidence_scores, unique_objects, unique_confidences, properties

    except Exception as e:
        logger.error(f"Error in process: {str(e)}\n{traceback.format_exc()}")
        raise RuntimeError(f"Failed to process image: {str(e)}")

# ------------------------------
# FastAPI Setup
# ------------------------------

app = FastAPI(title="Object Detection API")

@app.post("/detect")
async def detect_objects_endpoint(
    file: Optional[UploadFile] = File(None),
    image_url: Optional[str] = Form(None),
    model_name: str = Form(VALID_MODELS[0]),
) -> JSONResponse:
    """
    FastAPI endpoint to detect objects in an image from file upload or URL.

    Args:
        file: Uploaded image file (optional).
        image_url: URL of the image (optional).
        model_name: Model to use for detection (default: first VALID_MODELS entry).

    Returns:
        JSONResponse containing the processed image (base64), detected objects, and confidences.

    Raises:
        HTTPException: If input validation fails or processing errors occur.
    """
    try:
        # Validate input
        if (file is None and not image_url) or (file is not None and image_url):
            raise HTTPException(
                status_code=400,
                detail="Provide either an image file or an image URL, not both.",
            )

        # Load image
        if file:
            if not file.content_type.startswith("image/"):
                raise HTTPException(status_code=400, detail="File must be an image")
            contents = await file.read()
            image = Image.open(BytesIO(contents)).convert("RGB")
        else:
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert("RGB")

        if model_name not in VALID_MODELS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model. Choose from: {VALID_MODELS}",
            )

        # Process image
        detected_image, detected_objects, detected_confidences, unique_objects, unique_confidences, _ = process(
            image, model_name
        )

        # Encode image as base64
        buffered = BytesIO()
        detected_image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        img_url = f"data:image/png;base64,{img_base64}"

        return JSONResponse(
            content={
                "image_url": img_url,
                "detected_objects": detected_objects,
                "confidence_scores": detected_confidences,
                "unique_objects": unique_objects,
                "unique_confidence_scores": unique_confidences,
            }
        )

    except requests.RequestException as e:
        logger.error(f"Error fetching image from URL: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to fetch image: {str(e)}")
    except Exception as e:
        logger.error(f"Error in FastAPI endpoint: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

# ------------------------------
# Gradio UI Setup
# ------------------------------

def create_gradio_ui() -> gr.Blocks:
    """
    Create and configure the Gradio UI for object detection.

    Returns:
        Gradio Blocks object representing the UI.

    Raises:
        RuntimeError: If UI creation fails.
    """
    try:
        with gr.Blocks(theme=gr.themes.Default(primary_hue="blue", secondary_hue="gray")) as app:
            gr.Markdown(
                f"""
                # 🚀 Object Detection App
                Upload an image or provide a URL to detect objects using state-of-the-art transformer models (DETR, YOLOS).
                Running on port: {os.getenv('GRADIO_SERVER_PORT', 'auto-selected')}
                """
            )

            with gr.Tabs():
                with gr.Tab("📷 Image Upload"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("### Input")
                            model_choice = gr.Dropdown(
                                choices=VALID_MODELS,
                                value=VALID_MODELS[0],
                                label="🔎 Select Model",
                                info="Choose a model for object detection or panoptic segmentation.",
                            )
                            model_info = gr.Markdown(
                                f"**Model Info**: {MODEL_DESCRIPTIONS[VALID_MODELS[0]]}",
                                visible=True,
                            )
                            image_input = gr.Image(type="pil", label="📷 Upload Image")
                            image_url_input = gr.Textbox(
                                label="🔗 Image URL",
                                placeholder="https://example.com/image.jpg",
                            )
                            with gr.Row():
                                submit_btn = gr.Button("✨ Detect", variant="primary")
                                clear_btn = gr.Button("🗑️ Clear", variant="secondary")

                            model_choice.change(
                                fn=lambda model_name: (
                                    f"**Model Info**: {MODEL_DESCRIPTIONS.get(model_name, 'No description available.')}"
                                ),
                                inputs=model_choice,
                                outputs=model_info,
                            )

                        with gr.Column(scale=2):
                            gr.Markdown("### Results")
                            error_output = gr.Textbox(
                                label="⚠️ Errors",
                                visible=False,
                                lines=3,
                                max_lines=5,
                            )
                            output_image = gr.Image(
                                type="pil",
                                label="🎯 Detected Image",
                                interactive=False,
                            )
                            with gr.Row():
                                objects_output = gr.DataFrame(
                                    label="📋 Detected Objects",
                                    interactive=False,
                                    value=None,
                                )
                                unique_objects_output = gr.DataFrame(
                                    label="🔍 Unique Objects",
                                    interactive=False,
                                    value=None,
                                )
                            properties_output = gr.DataFrame(
                                label="📄 Image Properties",
                                interactive=False,
                                value=None,
                            )

                    def process_for_gradio(image: Optional[Image.Image], url: Optional[str], model_name: str) -> Tuple[
                        Optional[Image.Image], Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame], str
                    ]:
                        """
                        Process image for Gradio UI and return results.

                        Args:
                            image: Uploaded PIL Image (optional).
                            url: Image URL (optional).
                            model_name: Model to use for detection.

                        Returns:
                            Tuple of detected image, objects DataFrame, unique objects DataFrame, properties DataFrame, and error message.
                        """
                        try:
                            if image is None and not url:
                                return None, None, None, None, "Please provide an image or URL"
                            if image and url:
                                return None, None, None, None, "Please provide either an image or URL, not both"

                            if url:
                                response = requests.get(url, timeout=10)
                                response.raise_for_status()
                                image = Image.open(BytesIO(response.content)).convert("RGB")

                            detected_image, objects, scores, unique_objects, unique_scores, properties = process(
                                image, model_name
                            )
                            objects_df = (
                                pd.DataFrame(
                                    {
                                        "Object": objects,
                                        "Confidence Score": [f"{score:.2f}" for score in scores],
                                    }
                                )
                                if objects
                                else pd.DataFrame(columns=["Object", "Confidence Score"])
                            )
                            unique_objects_df = (
                                pd.DataFrame(
                                    {
                                        "Unique Object": unique_objects,
                                        "Confidence Score": [f"{score:.2f}" for score in unique_scores],
                                    }
                                )
                                if unique_objects
                                else pd.DataFrame(columns=["Unique Object", "Confidence Score"])
                            )
                            properties_df = (
                                pd.DataFrame([properties])
                                if properties
                                else pd.DataFrame(columns=properties.keys())
                            )
                            return detected_image, objects_df, unique_objects_df, properties_df, ""

                        except requests.RequestException as e:
                            error_msg = f"Error fetching image from URL: {str(e)}"
                            logger.error(f"{error_msg}\n{traceback.format_exc()}")
                            return None, None, None, None, error_msg
                        except Exception as e:
                            error_msg = f"Error processing image: {str(e)}"
                            logger.error(f"{error_msg}\n{traceback.format_exc()}")
                            return None, None, None, None, error_msg

                    submit_btn.click(
                        fn=process_for_gradio,
                        inputs=[image_input, image_url_input, model_choice],
                        outputs=[output_image, objects_output, unique_objects_output, properties_output, error_output],
                    )

                    clear_btn.click(
                        fn=lambda: [None, "", None, None, None, None],
                        inputs=None,
                        outputs=[
                            image_input,
                            image_url_input,
                            output_image,
                            objects_output,
                            unique_objects_output,
                            properties_output,
                            error_output,
                        ],
                    )

                with gr.Tab("🔗 JSON Output"):
                    gr.Markdown("### Process Image for JSON Output")
                    image_input_json = gr.Image(type="pil", label="📷 Upload Image")
                    image_url_input_json = gr.Textbox(
                        label="🔗 Image URL",
                        placeholder="https://example.com/image.jpg",
                    )
                    url_model_choice = gr.Dropdown(
                        choices=VALID_MODELS,
                        value=VALID_MODELS[0],
                        label="🔎 Select Model",
                    )
                    url_model_info = gr.Markdown(
                        f"**Model Info**: {MODEL_DESCRIPTIONS[VALID_MODELS[0]]}",
                        visible=True,
                    )
                    url_submit_btn = gr.Button("🔄 Process", variant="primary")
                    url_output = gr.JSON(label="API Response")

                    url_model_choice.change(
                        fn=lambda model_name: (
                            f"**Model Info**: {MODEL_DESCRIPTIONS.get(model_name, 'No description available.')}"
                        ),
                        inputs=url_model_choice,
                        outputs=url_model_info,
                    )

                    def process_url_for_gradio(image: Optional[Image.Image], url: Optional[str], model_name: str) -> Dict:
                        """
                        Process image from file or URL for Gradio UI and return JSON response.

                        Args:
                            image: Uploaded PIL Image (optional).
                            url: Image URL (optional).
                            model_name: Model to use for detection.

                        Returns:
                            Dictionary with processed image (base64), detected objects, and confidences.
                        """
                        try:
                            if image is None and not url:
                                return {"error": "Please provide an image or URL"}
                            if image and url:
                                return {"error": "Please provide either an image or URL, not both"}

                            if url:
                                response = requests.get(url, timeout=10)
                                response.raise_for_status()
                                image = Image.open(BytesIO(response.content)).convert("RGB")

                            detected_image, objects, scores, unique_objects, unique_scores, _ = process(
                                image, model_name
                            )
                            buffered = BytesIO()
                            detected_image.save(buffered, format="PNG")
                            img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                            return {
                                "image_url": f"data:image/png;base64,{img_base64}",
                                "detected_objects": objects,
                                "confidence_scores": scores,
                                "unique_objects": unique_objects,
                                "unique_confidence_scores": unique_scores,
                            }
                        except requests.RequestException as e:
                            error_msg = f"Error fetching image from URL: {str(e)}"
                            logger.error(f"{error_msg}\n{traceback.format_exc()}")
                            return {"error": error_msg}
                        except Exception as e:
                            error_msg = f"Error processing image: {str(e)}"
                            logger.error(f"{error_msg}\n{traceback.format_exc()}")
                            return {"error": error_msg}

                    url_submit_btn.click(
                        fn=process_url_for_gradio,
                        inputs=[image_input_json, image_url_input_json, url_model_choice],
                        outputs=[url_output],
                    )

                with gr.Tab("ℹ️ Help"):
                    gr.Markdown(
                        """
                        ## How to Use
                        - **Image Upload**: Select a model, upload an image or provide a URL, and click "Detect" to see detected objects and image properties.
                        - **JSON Output**: Upload an image or enter a URL, select a model, and click "Process" to get results in JSON format.
                        - **Models**: Choose from DETR (object detection or panoptic segmentation) or YOLOS (lightweight detection).
                        - **Clear**: Reset all inputs and outputs using the "Clear" button in the Image Upload tab.
                        - **Errors**: Check the error box (Image Upload) or JSON response (JSON Output) for issues.
                        
                        ## Tips
                        - Use high-quality images for better detection results.
                        - Panoptic models (e.g., DETR-ResNet-50-panoptic) provide segmentation masks for complex scenes.
                        - For faster processing, try YOLOS-Tiny on resource-constrained devices.
                        """
                    )

        return app

    except Exception as e:
        logger.error(f"Error creating Gradio UI: {str(e)}\n{traceback.format_exc()}")
        raise RuntimeError(f"Failed to create Gradio UI: {str(e)}")

# ------------------------------
# Launcher
# ------------------------------

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments with defaults and ignore unrecognized arguments.

    Returns:
        Parsed arguments as a Namespace object.

    Raises:
        SystemExit: If argument parsing fails (handled by argparse).
    """
    parser = argparse.ArgumentParser(
        description="Launcher for Object Detection App with Gradio UI and optional FastAPI server."
    )
    parser.add_argument(
        "--gradio-port",
        type=int,
        default=DEFAULT_GRADIO_PORT,
        help=f"Port for the Gradio UI (default: {DEFAULT_GRADIO_PORT}).",
    )
    parser.add_argument(
        "--enable-fastapi",
        action="store_true",
        default=False,
        help="Enable the FastAPI server (disabled by default).",
    )
    parser.add_argument(
        "--fastapi-port",
        type=int,
        default=DEFAULT_FASTAPI_PORT,
        help=f"Port for the FastAPI server if enabled (default: {DEFAULT_FASTAPI_PORT}).",
    )

    # Parse known arguments and ignore unrecognized ones (e.g., Jupyter kernel args)
    args, _ = parser.parse_known_args()
    return args

def find_available_port(start_port: int, port_range: range, max_attempts: int) -> Optional[int]:
    """
    Find an available port within the specified range.

    Args:
        start_port: Initial port to try (e.g., from args or environment).
        port_range: Range of ports to attempt.
        max_attempts: Maximum number of ports to try.

    Returns:
        Available port number, or None if no port is found.

    Raises:
        OSError: If port binding fails for reasons other than port in use.
    """
    import socket

    port = start_port
    attempts = 0

    # Check environment variable GRADIO_SERVER_PORT
    env_port = os.getenv("GRADIO_SERVER_PORT")
    if env_port and env_port.isdigit():
        port = int(env_port)
        logger.info(f"Using GRADIO_SERVER_PORT from environment: {port}")

    while attempts < max_attempts:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", port))
                logger.debug(f"Port {port} is available")
                return port
            except OSError as e:
                if e.errno == 98:  # Port in use
                    logger.debug(f"Port {port} is in use")
                    port = port + 1 if port < max(port_range) else min(port_range)
                    attempts += 1
                else:
                    raise
            except Exception as e:
                logger.error(f"Error checking port {port}: {str(e)}")
                raise
    logger.error(f"No available port found in range {min(port_range)}-{max(port_range)} after {max_attempts} attempts")
    return None

def run_fastapi_server(host: str, port: int) -> None:
    """
    Run the FastAPI server using Uvicorn.

    Args:
        host: Host address for the FastAPI server.
        port: Port for the FastAPI server.
    """
    try:
        uvicorn.run(app, host=host, port=port)
    except Exception as e:
        logger.error(f"Error running FastAPI server: {str(e)}\n{traceback.format_exc()}")
        sys.exit(1)

def main() -> None:
    """
    Main function to launch Gradio UI and optional FastAPI server.

    Raises:
        SystemExit: If the application is interrupted or encounters an error.
    """
    try:
        # Apply nest_asyncio to allow nested event loops in Jupyter/Colab
        nest_asyncio.apply()

        # Parse command-line arguments
        args = parse_args()
        logger.info(f"Parsed arguments: {args}")

        # Find available port for Gradio
        gradio_port = find_available_port(args.gradio_port, PORT_RANGE, MAX_PORT_ATTEMPTS)
        if gradio_port is None:
            logger.error("Failed to find an available port for Gradio UI")
            sys.exit(1)

        # Launch FastAPI server in a separate thread if enabled
        if args.enable_fastapi:
            logger.info(f"Starting FastAPI server on port {args.fastapi_port}")
            fastapi_thread = threading.Thread(
                target=run_fastapi_server,
                args=("0.0.0.0", args.fastapi_port),
                daemon=True
            )
            fastapi_thread.start()

        # Launch Gradio UI
        logger.info(f"Starting Gradio UI on port {gradio_port}")
        app = create_gradio_ui()
        app.launch(server_port=gradio_port, server_name="0.0.0.0")

    except KeyboardInterrupt:
        logger.info("Application terminated by user.")
        sys.exit(0)
    except OSError as e:
        logger.error(f"Port binding error: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error running application: {str(e)}\n{traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()