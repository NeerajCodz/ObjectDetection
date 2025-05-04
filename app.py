import argparse
import base64
import logging
import os
import sys
import threading
from collections import Counter
from io import BytesIO
from typing import Dict, List, Optional, Tuple, Union

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

# Configure logging for debugging and monitoring
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Define constants for model and server configuration
CONFIDENCE_THRESHOLD: float = 0.5  # Default threshold for object detection confidence
VALID_MODELS: List[str] = [
    "facebook/detr-resnet-50",
    "facebook/detr-resnet-101",
    "facebook/detr-resnet-50-panoptic",
    "facebook/detr-resnet-101-panoptic",
    "hustvl/yolos-tiny",
    "hustvl/yolos-base",
]
MODEL_DESCRIPTIONS: Dict[str, str] = {
    "facebook/detr-resnet-50": "DETR with ResNet-50 for object detection. Fast and accurate.",
    "facebook/detr-resnet-101": "DETR with ResNet-101 for object detection. More accurate, slower.",
    "facebook/detr-resnet-50-panoptic": "DETR with ResNet-50 for panoptic segmentation.",
    "facebook/detr-resnet-101-panoptic": "DETR with ResNet-101 for panoptic segmentation.",
    "hustvl/yolos-tiny": "YOLOS Tiny. Lightweight and fast.",
    "hustvl/yolos-base": "YOLOS Base. Balances speed and accuracy."
}
DEFAULT_GRADIO_PORT: int = 7860  # Default port for Gradio UI
DEFAULT_FASTAPI_PORT: int = 8000  # Default port for FastAPI server
PORT_RANGE: range = range(7860, 7870)  # Range of ports to try for Gradio
MAX_PORT_ATTEMPTS: int = 10  # Maximum attempts to find an available port

# Thread-safe storage for lazy-loaded models and processors
models: Dict[str, any] = {}
processors: Dict[str, any] = {}
model_lock = threading.Lock()

# ------------------------------
# Image Processing
# ------------------------------

def process_image(
    image: Optional[Image.Image],
    url: Optional[str],
    model_name: str,
    for_json: bool = False,
    confidence_threshold: float = CONFIDENCE_THRESHOLD
) -> Union[Dict, Tuple[Optional[Image.Image], Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame], str]]:
    """
    Process an image for object detection or panoptic segmentation, handling Gradio and FastAPI inputs.

    Args:
        image: PIL Image object from file upload (optional).
        url: URL of the image to process (optional).
        model_name: Name of the model to use (must be in VALID_MODELS).
        for_json: If True, return JSON dict for API/JSON tab; else, return tuple for Gradio Home tab.
        confidence_threshold: Minimum confidence score for detection (default: 0.5).

    Returns:
        For JSON: Dict with base64-encoded image, detected objects, and confidence scores.
        For Gradio: Tuple of (annotated image, objects DataFrame, unique objects DataFrame, properties DataFrame, error message).
    """
    try:
        # Validate input: ensure exactly one of image or URL is provided
        if image is None and not url:
            return {"error": "Please provide an image or URL"} if for_json else (None, None, None, None, "Please provide an image or URL")
        if image and url:
            return {"error": "Provide either an image or URL, not both"} if for_json else (None, None, None, None, "Provide either an image or URL, not both")
        if model_name not in VALID_MODELS:
            error_msg = f"Invalid model: {model_name}. Choose from: {VALID_MODELS}"
            return {"error": error_msg} if for_json else (None, None, None, None, error_msg)

        # Calculate margin threshold: (1 - confidence_threshold) / 2 + confidence_threshold
        margin_threshold = (1 - confidence_threshold) / 2 + confidence_threshold

        # Load image from URL if provided
        if url:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert("RGB")

        # Load model and processor thread-safely
        with model_lock:
            if model_name not in models:
                logger.info(f"Loading model: {model_name}")
                try:
                    # Select appropriate model and processor based on model name
                    if "yolos" in model_name:
                        models[model_name] = YolosForObjectDetection.from_pretrained(model_name)
                        processors[model_name] = YolosImageProcessor.from_pretrained(model_name)
                    elif "panoptic" in model_name:
                        models[model_name] = DetrForSegmentation.from_pretrained(model_name)
                        processors[model_name] = DetrImageProcessor.from_pretrained(model_name)
                    else:
                        models[model_name] = DetrForObjectDetection.from_pretrained(model_name)
                        processors[model_name] = DetrImageProcessor.from_pretrained(model_name)
                except Exception as e:
                    error_msg = f"Failed to load model: {str(e)}"
                    logger.error(error_msg)
                    return {"error": error_msg} if for_json else (None, None, None, None, error_msg)
            model, processor = models[model_name], processors[model_name]

        # Prepare image for model processing
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)

        # Initialize drawing context for annotations
        draw = ImageDraw.Draw(image)
        object_names: List[str] = []
        confidence_scores: List[float] = []
        object_counter = Counter()
        target_sizes = torch.tensor([image.size[::-1]])

        # Process results based on model type (panoptic or object detection)
        if "panoptic" in model_name:
            # Handle panoptic segmentation
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
                if score > confidence_threshold:
                    object_names.append(label_name)
                    confidence_scores.append(float(score))
                    object_counter[label_name] = float(score)
        else:
            # Handle object detection
            results = processor.post_process_object_detection(outputs, target_sizes=target_sizes)[0]
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                if score > confidence_threshold:
                    x, y, x2, y2 = box.tolist()
                    label_name = model.config.id2label.get(label.item(), "Unknown")
                    text = f"{label_name}: {score:.2f}"
                    text_bbox = draw.textbbox((0, 0), text)
                    text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
                    # Use yellow for confidence_threshold <= score < margin_threshold, green for >= margin_threshold
                    color = "#FFFF00" if score < margin_threshold else "#32CD32"
                    draw.rectangle([x, y, x2, y2], outline=color, width=2)
                    draw.text((x2 - text_width - 2, y - text_height - 2), text, fill=color)
                    object_names.append(label_name)
                    confidence_scores.append(float(score))
                    object_counter[label_name] = float(score)

        # Compile unique objects and their highest confidence scores
        unique_objects = list(object_counter.keys())
        unique_confidences = [object_counter[obj] for obj in unique_objects]

        # Calculate image properties (metadata)
        properties: Dict[str, str] = {
            "Format": image.format if hasattr(image, "format") and image.format else "Unknown",
            "Size": f"{image.width}x{image.height}",
            "Width": f"{image.width} px",
            "Height": f"{image.height} px",
            "Mode": image.mode,
            "Aspect Ratio": f"{round(image.width / image.height, 2)}" if image.height != 0 else "Undefined",
            "File Size": "Unknown",
            "Mean (R,G,B)": "Unknown",
            "StdDev (R,G,B)": "Unknown",
        }
        try:
            # Compute file size
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            properties["File Size"] = f"{len(buffered.getvalue()) / 1024:.2f} KB"
            # Compute color statistics
            stat = ImageStat.Stat(image)
            properties["Mean (R,G,B)"] = ", ".join(f"{m:.2f}" for m in stat.mean)
            properties["StdDev (R,G,B)"] = ", ".join(f"{s:.2f}" for s in stat.stddev)
        except Exception as e:
            logger.error(f"Error calculating image stats: {str(e)}")

        # Prepare output based on request type
        if for_json:
            # Return JSON with base64-encoded image
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            return {
                "image_url": f"data:image/png;base64,{img_base64}",
                "detected_objects": object_names,
                "confidence_scores": confidence_scores,
                "unique_objects": unique_objects,
                "unique_confidence_scores": unique_confidences,
            }
        else:
            # Return tuple for Gradio Home tab with DataFrames
            objects_df = (
                pd.DataFrame({"Object": object_names, "Confidence Score": [f"{score:.2f}" for score in confidence_scores]})
                if object_names else pd.DataFrame(columns=["Object", "Confidence Score"])
            )
            unique_objects_df = (
                pd.DataFrame({"Unique Object": unique_objects, "Confidence Score": [f"{score:.2f}" for score in unique_confidences]})
                if unique_objects else pd.DataFrame(columns=["Unique Object", "Confidence Score"])
            )
            properties_df = pd.DataFrame([properties]) if properties else pd.DataFrame(columns=properties.keys())
            return image, objects_df, unique_objects_df, properties_df, ""

    except requests.RequestException as e:
        # Handle URL fetch errors
        error_msg = f"Error fetching image from URL: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        return {"error": error_msg} if for_json else (None, None, None, None, error_msg)
    except Exception as e:
        # Handle general processing errors
        error_msg = f"Error processing image: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        return {"error": error_msg} if for_json else (None, None, None, None, error_msg)

# ------------------------------
# FastAPI Setup
# ------------------------------

app = FastAPI(title="Object Detection API")

@app.post("/detect")
async def detect_objects_endpoint(
    file: Optional[UploadFile] = File(None),
    image_url: Optional[str] = Form(None),
    model_name: str = Form(VALID_MODELS[0]),
    confidence_threshold: float = Form(CONFIDENCE_THRESHOLD),
) -> JSONResponse:
    """
    FastAPI endpoint to detect objects in an image from file upload or URL.

    Args:
        file: Uploaded image file (optional).
        image_url: URL of the image (optional).
        model_name: Model to use for detection (default: first VALID_MODELS entry).
        confidence_threshold: Confidence threshold for detection (default: 0.5).

    Returns:
        JSONResponse with base64-encoded image, detected objects, and confidence scores.

    Raises:
        HTTPException: For invalid inputs or processing errors.
    """
    try:
        # Validate input: ensure exactly one of file or URL
        if (file is None and not image_url) or (file is not None and image_url):
            raise HTTPException(status_code=400, detail="Provide either an image file or an image URL, not both.")
        # Validate confidence threshold
        if not 0 <= confidence_threshold <= 1:
            raise HTTPException(status_code=400, detail="Confidence threshold must be between 0 and 1.")
        # Load image from file if provided
        image = None
        if file:
            if not file.content_type.startswith("image/"):
                raise HTTPException(status_code=400, detail="File must be an image")
            contents = await file.read()
            image = Image.open(BytesIO(contents)).convert("RGB")
        # Process image with specified parameters
        result = process_image(image, image_url, model_name, for_json=True, confidence_threshold=confidence_threshold)
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in FastAPI endpoint: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

# ------------------------------
# Gradio UI Setup
# ------------------------------

def create_gradio_ui() -> gr.Blocks:
    """
    Create and configure the Gradio UI for object detection with Home, JSON, and Help tabs.

    Returns:
        Gradio Blocks object representing the UI.

    Raises:
        RuntimeError: If UI creation fails.
    """
    try:
        # Initialize Gradio Blocks with a custom theme
        with gr.Blocks(theme=gr.themes.Default(primary_hue="blue", secondary_hue="gray")) as demo:
            # Display app header
            gr.Markdown(
                f"""
                # 🚀 Object Detection App
                Upload an image or provide a URL to detect objects using transformer models (DETR, YOLOS).
                Running on port: {os.getenv('GRADIO_SERVER_PORT', 'auto-selected')}
                """
            )

            # Create tabbed interface
            with gr.Tabs():
                # Home tab (formerly Image Upload)
                with gr.Tab("🏠 Home"):
                    with gr.Row():
                        # Left column for inputs
                        with gr.Column(scale=1):
                            gr.Markdown("### Input")
                            # Model selection dropdown
                            model_choice = gr.Dropdown(choices=VALID_MODELS, value=VALID_MODELS[0], label="🔎 Select Model")
                            model_info = gr.Markdown(f"**Model Info**: {MODEL_DESCRIPTIONS[VALID_MODELS[0]]}")
                            # Image upload input
                            image_input = gr.Image(type="pil", label="📷 Upload Image")
                            # Image URL input
                            image_url_input = gr.Textbox(label="🔗 Image URL", placeholder="https://example.com/image.jpg")
                            # Buttons for submission and clearing
                            with gr.Row():
                                submit_btn = gr.Button("✨ Detect", variant="primary")
                                clear_btn = gr.Button("🗑️ Clear", variant="secondary")

                            # Update model info when model changes
                            model_choice.change(
                                fn=lambda model_name: f"**Model Info**: {MODEL_DESCRIPTIONS.get(model_name, 'No description available.')}",
                                inputs=model_choice,
                                outputs=model_info,
                            )

                        # Right column for results
                        with gr.Column(scale=2):
                            gr.Markdown("### Results")
                            # Error display (hidden by default)
                            error_output = gr.Textbox(label="⚠️ Errors", visible=False, lines=3, max_lines=5)
                            # Annotated image output
                            output_image = gr.Image(type="pil", label="🎯 Detected Image", interactive=False)
                            # Detected and unique objects tables
                            with gr.Row():
                                objects_output = gr.DataFrame(label="📋 Detected Objects", interactive=False)
                                unique_objects_output = gr.DataFrame(label="🔍 Unique Objects", interactive=False)
                            # Image properties table
                            properties_output = gr.DataFrame(label="📄 Image Properties", interactive=False)

                    # Process image when Detect button is clicked
                    submit_btn.click(
                        fn=process_image,
                        inputs=[image_input, image_url_input, model_choice],
                        outputs=[output_image, objects_output, unique_objects_output, properties_output, error_output],
                    )

                    # Clear all inputs and outputs
                    clear_btn.click(
                        fn=lambda: [None, "", None, None, None, None],
                        inputs=None,
                        outputs=[image_input, image_url_input, output_image, objects_output, unique_objects_output, properties_output, error_output],
                    )

                # JSON tab for API-like output
                with gr.Tab("🔗 JSON"):
                    with gr.Row():
                        # Left column for inputs
                        with gr.Column(scale=1):
                            gr.Markdown("### Process Image for JSON")
                            # Model selection dropdown
                            url_model_choice = gr.Dropdown(choices=VALID_MODELS, value=VALID_MODELS[0], label="🔎 Select Model")
                            url_model_info = gr.Markdown(f"**Model Info**: {MODEL_DESCRIPTIONS[VALID_MODELS[0]]}")
                            # Image upload input
                            image_input_json = gr.Image(type="pil", label="📷 Upload Image")
                            # Image URL input
                            image_url_input_json = gr.Textbox(label="🔗 Image URL", placeholder="https://example.com/image.jpg")
                            # Process button
                            url_submit_btn = gr.Button("🔄 Process", variant="primary")

                            # Update model info when model changes
                            url_model_choice.change(
                                fn=lambda model_name: f"**Model Info**: {MODEL_DESCRIPTIONS.get(model_name, 'No description available.')}",
                                inputs=url_model_choice,
                                outputs=url_model_info,
                            )

                        # Right column for JSON output
                        with gr.Column(scale=1):
                            # JSON output display
                            url_output = gr.JSON(label="API Response")

                    # Process image and return JSON when Process button is clicked
                    url_submit_btn.click(
                        fn=lambda img, url, model: process_image(img, url, model, for_json=True),
                        inputs=[image_input_json, image_url_input_json, url_model_choice],
                        outputs=[url_output],
                    )

                # Help tab with usage instructions
                with gr.Tab("ℹ️ Help"):
                    gr.Markdown(
                        """
                        ## How to Use
                        - **Home**: Select a model, upload an image or provide a URL, click "Detect" to see results.
                        - **JSON**: Select a model, upload an image or enter a URL, click "Process" for JSON output.
                        - **Models**: Choose DETR (detection or panoptic) or YOLOS (lightweight detection).
                        - **Clear**: Reset inputs/outputs in Home tab.
                        - **Errors**: Check error box (Home) or JSON response (JSON) for issues.

                        ## Tips
                        - Use high-quality images for better results.
                        - Panoptic models provide segmentation masks for complex scenes.
                        - YOLOS-Tiny is faster for resource-constrained devices.
                        """
                    )

        return demo

    except Exception as e:
        logger.error(f"Error creating Gradio UI: {str(e)}\n{traceback.format_exc()}")
        raise RuntimeError(f"Failed to create Gradio UI: {str(e)}")

# ------------------------------
# Launcher
# ------------------------------

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for configuring the application.

    Returns:
        Parsed arguments as a Namespace object.
    """
    parser = argparse.ArgumentParser(description="Object Detection App with Gradio and FastAPI.")
    # Gradio port argument
    parser.add_argument("--gradio-port", type=int, default=DEFAULT_GRADIO_PORT, help=f"Gradio port (default: {DEFAULT_GRADIO_PORT}).")
    # FastAPI enable flag
    parser.add_argument("--enable-fastapi", action="store_true", help="Enable FastAPI server.")
    # FastAPI port argument
    parser.add_argument("--fastapi-port", type=int, default=DEFAULT_FASTAPI_PORT, help=f"FastAPI port (default: {DEFAULT_FASTAPI_PORT}).")
    # Confidence threshold argument
    parser.add_argument("--confidence-threshold", type=float, default=CONFIDENCE_THRESHOLD, help="Confidence threshold for detection (default: 0.5).")
    # Parse known arguments, ignoring unrecognized ones
    args, _ = parser.parse_known_args()
    # Validate confidence threshold
    if not 0 <= args.confidence_threshold <= 1:
        parser.error("Confidence threshold must be between 0 and 1.")
    return args

def find_available_port(start_port: int, port_range: range, max_attempts: int) -> Optional[int]:
    """
    Find an available port within the specified range.

    Args:
        start_port: Initial port to try.
        port_range: Range of ports to attempt.
        max_attempts: Maximum number of ports to try.

    Returns:
        Available port number, or None if no port is found.
    """
    import socket
    # Check environment variable for port override
    port = int(os.getenv("GRADIO_SERVER_PORT", start_port))
    attempts = 0
    while attempts < max_attempts:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                # Attempt to bind to the port
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
    logger.error(f"No available port in range {min(port_range)}-{max(port_range)}")
    return None

def main() -> None:
    """
    Launch the Gradio UI and optional FastAPI server.

    Raises:
        SystemExit: On interruption or critical errors.
    """
    try:
        # Apply nest_asyncio for compatibility with Jupyter/Colab
        nest_asyncio.apply()
        # Parse command-line arguments
        args = parse_args()
        logger.info(f"Parsed arguments: {args}")
        # Find available port for Gradio
        gradio_port = find_available_port(args.gradio_port, PORT_RANGE, MAX_PORT_ATTEMPTS)
        if gradio_port is None:
            logger.error("Failed to find an available port for Gradio UI")
            sys.exit(1)

        # Start FastAPI server in a thread if enabled
        if args.enable_fastapi:
            logger.info(f"Starting FastAPI on port {args.fastapi_port}")
            fastapi_thread = threading.Thread(
                target=lambda: uvicorn.run(app, host="0.0.0.0", port=args.fastapi_port),
                daemon=True
            )
            fastapi_thread.start()

        # Launch Gradio UI
        logger.info(f"Starting Gradio UI on port {gradio_port}")
        demo = create_gradio_ui()
        demo.launch(server_port=gradio_port, server_name="0.0.0.0")

    except KeyboardInterrupt:
        logger.info("Application terminated by user.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error: {str(e)}\n{traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()