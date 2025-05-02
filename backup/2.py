import gradio as gr
import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
from transformers import YolosImageProcessor, YolosForObjectDetection
from transformers import DetrForSegmentation
from PIL import Image, ImageDraw, ImageStat
import requests
from io import BytesIO
import base64
from collections import Counter
import os
import traceback
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
import uvicorn
import pandas as pd
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ✅ Constants
CONFIDENCE_THRESHOLD = 0.5
# ✅ Supported Models
VALID_MODELS = [
    "facebook/detr-resnet-50",
    "facebook/detr-resnet-101",
    "facebook/detr-resnet-50-panoptic",
    "facebook/detr-resnet-101-panoptic",
    "hustvl/yolos-tiny",
    "hustvl/yolos-base"
]

# ✅ Model Descriptions
MODEL_DESCRIPTIONS = {
    "facebook/detr-resnet-50": "DETR with ResNet-50 backbone for object detection. Fast and accurate for general use.",
    "facebook/detr-resnet-101": "DETR with ResNet-101 backbone for object detection. More accurate but slower than ResNet-50.",
    "facebook/detr-resnet-50-panoptic": "DETR with ResNet-50 for panoptic segmentation. Detects objects and segments scenes.",
    "facebook/detr-resnet-101-panoptic": "DETR with ResNet-101 for panoptic segmentation. High accuracy for complex scenes.",
    "hustvl/yolos-tiny": "YOLOS Tiny model. Lightweight and fast, ideal for resource-constrained environments.",
    "hustvl/yolos-base": "YOLOS Base model. Balances speed and accuracy for object detection."
}

# ✅ Lazy model loading
models = {}
processors = {}

def get_model(model_name):
    """Load or retrieve a model and processor, handling panoptic models correctly."""
    if model_name not in VALID_MODELS:
        raise ValueError(f"Invalid model: {model_name}. Choose from: {VALID_MODELS}")
    
    if model_name not in models:
        logger.info(f"Loading model: {model_name}")
        if "yolos" in model_name:
            models[model_name] = YolosForObjectDetection.from_pretrained(model_name)
            processors[model_name] = YolosImageProcessor.from_pretrained(model_name)
        elif "panoptic" in model_name:
            models[model_name] = DetrForSegmentation.from_pretrained(model_name)
            processors[model_name] = DetrImageProcessor.from_pretrained(model_name)
        else:
            models[model_name] = DetrForObjectDetection.from_pretrained(model_name)
            processors[model_name] = DetrImageProcessor.from_pretrained(model_name)
    return models[model_name], processors[model_name]

# 🧠 Helper function to get image from URL
def get_image_from_url(url):
    """Fetch an image from a URL and convert to RGB."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        raise ValueError(f"Failed to fetch image from URL: {str(e)}")

# 🧠 Get image properties
def get_image_properties(image, file_path=None):
    """Extract properties from an image, handling file size robustly."""
    try:
        file_size = "Unknown"
        if file_path and os.path.exists(file_path):
            file_size = f"{os.path.getsize(file_path) / 1024:.2f} KB"
        elif hasattr(image, "fp") and image.fp is not None:
            # For in-memory images, estimate size
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            file_size = f"{len(buffered.getvalue()) / 1024:.2f} KB"
        
        properties = {
            "format": image.format if hasattr(image, "format") and image.format else "Unknown",
            "size": f"{image.width}x{image.height}",
            "width": image.width,
            "height": image.height,
            "mode": image.mode,
            "aspect_ratio": round(image.width / image.height, 2) if image.height != 0 else "Undefined",
            "info": image.info if hasattr(image, "info") else {},
            "color_stats": get_color_statistics(image),
            "file_size": file_size
        }
        return properties
    except Exception as e:
        logger.error(f"Error getting image properties: {str(e)}")
        return {"error": f"Failed to get image properties: {str(e)}"}

def get_color_statistics(image):
    """Calculate basic color statistics (mean and std deviation) for each channel."""
    try:
        stat = ImageStat.Stat(image)
        return {
            "mean": stat.mean,
            "stddev": stat.stddev
        }
    except Exception as e:
        logger.error(f"Error calculating color statistics: {str(e)}")
        return {"error": str(e)}

# 🧠 Detection Logic
def detect_objects(image, model_name):
    """Perform object detection or panoptic segmentation on an image."""
    try:
        model, processor = get_model(model_name)
        inputs = processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]])

        if "panoptic" in model_name:
            processed_sizes = torch.tensor([[inputs["pixel_values"].shape[2], inputs["pixel_values"].shape[3]]])
            results = processor.post_process_panoptic(outputs, target_sizes=target_sizes, processed_sizes=processed_sizes)[0]
            
            draw = ImageDraw.Draw(image)
            object_names = []
            confidence_scores = []
            object_counter = Counter()

            for segment in results["segments_info"]:
                label = segment["label_id"]
                label_name = model.config.id2label.get(label, "Unknown")
                score = segment.get("score", 1.0)

                if "masks" in results and segment["id"] < len(results["masks"]):
                    mask = results["masks"][segment["id"]].cpu().numpy()
                    if mask.shape[0] > 0 and mask.shape[1] > 0:  # Validate mask
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
            draw = ImageDraw.Draw(image)
            object_names = []
            confidence_scores = []
            object_counter = Counter()

            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                if score > CONFIDENCE_THRESHOLD:
                    x, y, x2, y2 = box.tolist()
                    draw.rectangle([x, y, x2, y2], outline="#32CD32", width=2)
                    label_name = model.config.id2label.get(label.item(), "Unknown")
                    draw.text((x, y), f"{label_name}: {score:.2f}", fill="#32CD32")
                    object_names.append(label_name)
                    confidence_scores.append(float(score))
                    object_counter[label_name] = float(score)

        unique_objects = list(object_counter.keys())
        unique_confidences = [object_counter[obj] for obj in unique_objects]
        image_properties = get_image_properties(image)

        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        img_url = f"data:image/png;base64,{img_base64}"

        return image, object_names, confidence_scores, image_properties, unique_objects, unique_confidences, img_url
    except Exception as e:
        logger.error(f"Error in detect_objects: {str(e)}\n{traceback.format_exc()}")
        raise

# 🎯 Process image for Gradio and FastAPI
def process(image, model_name):
    """Process an image and return results for Gradio and FastAPI."""
    try:
        if model_name not in VALID_MODELS:
            raise ValueError(f"Invalid model: {model_name}. Choose from: {VALID_MODELS}")

        result = detect_objects(image, model_name)
        detected_image, objects, scores, properties, unique_objects, unique_scores, img_url = result

        # Create a more visually appealing properties display in a box-like format
        properties_text = """
        <div style="border: 1px solid #ddd; border-radius: 8px; padding: 15px; background-color: #f9f9f9;">
            <h3 style="margin-top: 0;">Image Properties</h3>
            <table style="width: 100%;">
                <tr><td><b>Format:</b></td><td>{format}</td></tr>
                <tr><td><b>Size:</b></td><td>{size}</td></tr>
                <tr><td><b>Width:</b></td><td>{width}px</td></tr>
                <tr><td><b>Height:</b></td><td>{height}px</td></tr>
                <tr><td><b>Mode:</b></td><td>{mode}</td></tr>
                <tr><td><b>Aspect Ratio:</b></td><td>{aspect_ratio}</td></tr>
                <tr><td><b>File Size:</b></td><td>{file_size}</td></tr>
            </table>
        </div>
        """.format(
            format=properties['format'],
            size=properties['size'],
            width=properties['width'],
            height=properties['height'],
            mode=properties['mode'],
            aspect_ratio=properties['aspect_ratio'],
            file_size=properties['file_size']
        )

        objects_df = pd.DataFrame({
            "Object": objects,
            "Confidence Score": [f"{score:.2f}" for score in scores]
        }) if objects else pd.DataFrame(columns=["Object", "Confidence Score"])

        unique_objects_df = pd.DataFrame({
            "Unique Object": unique_objects,
            "Confidence Score": [f"{score:.2f}" for score in unique_scores]
        }) if unique_objects else pd.DataFrame(columns=["Unique Object", "Confidence Score"])

        gradio_output = (detected_image, objects_df, properties_text, unique_objects_df, "")
        api_output = {
            "image_url": img_url,
            "detected_objects": objects,
            "confidence_scores": [float(score) for score in scores],
            "image_properties": {k: str(v) if not isinstance(v, (int, float, bool, list, dict)) else v for k, v in properties.items()},
            "unique_objects": unique_objects,
            "unique_confidence_scores": [float(score) for score in unique_scores]
        }

        return gradio_output, api_output
    except Exception as e:
        error_msg = f"Error processing image: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        return (
            None,
            pd.DataFrame(columns=["Object", "Confidence Score"]),
            "Error occurred",
            pd.DataFrame(columns=["Unique Object", "Confidence Score"]),
            error_msg
        ), {"error": error_msg, "traceback": traceback.format_exc()}

# 🎯 Process image from URL
def process_image_url(url, model_name):
    """Process an image from a URL."""
    try:
        image = get_image_from_url(url)
        _, api_output = process(image, model_name)
        return api_output
    except Exception as e:
        error_msg = f"Error processing URL: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        return {"error": error_msg, "traceback": traceback.format_exc()}

# 🎯 Clear inputs and outputs
def clear_inputs():
    """Clear Gradio inputs and outputs."""
    return None, None, None, None, ""

# 📚 Generate API documentation
def generate_api_documentation():
    """Generate human-readable API documentation."""
    doc = """
    # 📚 Object Detection API Documentation

    ## Process Endpoint
    
    This API allows you to detect objects in images using state-of-the-art transformer models.

    ### 1. Using Python Client

    ```python
    from gradio_client import Client, handle_file

    client = Client("YOUR_DEPLOYED_APP_URL")
    
    # Process with file upload
    result = client.predict(
        handle_file("path/to/your/image.jpg"),
        "facebook/detr-resnet-50",  # Choose your model
        api_name="/process"
    )
    
    # OR process with URL
    result = client.predict(
        "https://example.com/image.jpg",  # Image URL
        "facebook/detr-resnet-50",  # Choose your model
        api_name="/process_url"
    )
    ```

    ### 2. Using HTTP Requests

    #### POST /detect
    Process an image from file upload or URL.

    **Parameters:**
    - `file`: Image file upload (optional)
    - `image_url`: URL to image (optional, use either this or file)
    - `model_name`: Model to use for detection (choose from: {})

    **Response:**
    ```json
    {{
        "image_url": "base64 encoded image with detections",
        "detected_objects": ["person", "car", ...],
        "confidence_scores": [0.98, 0.87, ...],
        "image_properties": {{
            "format": "JPEG",
            "size": "800x600",
            "width": 800,
            "height": 600,
            "mode": "RGB",
            "aspect_ratio": 1.33,
            "file_size": "125.5 KB"
        }},
        "unique_objects": ["person", "car", ...],
        "unique_confidence_scores": [0.98, 0.87, ...]
    }}
    ```
    """.format(", ".join([f'"{model}"' for model in VALID_MODELS]))
    
    return doc

# ✅ Gradio UI
with gr.Blocks(theme=gr.themes.Default(primary_hue="blue", secondary_hue="gray")) as demo:
    gr.Markdown(
        """
        # 🚀 Object Detection App
        Upload an image or provide a URL to detect objects using state-of-the-art transformer models (DETR, YOLOS).
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
                        info="Choose a model for object detection or panoptic segmentation."
                    )
                    model_info = gr.Markdown(
                        f"**Model Info**: {MODEL_DESCRIPTIONS[VALID_MODELS[0]]}",
                        visible=True
                    )
                    image_input = gr.Image(type="pil", label="📷 Upload Image")
                    with gr.Row():
                        submit_btn = gr.Button("✨ Detect", variant="primary")
                        clear_btn = gr.Button("🗑️ Clear", variant="secondary")
                    
                    def update_model_info(model_name):
                        return f"**Model Info**: {MODEL_DESCRIPTIONS.get(model_name, 'No description available.')}"
                    model_choice.change(
                        fn=update_model_info,
                        inputs=model_choice,
                        outputs=model_info
                    )
                
                with gr.Column(scale=2):
                    gr.Markdown("### Results")
                    error_output = gr.Textbox(
                        label="⚠️ Errors",
                        visible=False,
                        lines=3,
                        max_lines=5
                    )
                    output_image = gr.Image(
                        type="pil",
                        label="🎯 Detected Image",
                        interactive=False
                    )
                    with gr.Row():
                        objects_output = gr.DataFrame(
                            label="📋 Detected Objects",
                            interactive=False
                        )
                        unique_objects_output = gr.DataFrame(
                            label="🔍 Unique Objects",
                            interactive=False
                        )
                    properties_output = gr.HTML(
                        label="📄 Image Properties"
                    )
        
        with gr.Tab("🔗 URL Input"):
            gr.Markdown("### Process Image from URL")
            image_url_input = gr.Textbox(
                label="🔗 Image URL",
                placeholder="https://example.com/image.jpg"
            )
            url_model_choice = gr.Dropdown(
                choices=VALID_MODELS,
                value=VALID_MODELS[0],
                label="🔎 Select Model"
            )
            url_model_info = gr.Markdown(
                f"**Model Info**: {MODEL_DESCRIPTIONS[VALID_MODELS[0]]}",
                visible=True
            )
            url_submit_btn = gr.Button("🔄 Process URL", variant="primary")
            url_output = gr.JSON(label="API Response")
            
            url_model_choice.change(
                fn=update_model_info,
                inputs=url_model_choice,
                outputs=url_model_info
            )
        
        with gr.Tab("ℹ️ Help"):
            gr.Markdown(
                """
                ## How to Use
                - **Image Upload**: Select a model, upload an image, and click "Detect" to see detected objects and image properties.
                - **URL Input**: Enter an image URL, select a model, and click "Process URL" to get results in JSON format.
                - **Models**: Choose from DETR (object detection or panoptic segmentation) or YOLOS (lightweight detection).
                - **Clear**: Reset all inputs and outputs using the "Clear" button.
                - **Errors**: Check the error box for any processing issues.
                
                ## Tips
                - Use high-quality images for better detection results.
                - Panoptic models (e.g., DETR-ResNet-50-panoptic) provide segmentation masks for complex scenes.
                - For faster processing, try YOLOS-Tiny on resource-constrained devices.
                """
            )
        
        with gr.Tab("📚 API Documentation"):
            api_docs = gr.Markdown(generate_api_documentation())
        
        with gr.Tab("🔌 Use via API"):
            gr.Markdown(
                """
                ## Process Endpoint

                Use this endpoint to process an image using our object detection models:

                ```python
                from gradio_client import Client, handle_file

                client = Client("YOUR_DEPLOYED_APP_URL")
                
                # Process with image file
                result = client.predict(
                    handle_file("path/to/your/image.jpg"),  # Your image file
                    "facebook/detr-resnet-50",              # Choose a model
                    api_name="/process"                     # Endpoint name
                )
                print(result)
                ```

                Click the "Process Here" button below to try it out directly:
                """
            )
            with gr.Row():
                api_image_input = gr.Image(type="pil", label="📷 Upload Image")
                api_model_choice = gr.Dropdown(
                    choices=VALID_MODELS,
                    value=VALID_MODELS[0],
                    label="🔎 Select Model"
                )
            api_process_btn = gr.Button("🔄 Process Here", variant="primary")
            api_result = gr.JSON(label="API Response")
    
    # Handle Image Upload tab events
    submit_btn.click(
        fn=lambda image, model_name: process(image, model_name)[0],
        inputs=[image_input, model_choice],
        outputs=[
            output_image,
            objects_output,
            properties_output,
            unique_objects_output,
            error_output
        ]
    )
    
    clear_btn.click(
        fn=clear_inputs,
        inputs=None,
        outputs=[
            image_input,
            output_image,
            objects_output,
            unique_objects_output,
            error_output
        ]
    )
    
    # Handle URL Input tab events
    url_submit_btn.click(
        fn=process_image_url,
        inputs=[image_url_input, url_model_choice],
        outputs=[url_output]
    )
    
    # Handle API tab events
    api_process_btn.click(
        fn=lambda image, model_name: process(image, model_name)[1],
        inputs=[api_image_input, api_model_choice],
        outputs=[api_result]
    )

# ✅ FastAPI Setup
app = FastAPI(title="Object Detection API")

@app.post("/detect")
async def detect_objects_endpoint(
    file: UploadFile = File(None),
    image_url: str = Form(None),
    model_name: str = Form(VALID_MODELS[0])
):
    """FastAPI endpoint to detect objects in an image from file or URL."""
    try:
        if (file is None and not image_url) or (file is not None and image_url):
            raise HTTPException(status_code=400, detail="Provide either an image file or an image URL, but not both.")

        if file:
            if not file.content_type.startswith("image/"):
                raise HTTPException(status_code=400, detail="File must be an image")
            contents = await file.read()
            image = Image.open(BytesIO(contents)).convert("RGB")
        else:
            image = get_image_from_url(image_url)

        if model_name not in VALID_MODELS:
            raise HTTPException(status_code=400, detail=f"Invalid model. Choose from: {VALID_MODELS}")

        _, api_output = process(image, model_name)
        return JSONResponse(content=api_output)
    except Exception as e:
        logger.error(f"Error in FastAPI endpoint: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

# 🌍 Launch Gradio app for local execution
if __name__ == "__main__":
    demo.launch()
    # To run FastAPI, use: uvicorn object_detection:app --host 0.0.0.0 --port 8000
    # uvicorn.run(app, host="0.0.0.0", port=8000)