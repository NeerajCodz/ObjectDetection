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
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
import uvicorn
import pandas as pd
import traceback
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Constants
CONFIDENCE_THRESHOLD = 0.5
VALID_MODELS = [
    "facebook/detr-resnet-50",
    "facebook/detr-resnet-101",
    "facebook/detr-resnet-50-panoptic",
    "facebook/detr-resnet-101-panoptic",
    "hustvl/yolos-tiny",
    "hustvl/yolos-base"
]
MODEL_DESCRIPTIONS = {
    "facebook/detr-resnet-50": "DETR with ResNet-50 backbone for object detection. Fast and accurate for general use.",
    "facebook/detr-resnet-101": "DETR with ResNet-101 backbone for object detection. More accurate but slower than ResNet-50.",
    "facebook/detr-resnet-50-panoptic": "DETR with ResNet-50 for panoptic segmentation. Detects objects and segments scenes.",
    "facebook/detr-resnet-101-panoptic": "DETR with ResNet-101 for panoptic segmentation. High accuracy for complex scenes.",
    "hustvl/yolos-tiny": "YOLOS Tiny model. Lightweight and fast, ideal for resource-constrained environments.",
    "hustvl/yolos-base": "YOLOS Base model. Balances speed and accuracy for object detection."
}

# Lazy model loading
models = {}
processors = {}

def process(image, model_name):
    """Process an image and return detected image, objects, confidences, unique objects, unique confidences, and properties."""
    try:
        if model_name not in VALID_MODELS:
            raise ValueError(f"Invalid model: {model_name}. Choose from: {VALID_MODELS}")

        # Load model and processor
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

        model, processor = models[model_name], processors[model_name]
        inputs = processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]])
        draw = ImageDraw.Draw(image)
        object_names = []
        confidence_scores = []
        object_counter = Counter()

        # Load a larger font
        try:
            from PIL import ImageFont
            font = ImageFont.truetype("arial.ttf", 18)  # Increased font size to 18pt
        except:
            font = None  # Fallback to default font if arial.ttf is unavailable

        if "panoptic" in model_name:
            processed_sizes = torch.tensor([[inputs["pixel_values"].shape[2], inputs["pixel_values"].shape[3]]])
            results = processor.post_process_panoptic(outputs, target_sizes=target_sizes, processed_sizes=processed_sizes)[0]

            for segment in results["segments_info"]:
                label = segment["label_id"]
                label_name = model.config.id2label.get(label, "Unknown")
                score = segment.get("score", 1.0)

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
                    # Object name on top-right, confidence on top-left
                    name_text = label_name
                    conf_text = f"{score:.2f}"
                    # Get text dimensions
                    name_bbox = draw.textbbox((0, 0), name_text, font=font) if font else draw.textbbox((0, 0), name_text)
                    conf_bbox = draw.textbbox((0, 0), conf_text, font=font) if font else draw.textbbox((0, 0), conf_text)
                    name_width, name_height = name_bbox[2] - name_bbox[0], name_bbox[3] - name_bbox[1]
                    conf_width, conf_height = conf_bbox[2] - conf_bbox[0], conf_bbox[3] - conf_bbox[1]
                    # Draw text: name on top-right, confidence on top-left
                    draw.text((x2 - name_width - 2, y - name_height - 2), name_text, fill="#32CD32", font=font)
                    draw.text((x + 2, y - conf_height - 2), conf_text, fill="#32CD32", font=font)
                    object_names.append(label_name)
                    confidence_scores.append(float(score))
                    object_counter[label_name] = float(score)

        unique_objects = list(object_counter.keys())
        unique_confidences = [object_counter[obj] for obj in unique_objects]

        # Image properties
        file_size = "Unknown"
        if hasattr(image, "fp") and image.fp is not None:
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            file_size = f"{len(buffered.getvalue()) / 1024:.2f} KB"
        
        # Color statistics
        try:
            stat = ImageStat.Stat(image)
            color_stats = {
                "mean": [f"{m:.2f}" for m in stat.mean],
                "stddev": [f"{s:.2f}" for s in stat.stddev]
            }
        except Exception as e:
            logger.error(f"Error calculating color statistics: {str(e)}")
            color_stats = {"mean": "Error", "stddev": "Error"}

        properties = {
            "Format": image.format if hasattr(image, "format") and image.format else "Unknown",
            "Size": f"{image.width}x{image.height}",
            "Width": f"{image.width} px",
            "Height": f"{image.height} px",
            "Mode": image.mode,
            "Aspect Ratio": f"{round(image.width / image.height, 2) if image.height != 0 else 'Undefined'}",
            "File Size": file_size,
            "Mean (R,G,B)": ", ".join(color_stats["mean"]) if isinstance(color_stats["mean"], list) else color_stats["mean"],
            "StdDev (R,G,B)": ", ".join(color_stats["stddev"]) if isinstance(color_stats["stddev"], list) else color_stats["stddev"]
        }

        return image, object_names, confidence_scores, unique_objects, unique_confidences, properties
    except Exception as e:
        logger.error(f"Error in process: {str(e)}\n{traceback.format_exc()}")
        raise

# FastAPI Setup
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
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert("RGB")

        if model_name not in VALID_MODELS:
            raise HTTPException(status_code=400, detail=f"Invalid model. Choose from: {VALID_MODELS}")

        detected_image, detected_objects, detected_confidences, unique_objects, unique_confidences, _ = process(image, model_name)

        buffered = BytesIO()
        detected_image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        img_url = f"data:image/png;base64,{img_base64}"

        return JSONResponse(content={
            "image_url": img_url,
            "detected_objects": detected_objects,
            "confidence_scores": detected_confidences,
            "unique_objects": unique_objects,
            "unique_confidence_scores": unique_confidences
        })
    except Exception as e:
        logger.error(f"Error in FastAPI endpoint: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

# Gradio UI
def create_gradio_ui():
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
                        image_url_input = gr.Textbox(
                            label="🔗 Image URL",
                            placeholder="https://example.com/image.jpg"
                        )
                        with gr.Row():
                            submit_btn = gr.Button("✨ Detect", variant="primary")
                            clear_btn = gr.Button("🗑️ Clear", variant="secondary")
                        
                        model_choice.change(
                            fn=lambda model_name: f"**Model Info**: {MODEL_DESCRIPTIONS.get(model_name, 'No description available.')}",
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
                                interactive=False,
                                value=None
                            )
                            unique_objects_output = gr.DataFrame(
                                label="🔍 Unique Objects",
                                interactive=False,
                                value=None
                            )
                        properties_output = gr.DataFrame(
                            label="📄 Image Properties",
                            interactive=False,
                            value=None
                        )
                
                def process_for_gradio(image, url, model_name):
                    try:
                        if image is None and not url:
                            return None, None, None, None, "Please provide an image or URL"
                        if image and url:
                            return None, None, None, None, "Please provide either an image or URL, not both"
                        
                        if url:
                            response = requests.get(url, timeout=10)
                            response.raise_for_status()
                            image = Image.open(BytesIO(response.content)).convert("RGB")
                        
                        detected_image, objects, scores, unique_objects, unique_scores, properties = process(image, model_name)
                        objects_df = pd.DataFrame({
                            "Object": objects,
                            "Confidence Score": [f"{score:.2f}" for score in scores]
                        }) if objects else pd.DataFrame(columns=["Object", "Confidence Score"])
                        unique_objects_df = pd.DataFrame({
                            "Unique Object": unique_objects,
                            "Confidence Score": [f"{score:.2f}" for score in unique_scores]
                        }) if unique_objects else pd.DataFrame(columns=["Unique Object", "Confidence Score"])
                        properties_df = pd.DataFrame([properties]) if properties else pd.DataFrame(columns=properties.keys())
                        return detected_image, objects_df, unique_objects_df, properties_df, ""
                    except Exception as e:
                        error_msg = f"Error processing image: {str(e)}"
                        logger.error(f"{error_msg}\n{traceback.format_exc()}")
                        return None, None, None, None, error_msg

                submit_btn.click(
                    fn=process_for_gradio,
                    inputs=[image_input, image_url_input, model_choice],
                    outputs=[output_image, objects_output, unique_objects_output, properties_output, error_output]
                )
                
                clear_btn.click(
                    fn=lambda: [None, "", None, None, None, None],
                    inputs=None,
                    outputs=[image_input, image_url_input, output_image, objects_output, unique_objects_output, properties_output, error_output]
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
                    fn=lambda model_name: f"**Model Info**: {MODEL_DESCRIPTIONS.get(model_name, 'No description available.')}",
                    inputs=url_model_choice,
                    outputs=url_model_info
                )
                
                def process_url_for_gradio(url, model_name):
                    try:
                        response = requests.get(url, timeout=10)
                        response.raise_for_status()
                        image = Image.open(BytesIO(response.content)).convert("RGB")
                        detected_image, objects, scores, unique_objects, unique_scores, _ = process(image, model_name)
                        buffered = BytesIO()
                        detected_image.save(buffered, format="PNG")
                        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                        return {
                            "image_url": f"data:image/png;base64,{img_base64}",
                            "detected_objects": objects,
                            "confidence_scores": scores,
                            "unique_objects": unique_objects,
                            "unique_confidence_scores": unique_scores
                        }
                    except Exception as e:
                        error_msg = f"Error processing URL: {str(e)}"
                        logger.error(f"{error_msg}\n{traceback.format_exc()}")
                        return {"error": error_msg}

                url_submit_btn.click(
                    fn=process_url_for_gradio,
                    inputs=[image_url_input, url_model_choice],
                    outputs=[url_output]
                )
            
            with gr.Tab("ℹ️ Help"):
                gr.Markdown(
                    """
                    ## How to Use
                    - **Image Upload**: Select a model, upload an image or provide a URL, and click "Detect" to see detected objects and image properties.
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

    return demo

if __name__ == "__main__":
    demo = create_gradio_ui()
    demo.launch()
    # To run FastAPI, use: uvicorn object_detection:app --host 0.0.0.0 --port 8000