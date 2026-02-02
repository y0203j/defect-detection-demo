import base64
import io
from io import BytesIO
import torch
import numpy as np
import segmentation_models_pytorch as smp
import albumentations as albu
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw

#initialize the API
app = FastAPI(title="Defect Detector")

#serve static UI under /static and an index at /
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", include_in_schema=False)
def root():
    return FileResponse("static/index.html")

MODEL_PATH = "defect_model_float32.pth"
ENCODER = 'mobilenet_v2'
ENCODER_WEIGHTS = 'imagenet'
DEVICE = 'cpu'

#load model safely
model = None
try:
    model = torch.load(MODEL_PATH, map_location=DEVICE,weights_only=False)
    model.eval()
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    model = None

#preprocessing
preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

def preprocess_image(image_bytes):
    """Return (tensor, original PIL image)."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_np = np.array(image)

    transform = albu.Compose([
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=lambda x, **k: x.transpose(2, 0, 1).astype('float32'))
    ])

    processed = transform(image=image_np)['image']
    tensor = torch.from_numpy(processed).unsqueeze(0)
    return tensor, image

#API endpoint
@app.post("/predict")
async def predict_defect(file: UploadFile = File(...)):
    """
    Detect defects; returns JSON with overlay image (data URL) and bbox.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    image_bytes = await file.read()
    input_tensor, orig_image = preprocess_image(image_bytes)
    input_tensor = input_tensor.to(DEVICE)

    #inference
    with torch.no_grad():
        logits = model(input_tensor)
        probs = logits.sigmoid().cpu().squeeze().numpy()

    # normalize shape to 2D mask
    if probs.ndim == 3:
        probs = probs[0]

    max_defect_prob = float(np.max(probs))
    mask_bool = probs > 0.5
    defect_area_ratio = float(mask_bool.mean())
    is_defective = max_defect_prob > 0.5

    # bounding box in mask coords
    ys, xs = np.where(mask_bool)
    if len(xs) == 0 or len(ys) == 0:
        bbox = None
    else:
        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
        bbox = [x_min, y_min, x_max, y_max]

    # create overlay PNG (red translucent mask + bbox)
    data_url = None
    try:
        overlay = orig_image.convert('RGBA')
        mask_img = Image.fromarray((mask_bool * 255).astype('uint8'))

        # resize mask to match overlay if necessary and scale bbox
        mask_w, mask_h = mask_img.size
        overlay_w, overlay_h = overlay.size
        if (mask_w, mask_h) != (overlay_w, overlay_h):
            scale_x = overlay_w / float(mask_w)
            scale_y = overlay_h / float(mask_h)
            mask_img = mask_img.resize(overlay.size, resample=Image.NEAREST)
            if bbox is not None:
                x_min_s = int(bbox[0] * scale_x)
                y_min_s = int(bbox[1] * scale_y)
                x_max_s = int(bbox[2] * scale_x)
                y_max_s = int(bbox[3] * scale_y)
                bbox = [x_min_s, y_min_s, x_max_s, y_max_s]

        mask_pixels = mask_img.convert('L')
        red_layer = Image.new('RGBA', overlay.size, (255, 0, 0, 100))
        mask_rgba = Image.new('RGBA', overlay.size, (255, 0, 0, 0))
        mask_rgba.paste(red_layer, (0, 0), mask_pixels)
        composed = Image.alpha_composite(overlay, mask_rgba)

        if bbox is not None:
            draw = ImageDraw.Draw(composed)
            draw.rectangle(bbox, outline=(255, 255, 0, 255), width=3)

        buf = BytesIO()
        composed.convert('RGB').save(buf, format='PNG')
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode('utf-8')
        data_url = f"data:image/png;base64,{encoded}"
    except Exception:
        data_url = None

    return {
        "filename": file.filename,
        "is_defective": bool(is_defective),
        "max_confidence": round(max_defect_prob, 4),
        "defect_area_percentage": round(defect_area_ratio * 100, 4),
        "bbox": bbox,
        "mask_image": data_url
    }

@app.get("/health")
def health():
    loaded = model is not None
    if not loaded:
        return JSONResponse(status_code=503, content={"status": "error", "model_loaded": False})
    return {"status": "ok", "model_loaded": True}
