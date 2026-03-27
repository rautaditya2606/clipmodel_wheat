from fastapi import FastAPI, UploadFile, File, HTTPException
from app.core.config import settings
from app.models.clip_model import clip_processor
from PIL import Image
from io import BytesIO
import json

app = FastAPI(title=settings.PROJECT_NAME)

@app.post("/verify-crop/")
async def verify_crop(file: UploadFile = File(...)):
    """
    Endpoint for uploading an image and verifying if it contains wheat or crops.
    """
    try:
        # 1. Read file into memory and convert to PIL Image
        contents = await file.read()
        image = Image.open(BytesIO(contents))

        # 2. Define whitelist/allowed labels for wheat and crops
        allowed_labels = ["a photo of wheat", "a crop field", "a field of wheat", "wheat plants", "agriculture crop"]
        # Define exclusion/negative labels to better distinguish
        exclusion_labels = ["a city street", "a person", "an animal", "an indoor room", "electronic gadgets", "forest without wheat"]
        
        all_labels = allowed_labels + exclusion_labels
        
        # 3. CLIP verification (Processes Image object directly)
        clip_results = clip_processor.process_image(image, all_labels)
        
        # Get scores for allowed labels
        allowed_scores = clip_results[:len(allowed_labels)]
        max_allowed_score = float(max(allowed_scores))
        
        # Get the label with the highest overall score
        top_label_index = clip_results.argmax()
        is_allowed_top = top_label_index < len(allowed_labels)

        # 4. Decision logic (Flagging)
        if not is_allowed_top or max_allowed_score < (settings.REJECTION_THRESHOLD / 100):
            return {
                "status": "rejected",
                "is_crop": False,
                "message": "Image does not appear to be wheat or a crop field.",
                "top_detected_category": all_labels[top_label_index],
                "confidence": float(clip_results[top_label_index])
            }

        return {
            "status": "accepted",
            "is_crop": True,
            "message": "Image verified as wheat or crop.",
            "detected_category": all_labels[top_label_index],
            "confidence": max_allowed_score
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.get("/")
def home():
    return {"message": "CLIP Image Verification API"}
