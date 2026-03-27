import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    PROJECT_NAME: str = "CLIP FastAPI Image Rejection System"
    CLOUDINARY_CLOUD_NAME: str = os.getenv("CLOUDINARY_CLOUD_NAME")
    CLOUDINARY_API_KEY: str = os.getenv("CLOUDINARY_API_KEY")
    CLOUDINARY_API_SECRET: str = os.getenv("CLOUDINARY_API_SECRET")
    PORT: int = int(os.getenv("PORT", 8000))
    CLIP_MODEL_NAME: str = "ViT-B/32" # CLIP model size
    REJECTION_THRESHOLD: float = 35.0 # Increased for more specific wheat/crop filtering

settings = Settings()
