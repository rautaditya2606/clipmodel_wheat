import torch
import clip
from PIL import Image
import requests
from io import BytesIO
from app.core.config import settings

class CLIPProcessor:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(settings.CLIP_MODEL_NAME, device=self.device)

    def process_image(self, image: Image.Image, text_descriptions: list[str]):
        """
        Processes a PIL Image object and compares it against text descriptions.
        """
        # Preprocess image and tokenize text
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        text_tokens = clip.tokenize(text_descriptions).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            text_features = self.model.encode_text(text_tokens)

            # Calculate similarity (cosine similarity scaled)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

        return similarity.cpu().numpy()[0]

clip_processor = CLIPProcessor()
