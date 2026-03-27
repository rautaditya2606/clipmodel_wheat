import torch
import clip
from PIL import Image
import numpy as np
from app.core.config import settings
import gc

class CLIPProcessor:
    def __init__(self):
        self.device = "cpu"
        # Load the model and preprocessing function
        self.model, self.preprocess = clip.load(settings.CLIP_MODEL_NAME, device=self.device)
        self.model.eval()

    def process_image(self, image: Image.Image, text_descriptions: list[str]):
        """
        Processes a PIL Image object using the official PyTorch CLIP model.
        """
        try:
            with torch.no_grad():
                # 1. Preprocess image
                image_input = self.preprocess(image).unsqueeze(0).to(self.device)
                
                # 2. Tokenize text
                text_tokens = clip.tokenize(text_descriptions).to(self.device)

                # 3. Inference
                logits_per_image, logits_per_text = self.model(image_input, text_tokens)
                
                # 4. Calculate probabilities (softmax)
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()

                return probs[0].tolist()
        finally:
            # Try to keep memory as clean as possible
            gc.collect()

clip_processor = CLIPProcessor()
