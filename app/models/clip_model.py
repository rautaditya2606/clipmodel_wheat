import torch
import clip
import onnxruntime as ort
import numpy as np
from PIL import Image
from app.core.config import settings

class CLIPProcessor:
    def __init__(self):
        # Determine execution provider
        providers = ['CPUExecutionProvider']
        
        # Load Quantized ONNX models (optimized for 512MB RAM)
        visual_path = "clip_visual_int8.onnx"
        text_path = "clip_text_int8.onnx"
        
        self.visual_session = ort.InferenceSession(visual_path, providers=providers)
        self.text_session = ort.InferenceSession(text_path, providers=providers)
        
        # We still need the CLIP preprocessor/tokenizer logic
        # Using a minimal approach to get dependencies without loading the heavy model
        import clip
        _, self.preprocess = clip.load(settings.CLIP_MODEL_NAME, device="cpu")
        
        # Free up memory by deleting the PyTorch model object immediately
        import gc
        del _
        gc.collect()

    def process_image(self, image: Image.Image, text_descriptions: list[str]):
        """
        Processes a PIL Image object using ONNX Runtime.
        """
        # 1. Preprocess image
        image_input = self.preprocess(image).unsqueeze(0).numpy().astype(np.float32)
        
        # 2. Tokenize text (still using clip.tokenize for simplicity)
        text_tokens = clip.tokenize(text_descriptions).numpy().astype(np.int32)

        # 3. Inference
        image_features = self.visual_session.run(None, {"input": image_input})[0]
        text_features = self.text_session.run(None, {"input": text_tokens})[0]

        # 4. Normalize and calculate similarity
        image_features /= np.linalg.norm(image_features, axis=-1, keepdims=True)
        text_features /= np.linalg.norm(text_features, axis=-1, keepdims=True)
        
        # Calculate cosine similarity and softmax
        logit_scale = 100.0  # Default CLIP logit scale
        similarity = (logit_scale * image_features @ text_features.T)
        
        # Softmax over the results
        exp_similarity = np.exp(similarity - np.max(similarity, axis=-1, keepdims=True))
        probs = exp_similarity / np.sum(exp_similarity, axis=-1, keepdims=True)

        return probs[0]

clip_processor = CLIPProcessor()
