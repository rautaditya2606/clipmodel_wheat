import torch
import clip
from onnxruntime import InferenceSession
import numpy as np
from PIL import Image
import os

def export_clip_to_onnx():
    print("Loading CLIP model...")
    device = "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()

    # 1. Export Image Encoder
    print("Exporting Image Encoder...")
    dummy_image = torch.randn(1, 3, 224, 224).to(device)
    torch.onnx.export(
        model.visual,
        dummy_image,
        "clip_visual.onnx",
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )

    # 2. Export Text Encoder
    print("Exporting Text Encoder...")
    # Representative text token shape
    dummy_text = clip.tokenize(["a photo of a field"]).to(device)
    
    # We need a wrapper because model.encode_text is a method
    class TextEncoderWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, text):
            return self.model.encode_text(text)

    text_wrapper = TextEncoderWrapper(model)
    torch.onnx.export(
        text_wrapper,
        dummy_text,
        "clip_text.onnx",
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print("Export complete.")

if __name__ == "__main__":
    export_clip_to_onnx()
