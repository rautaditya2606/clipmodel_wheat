import os
import psutil
import torch
import clip
import numpy as np
import onnxruntime as ort
from app.models.clip_model import CLIPProcessor
from PIL import Image

def get_process_memory():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 * 1024)  # Convert to MB

def measure_memory_usage():
    print(f"--- Memory Usage Tracker ---")
    
    # Baseline
    baseline = get_process_memory()
    print(f"1. Baseline Memory: {baseline:.2f} MB")
    
    # Initializing ONNX Processor
    print("\n2. Initializing ONNX CLIPProcessor...")
    processor = CLIPProcessor()
    after_init = get_process_memory()
    print(f"   Memory after init: {after_init:.2f} MB (+{after_init - baseline:.2f} MB)")
    
    # Running one inference
    print("\n3. Running inference with ONNX...")
    dummy_image = Image.new('RGB', (224, 224), color='white')
    labels = ["a photo of wheat", "a crop field"]
    processor.process_image(dummy_image, labels)
    after_inference = get_process_memory()
    print(f"   Memory after inference: {after_inference:.2f} MB (+{after_inference - after_init:.2f} MB)")
    
    # Showing total footprint
    print(f"\nTotal process footprint: {after_inference:.2f} MB")
    
    if after_inference > 512:
        print("\nWARNING: Current usage exceeds Render's 512MB limit!")
    else:
        print("\nSUCCESS: Current usage is within Render's 512MB limit.")

if __name__ == "__main__":
    measure_memory_usage()
