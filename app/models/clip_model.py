import onnxruntime as ort
import numpy as np
from PIL import Image
from app.core.config import settings
import gzip
import os
import ftfy
import regex as re

# --- MINIMAL CLIP TOKENIZER ---
# To avoid importing 'clip' and its heavy 'torch' dependency
class SimpleTokenizer:
    def __init__(self, bpe_path):
        self.byte_encoder = self.bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        merges = gzip.open(bpe_path).read().decode("utf-8").split('\n')
        merges = merges[1:49152-256-2+1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(self.byte_encoder.values())
        vocab = vocab + [v+'</w>' for v in vocab]
        for merge in merges:
            vocab.append(''.join(merge))
        vocab.extend(['<|startoftext|>', '<|endoftext|>'])
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {'<|startoftext|>': '<|startoftext|>', '<|endoftext|>': '<|endoftext|>'}
        self.pat = re.compile(r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]+|[^\s\p{L}\p{N}]+""", re.IGNORECASE)

    def bytes_to_unicode(self):
        bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
        cs = bs[:]
        n = 0
        for b in range(2**8):
            if b not in bs:
                bs.append(b)
                cs.append(2**8 + n)
                n += 1
        return dict(zip(bs, [chr(c) for c in cs]))

    def tokenize(self, texts, context_length=77):
        if isinstance(texts, str): texts = [texts]
        all_tokens = []
        for text in texts:
            text = ftfy.fix_text(text).lower()
            text = " ".join(text.split())
            tokens = [self.encoder['<|startoftext|>']]
            for token in re.findall(self.pat, text):
                tokens.extend([self.encoder[b] for b in self.bpe(token)])
            tokens = tokens[:context_length-1]
            tokens.append(self.encoder['<|endoftext|>'])
            tokens += [0] * (context_length - len(tokens))
            all_tokens.append(tokens)
        return np.array(all_tokens, dtype=np.int32)

    def bpe(self, token):
        if token in self.cache: return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + '</w>',)
        pairs = self.get_pairs(word)
        if not pairs: return token + '</w>'
        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks: break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break
                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1: break
            pairs = self.get_pairs(word)
        self.cache[token] = word
        return word

    def get_pairs(self, word):
        return set(zip(word[:-1], word[1:]))

class CLIPProcessor:
    def __init__(self):
        providers = ['CPUExecutionProvider']
        
        # Load Quantized ONNX models (optimized for 512MB RAM)
        visual_path = "clip_visual_int8.onnx"
        text_path = "clip_text_int8.onnx"
        
        self.visual_session = ort.InferenceSession(visual_path, providers=providers)
        self.text_session = ort.InferenceSession(text_path, providers=providers)
        
        # Load standalone tokenizer
        # Path to bpe vocab inside the models folder
        vocab_path = os.path.join(os.path.dirname(__file__), "bpe_simple_vocab_16e6.txt.gz")
        if not os.path.exists(vocab_path):
            import requests
            print(f"Downloading CLIP vocabulary to {vocab_path}...")
            url = "https://github.com/openai/CLIP/raw/main/clip/bpe_simple_vocab_16e6.txt.gz"
            r = requests.get(url)
            with open(vocab_path, 'wb') as f: f.write(r.content)
            
        self.tokenizer = SimpleTokenizer(vocab_path)

    def preprocess_image(self, image: Image.Image):
        # Manual CLIP Preprocessing (Resize + Center Crop + Normalize)
        w, h = image.size
        # Resize shortest side to 224
        if w < h:
            new_w, new_h = 224, int(h * 224 / w)
        else:
            new_w, new_h = int(w * 224 / h), 224
        image = image.resize((new_w, new_h), resample=Image.BICUBIC)
        
        # Center crop 224x224
        left = (new_w - 224) / 2
        top = (new_h - 224) / 2
        right = (new_w + 224) / 2
        bottom = (new_h + 224) / 2
        image = image.crop((left, top, right, bottom))
        
        # Normalize
        image_np = np.array(image.convert("RGB")).astype(np.float32) / 255.0
        mean = np.array([0.48145466, 0.4578275, 0.40821073])
        std = np.array([0.26862954, 0.26130258, 0.27577711])
        image_np = (image_np - mean) / std
        return np.transpose(image_np, (2, 0, 1))

    def process_image(self, image: Image.Image, text_descriptions: list[str]):
        """
        Processes a PIL Image object using ONNX Runtime.
        """
        # 1. Preprocess
        image_input = self.preprocess_image(image)[np.newaxis, ...].astype(np.float32)
        text_tokens = self.tokenizer.tokenize(text_descriptions)

        # 2. Inference
        image_features = self.visual_session.run(None, {"input": image_input})[0]
        text_features = self.text_session.run(None, {"input": text_tokens})[0]

        # 3. Normalize and calculate similarity
        image_features /= np.linalg.norm(image_features, axis=-1, keepdims=True)
        text_features /= np.linalg.norm(text_features, axis=-1, keepdims=True)
        
        logit_scale = 100.0  # Default CLIP logit scale
        similarity = (logit_scale * image_features @ text_features.T)
        
        # Softmax over result
        exp_similarity = np.exp(similarity - np.max(similarity, axis=-1, keepdims=True))
        probs = exp_similarity / np.sum(exp_similarity, axis=-1, keepdims=True)

        return probs[0].tolist()

clip_processor = CLIPProcessor()
