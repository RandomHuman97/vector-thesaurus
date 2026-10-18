import numpy as np
from sentence_transformers import SentenceTransformer

model_name = "BAAI/bge-small-en-v1.5"
print(f"loading model: {model_name}...")
model = SentenceTransformer(model_name)

print("loading vocabulary from system dictionary...")
with open("/etc/dictionaries-common/words", "r", encoding="utf-8") as file:
    vocabulary = [line.strip() for line in file if line.strip()]

print(f"encoding {len(vocabulary)} words/phrases...")

#generate embeddings
embeddings = model.encode(
    vocabulary,
    normalize_embeddings=True,
    show_progress_bar=True,
    batch_size=256
)

embeddings = embeddings.astype(np.float32)

print(f"embedding shape: {embeddings.shape}")

bin_filename = "embeddings.bin"
embeddings.tofile(bin_filename)
print(f"saved binary vectors to {bin_filename}")

vocab_filename = "vocab.txt"
with open(vocab_filename, "w", encoding="utf-8") as f:
    for word in vocabulary:
        f.write(f"{word}\n")
print(f"saved vocabulary list to {vocab_filename}")
