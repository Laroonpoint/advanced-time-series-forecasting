import numpy as np
import matplotlib.pyplot as plt
import json
import os

results = json.load(open("results.json"))
print("Results:", results)
# load attention weights if exist
if os.path.exists("attention_weights.npy"):
    w = np.load("attention_weights.npy")
    plt.figure(figsize=(6,4))
    plt.imshow(w[0], aspect='auto')
    plt.title("Attention weights (first head/view)")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("attention_weights.png")
    print("Saved attention_weights.png")
else:
    print("No attention weights found.")
