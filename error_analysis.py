import mynn as nn
import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'best_models_new', 'best_model.pickle')
MODEL_TYPE = 'MLP'  # 'MLP' or 'CNN'

test_images_path = os.path.join(BASE_DIR, 'dataset', 'MNIST', 't10k-images-idx3-ubyte.gz')
test_labels_path = os.path.join(BASE_DIR, 'dataset', 'MNIST', 't10k-labels-idx1-ubyte.gz')

with gzip.open(test_images_path, 'rb') as f:
    magic, num, rows, cols = unpack('>4I', f.read(16))
    imgs_flat = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)
with gzip.open(test_labels_path, 'rb') as f:
    magic, num = unpack('>2I', f.read(8))
    labs = np.frombuffer(f.read(), dtype=np.uint8)

imgs_flat = imgs_flat.astype(np.float32) / 255.0

if MODEL_TYPE == 'MLP':
    model = nn.models.Model_MLP()
    model.load_model(MODEL_PATH)
    logits = model(imgs_flat.reshape(-1, 784))
else:
    model = nn.models.Model_CNN()
    model.load_model(MODEL_PATH)
    logits = model(imgs_flat.reshape(-1, 1, 28, 28))

preds = np.argmax(logits, axis=1)
acc = (preds == labs).mean()
print(f"Test accuracy: {acc:.4f}")

cm = np.zeros((10, 10), dtype=int)
for t, p in zip(labs, preds):
    cm[t, p] += 1

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.matshow(cm, cmap='Blues')
for i in range(10):
    for j in range(10):
        ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                fontsize=8, color='white' if cm[i, j] > cm.max() / 2 else 'black')
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_xticks(range(10))
ax.set_yticks(range(10))
ax.set_title(f'{MODEL_TYPE} Confusion Matrix (acc={acc:.4f})')
plt.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout()
cm_path = os.path.join(BASE_DIR, 'figs', f'{MODEL_TYPE.lower()}_confusion_matrix.png')
plt.savefig(cm_path, dpi=150)
plt.show()
print(f"Confusion matrix saved to: {cm_path}")

wrong_idx = np.where(preds != labs)[0]
n_show = min(25, len(wrong_idx))
fig, axes = plt.subplots(5, 5, figsize=(8, 8))
fig.suptitle(f'{MODEL_TYPE} Misclassified Examples (true -> pred)', fontsize=14)
for k in range(n_show):
    ax = axes[k // 5, k % 5]
    idx = wrong_idx[k]
    ax.imshow(imgs_flat[idx], cmap='gray')
    ax.set_title(f'{labs[idx]}->{preds[idx]}', fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
for k in range(n_show, 25):
    axes[k // 5, k % 5].axis('off')
plt.tight_layout()
mis_path = os.path.join(BASE_DIR, 'figs', f'{MODEL_TYPE.lower()}_misclassified.png')
plt.savefig(mis_path, dpi=150)
plt.show()
print(f"Misclassified examples saved to: {mis_path}")
