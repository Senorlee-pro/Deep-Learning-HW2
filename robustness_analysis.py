import mynn as nn
import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import os
from scipy.ndimage import rotate, shift

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MLP_MODEL_PATH = os.path.join(BASE_DIR, 'best_models_new', 'best_model.pickle')
CNN_MODEL_PATH = os.path.join(BASE_DIR, 'best_models_cnn', 'best_model.pickle')

ROTATION_ANGLES = [5, 10, 15, 20, 30]
TRANSLATION_PIXELS = [1, 2, 3, 4, 5]
NOISE_STDS = [0.1, 0.2, 0.3, 0.4, 0.5]

N_TEST = 2000

def apply_perturbation(images, pert_type, intensity):
    """
    images: [N, H, W] in [0, 1]
    pert_type: 'rotation' | 'translation' | 'noise'
    intensity: angle (deg) | pixels | noise std
    """
    if pert_type == 'rotation':
        return rotate(images, intensity, axes=(2, 1), reshape=False, mode='constant', cval=0.0)
    elif pert_type == 'translation':
        return shift(images, (0, intensity, intensity), mode='constant', cval=0.0)
    elif pert_type == 'noise':
        if intensity == 0:
            return images.copy()
        noise = np.random.normal(0, intensity, images.shape)
        return np.clip(images + noise, 0, 1)


test_images_path = os.path.join(BASE_DIR, 'dataset', 'MNIST', 't10k-images-idx3-ubyte.gz')
test_labels_path = os.path.join(BASE_DIR, 'dataset', 'MNIST', 't10k-labels-idx1-ubyte.gz')

with gzip.open(test_images_path, 'rb') as f:
    magic, num, rows, cols = unpack('>4I', f.read(16))
    test_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols).astype(np.float32)
with gzip.open(test_labels_path, 'rb') as f:
    magic, num = unpack('>2I', f.read(8))
    test_labs = np.frombuffer(f.read(), dtype=np.uint8)

test_imgs = test_imgs / 255.0  # [10000, 28, 28]
test_imgs = test_imgs[:N_TEST]
test_labs = test_labs[:N_TEST]

mlp = nn.models.Model_MLP()
mlp.load_model(MLP_MODEL_PATH)

cnn = nn.models.Model_CNN()
cnn.load_model(CNN_MODEL_PATH)


# ---------- evaluate ----------
def evaluate_model(model, images_2d, labels, model_type):
    """images_2d: [N, 28, 28]"""
    if model_type == 'MLP':
        logits = model(images_2d.reshape(-1, 784))
    else:
        logits = model(images_2d.reshape(-1, 1, 28, 28))
    return nn.metric.accuracy(logits, labels)


results = {'rotation': {'angles': [], 'mlp': [], 'cnn': []},
           'translation': {'pixels': [], 'mlp': [], 'cnn': []},
           'noise': {'stds': [], 'mlp': [], 'cnn': []}}

print("Rotation:")
for angle in ROTATION_ANGLES:
    imgs_pert = apply_perturbation(test_imgs, 'rotation', angle)
    mlp_acc = evaluate_model(mlp, imgs_pert, test_labs, 'MLP')
    cnn_acc = evaluate_model(cnn, imgs_pert, test_labs, 'CNN')
    results['rotation']['angles'].append(angle)
    results['rotation']['mlp'].append(mlp_acc)
    results['rotation']['cnn'].append(cnn_acc)
    print(f"  angle={angle:3d}°: MLP={mlp_acc:.4f}, CNN={cnn_acc:.4f}")

print("Translation:")
for px in TRANSLATION_PIXELS:
    imgs_pert = apply_perturbation(test_imgs, 'translation', px)
    mlp_acc = evaluate_model(mlp, imgs_pert, test_labs, 'MLP')
    cnn_acc = evaluate_model(cnn, imgs_pert, test_labs, 'CNN')
    results['translation']['pixels'].append(px)
    results['translation']['mlp'].append(mlp_acc)
    results['translation']['cnn'].append(cnn_acc)
    print(f"  px={px}: MLP={mlp_acc:.4f}, CNN={cnn_acc:.4f}")

print("Gaussian Noise:")
for std in NOISE_STDS:
    np.random.seed(42)  # fixed seed for reproducibility
    imgs_pert = apply_perturbation(test_imgs, 'noise', std)
    mlp_acc = evaluate_model(mlp, imgs_pert, test_labs, 'MLP')
    cnn_acc = evaluate_model(cnn, imgs_pert, test_labs, 'CNN')
    results['noise']['stds'].append(std)
    results['noise']['mlp'].append(mlp_acc)
    results['noise']['cnn'].append(cnn_acc)
    print(f"  std={std:.2f}: MLP={mlp_acc:.4f}, CNN={cnn_acc:.4f}")


fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# rotation
ax = axes[0]
ax.plot(results['rotation']['angles'], results['rotation']['mlp'], 'o-', label='MLP')
ax.plot(results['rotation']['angles'], results['rotation']['cnn'], 's-', label='CNN')
ax.set_xlabel('Rotation (degrees)')
ax.set_ylabel('Test Accuracy')
ax.set_title('Rotation Robustness')
ax.legend()
ax.grid(True, alpha=0.3)

# translation
ax = axes[1]
ax.plot(results['translation']['pixels'], results['translation']['mlp'], 'o-', label='MLP')
ax.plot(results['translation']['pixels'], results['translation']['cnn'], 's-', label='CNN')
ax.set_xlabel('Translation (pixels)')
ax.set_ylabel('Test Accuracy')
ax.set_title('Translation Robustness')
ax.legend()
ax.grid(True, alpha=0.3)

# noise
ax = axes[2]
ax.plot(results['noise']['stds'], results['noise']['mlp'], 'o-', label='MLP')
ax.plot(results['noise']['stds'], results['noise']['cnn'], 's-', label='CNN')
ax.set_xlabel('Gaussian Noise Std')
ax.set_ylabel('Test Accuracy')
ax.set_title('Noise Robustness')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
save_path = os.path.join(BASE_DIR, 'figs', 'robustness_analysis.png')
plt.savefig(save_path, dpi=150)
plt.show()
print(f"\nFigure saved to: {save_path}")
