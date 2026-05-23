import mynn as nn
import numpy as np
from struct import unpack
import gzip
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'best_models_cnn', 'best_model.pickle')  # change this
MODEL_TYPE = 'CNN'  # 'MLP' or 'CNN'

if MODEL_TYPE == 'MLP':
    model = nn.models.Model_MLP()
elif MODEL_TYPE == 'CNN':
    model = nn.models.Model_CNN()
else:
    raise ValueError("MODEL_TYPE must be 'MLP' or 'CNN'")

model.load_model(MODEL_PATH)

test_images_path = os.path.join(BASE_DIR, 'dataset', 'MNIST', 't10k-images-idx3-ubyte.gz')
test_labels_path = os.path.join(BASE_DIR, 'dataset', 'MNIST', 't10k-labels-idx1-ubyte.gz')

with gzip.open(test_images_path, 'rb') as f:
    magic, num, rows, cols = unpack('>4I', f.read(16))
    test_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28 * 28)

with gzip.open(test_labels_path, 'rb') as f:
    magic, num = unpack('>2I', f.read(8))
    test_labs = np.frombuffer(f.read(), dtype=np.uint8)

test_imgs = test_imgs / 255.0

if MODEL_TYPE == 'CNN':
    test_imgs = test_imgs.reshape(-1, 1, 28, 28)

logits = model(test_imgs)
acc = nn.metric.accuracy(logits, test_labs)
print(f"Model: {MODEL_PATH}")
print(f"Test accuracy: {acc:.4f} ({acc*100:.2f}%)")
