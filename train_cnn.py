import mynn as nn
from draw_tools.plot import plot

import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle
import os

np.random.seed(309)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
train_images_path = os.path.join(BASE_DIR, 'dataset', 'MNIST', 'train-images-idx3-ubyte.gz')
train_labels_path = os.path.join(BASE_DIR, 'dataset', 'MNIST', 'train-labels-idx1-ubyte.gz')

with gzip.open(train_images_path, 'rb') as f:
    magic, num, rows, cols = unpack('>4I', f.read(16))
    train_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows * cols)

with gzip.open(train_labels_path, 'rb') as f:
    magic, num = unpack('>2I', f.read(8))
    train_labs = np.frombuffer(f.read(), dtype=np.uint8)

idx = np.random.permutation(np.arange(num))
with open(os.path.join(BASE_DIR, 'idx_cnn.pickle'), 'wb') as f:
    pickle.dump(idx, f)

train_imgs = train_imgs[idx]
train_labs = train_labs[idx]
valid_imgs = train_imgs[:10000]
valid_labs = train_labs[:10000]
train_imgs = train_imgs[10000:]
train_labs = train_labs[10000:]

train_imgs = train_imgs / 255.0
valid_imgs = valid_imgs / 255.0

train_imgs = train_imgs.reshape(-1, 1, 28, 28)
valid_imgs = valid_imgs.reshape(-1, 1, 28, 28)

# architecture: conv(1,8,5) -> ReLU -> conv(8,16,5) -> ReLU -> Flatten -> Linear(16*20*20, 10)
conv_config = [(1, 8, 5, 1, 0), (8, 16, 5, 1, 0)]
fc_config = [(16 * 20 * 20, 10)]

cnn_model = nn.models.Model_CNN(conv_config=conv_config, fc_config=fc_config, act_func='ReLU',
                                initialize_method=nn.op.xavier_init)
optimizer = nn.optimizer.SGD(init_lr=0.06, model=cnn_model)
# scheduler = nn.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[800, 2400, 4000], gamma=0.5)
loss_fn = nn.op.MultiCrossEntropyLoss(model=cnn_model, max_classes=10)

runner = nn.runner.RunnerM(cnn_model, optimizer, nn.metric.accuracy, loss_fn, scheduler=None)

save_dir = os.path.join(BASE_DIR, 'best_models_cnn')
runner.train([train_imgs, train_labs], [valid_imgs, valid_labs],
             num_epochs=5, log_iters=100, save_dir=save_dir)

print(f"\nBest dev accuracy: {runner.best_score:.4f}")

_, axes = plt.subplots(1, 2)
axes = axes.reshape(-1)
_.set_tight_layout(1)
plot(runner, axes)
plt.savefig(os.path.join(BASE_DIR, 'figs', 'cnn_learning_curve.png'))
plt.show()
