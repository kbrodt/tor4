import gzip
import os
from pathlib import Path

import numpy as np
import requests
import tqdm

if "USE_TORCH" in os.environ:
    print("Using pytorch")
    import torch as tor4
    import torch.nn as nn
else:
    print("Using tor4")
    import tor4
    import tor4.nn as nn


ROOT = Path(os.path.abspath(__file__)).parent / "mnist"
BASE_URL = "http://yann.lecun.com/exdb/mnist/"
MNIST_FILES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]


def download_mnist():
    for m in MNIST_FILES:
        if (ROOT / m).exists():
            continue

        r = requests.get(f"{BASE_URL}{m}")
        with open(ROOT / m, "wb") as f:
            f.write(r.content)


def get_mnist():
    if not ROOT.exists():
        ROOT.mkdir()
        download_mnist()

    out = []
    for m in MNIST_FILES:
        with gzip.open(ROOT / m, "rb") as f:
            f.read(8)
            if "images" in m:
                f.read(8)

            arr = np.frombuffer(f.read(), dtype="uint8")
            if "images" in m:
                arr = arr.reshape((-1, 1, 28, 28))

            out.append(arr)

    return out


def ohe(y):
    y_ohe = np.zeros((len(y), y.max() + 1))
    y_ohe[np.arange(len(y)), y] = 1

    return y_ohe


def epoch_step(model, loader, desc, criterion, opt=None):
    is_train = opt is not None
    if is_train:
        model.train()
    else:
        model.eval()

    with tqdm.tqdm(loader, desc=desc, mininterval=2, leave=False) as pbar:
        running_loss = running_acc = n = 0
        for x, y in pbar:
            logits = model(x)
            loss = criterion(logits, y)

            if is_train:
                for p in model.parameters():
                    p.grad = None
                # opt.zero_grad()
                loss.backward()
                opt.step()

                logits = logits.detach()

            bs = len(x)
            running_loss += loss.item() * bs
            running_acc += (logits.numpy().argmax(-1) == y.numpy().argmax(-1)).sum()
            n += bs

            pbar.set_postfix(
                **{"loss": f"{running_loss/n:.6f}", "acc": f"{running_acc/n:.6f}"}
            )

    return running_loss / n, running_acc / n


def get_loader(X, y, batch_size=32, shuffle=False):
    n = len(X)
    inds = np.arange(n)

    if shuffle:
        np.random.shuffle(inds)

    for _ in range(1 if shuffle else 1):
        for start in range(0, n, batch_size):
            end = start + batch_size
            sl = inds[start:end]
            yield X[sl], y[sl]


class ClfFC(nn.Module):
    def __init__(self, in_features, n_classes, hidden_dim=64):
        super().__init__()

        self.features = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=2 * hidden_dim),
            nn.ReLU(),
        )

        self.dropout = nn.Dropout(p=0.8)
        self.classifier = nn.Linear(in_features=hidden_dim * 2, out_features=n_classes)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.features(x)

        x = self.dropout(x)

        x = self.classifier(x)

        return x


class ClfConv(nn.Module):
    def __init__(self, in_channels, n_classes, hidden_dim=64):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=hidden_dim,
                kernel_size=7,
                stride=(2, 2),
                bias=False,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_dim,
                out_channels=2 * hidden_dim,
                kernel_size=3,
                stride=(2, 2),
                bias=False,
            ),
            nn.ReLU(),
        )

        self.dropout = nn.Dropout2d(p=0.8)
        self.classifier = nn.Linear(in_features=hidden_dim * 2, out_features=n_classes)

    def forward(self, x):
        x = self.features(x)

        x = x.max(dim=-1)[0]
        x = x.max(dim=-1)[0]

        x = self.dropout(x)

        x = self.classifier(x)

        return x


def main():
    X_train, y_train, X_test, y_test = get_mnist()
    X_train = tor4.tensor(data=X_train, dtype=tor4.float32) / 255
    y_train = tor4.tensor(data=ohe(y_train), dtype=tor4.float32)
    X_test = tor4.tensor(data=X_test, dtype=tor4.float32) / 255
    y_test = tor4.tensor(data=ohe(y_test), dtype=tor4.float32)

    if "USE_CONV" in os.environ:
        model = ClfConv(1, 10)
    else:
        model = ClfFC(28 * 28, 10)
    print(model)
    opt = tor4.optim.SGD(model.parameters(), lr=1e-1)

    def criterion(input, target):
        return nn.functional.cross_entropy(input, target.argmax(-1))
        return nn.functional.cross_entropy_slow3(input, target)
        return nn.functional.cross_entropy_slow2(input, target)
        return nn.functional.cross_entropy_slow(input, target)
        return nn.functional.binary_cross_entropy_with_logits(input, target)
        return nn.functional.mse_loss(input.sigmoid(), target)
        return nn.functional.mse_loss_slow(input.sigmoid(), target)

    batch_size = 32
    n_epochs = 50
    for epoch in range(1, n_epochs + 1):
        loss = epoch_step(
            model,
            get_loader(X_train, y_train, batch_size=batch_size, shuffle=True),
            f"[ Training {epoch}/{n_epochs}.. ]",
            criterion,
            opt,
        )
        print(f"Train loss: {loss}")
        with tor4.no_grad():
            loss = epoch_step(
                model,
                get_loader(X_test, y_test, batch_size=batch_size),
                f"[ Testing {epoch}/{n_epochs}.. ]",
                criterion,
                opt=None,
            )
        print(f"Test loss: {loss}")
        print()


if __name__ == "__main__":
    main()
