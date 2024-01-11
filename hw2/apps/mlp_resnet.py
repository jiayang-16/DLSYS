import sys
from python import needle as ndl
from python.needle import nn
import numpy as np
import time
import os

from python.needle.data import MNISTDataset, DataLoader

np.random.seed(0)


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    path = nn.Sequential(nn.Linear(dim, hidden_dim), norm(dim=hidden_dim), nn.ReLU(), nn.Dropout(drop_prob),
                         nn.Linear(hidden_dim, dim), norm(dim=dim))
    return nn.Sequential(nn.Residual(path), nn.ReLU())


def MLPResNet(dim, hidden_dim=100, num_blocks=3, num_classes=10, norm=nn.BatchNorm1d, drop_prob=0.1):
    return nn.Sequential(nn.Linear(dim, hidden_dim), nn.ReLU(),
                         *[ResidualBlock(dim=hidden_dim, hidden_dim=hidden_dim // 2, norm=norm, drop_prob=drop_prob) for
                           _ in
                           range(num_blocks)], nn.Linear(hidden_dim, num_classes))


def epoch(dataloader, model: nn.Module, opt=None):
    np.random.seed(4)
    loss_fn = nn.SoftmaxLoss()
    err, total_loss = 0.0, []
    if opt is None:
        model.eval()
    else:
        model.train()
    for X, y in dataloader:
        logits = model(X)
        loss = loss_fn(logits, y)
        err += np.sum(logits.numpy().argmax(axis=1) != y.numpy())
        total_loss.append(loss.numpy())
        if opt is not None:
            opt.reset_grad()
            loss.backward()
            opt.step()
    return err/len(dataloader.dataset), np.mean(total_loss)


def train_mnist(batch_size=100, epochs=10, optimizer=ndl.optim.Adam,
                lr=0.001, weight_decay=0.001, hidden_dim=100, data_dir="../data"):
    np.random.seed(4)
    resnet = MLPResNet(784, hidden_dim=hidden_dim, num_classes=10)
    opt = optimizer(resnet.parameters(), lr=lr, weight_decay=weight_decay)
    train_set = MNISTDataset(f"{data_dir}/train-images-idx3-ubyte.gz",
                             f"{data_dir}/train-labels-idx1-ubyte.gz")
    test_set = MNISTDataset(f"{data_dir}/t10k-images-idx3-ubyte.gz",
                            f"{data_dir}/t10k-labels-idx1-ubyte.gz")
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size)
    for _ in range(epochs):
        train_err, train_loss = epoch(train_loader, resnet, opt)
    test_err, test_loss = epoch(test_loader, resnet, None)
    return train_err, train_loss, test_err, test_loss


if __name__ == "__main__":
    train_mnist(data_dir="../data")
