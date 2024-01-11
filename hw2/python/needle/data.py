import collections
import gzip
import os
import struct

import numpy as np
from .autograd import Tensor

from typing import Iterator, Optional, List, Sized, Union, Iterable, Any


class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as n H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        if flip_img:
            return img[:, ::-1, :]
        else:
            return img


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """ Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return 
            H x W x C NAArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(low=-self.padding, high=self.padding + 1, size=2)
        new_image = np.pad(img, [(self.padding, self.padding), (self.padding, self.padding), (0, 0)])
        return new_image[self.padding + shift_x:self.padding + shift_x + img.shape[0],
               self.padding + shift_y:self.padding + shift_y + img.shape[1], :]


class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
     """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
            self,
            dataset: Dataset,
            batch_size: Optional[int] = 1,
            shuffle: bool = False,
    ):
        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        if not self.shuffle:
            self.ordering = np.array_split(np.arange(len(dataset)),
                                           range(batch_size, len(dataset), batch_size))

    def __iter__(self):
        if self.shuffle:
            self.ordering = np.array_split(np.random.permutation(len(self.dataset)),
                                           range(self.batch_size, len(self.dataset), self.batch_size))
        self.idx = -1
        return self

    def __next__(self):
        self.idx += 1
        if self.idx >= len(self.ordering):
            raise StopIteration
        samples = [self.dataset[i] for i in self.ordering[self.idx]]
        return [Tensor(np.stack([samples[i][j] for i in range(len(samples))])) for j in range(len(samples[0]))]


class MNISTDataset(Dataset):
    def __init__(self, image_filename: str, label_filename: str, transforms: Optional[List] = None):
        super().__init__(transforms)
        with gzip.open(image_filename, 'rb') as f:
            magic_num, image_num, row_num, col_num = struct.unpack(">4i", f.read(16))
            self.row_num = row_num
            self.col_num = col_num
            pixels = row_num * col_num
            image_arr = np.vstack(
                [np.array(struct.unpack(f"{pixels}B", f.read(pixels)), dtype=np.float32) for _ in range(image_num)])
            image_arr -= np.min(image_arr)
            image_arr /= np.max(image_arr)
            self.X = image_arr

        with gzip.open(label_filename, 'rb') as f:
            magic_num, image_num = struct.unpack(">2i", f.read(8))
            label_arr = np.array(struct.unpack(f"{image_num}B", f.read(image_num)), dtype=np.uint8)
            self.y = label_arr

    def __getitem__(self, index) -> object:
        x = self.X[index]
        label = self.y[index]
        if len(x.shape) > 1:
            return np.stack([self.apply_transforms(img.reshape(self.row_num, self.col_num, 1)).reshape(self.row_num*self.col_num) for img in x]), label
        return self.apply_transforms(x.reshape(self.row_num, self.col_num, 1)).reshape(self.row_num*self.col_num), label

    def __len__(self) -> int:
        return self.X.shape[0]


class NDArrayDataset(Dataset):
    def __init__(self, *arrays):
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, i) -> object:
        return tuple([a[i] for a in self.arrays])
