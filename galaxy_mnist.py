import os
import logging
from typing import Tuple, Any

import torch
import h5py
from PIL import Image
from sklearn import model_selection

from urllib.error import URLError
from torchvision.datasets.utils import download_and_extract_archive, download_url
from torchvision.datasets.mnist import MNIST


class GalaxyMNIST(MNIST):
    """`GalaxyMNIST <https://github.com/mwalmsley/galaxy_mnist>`_ Dataset.

    Based on MNIST/FashionMNIST torchvision datasets.

    self.data (self.targets) is uint8 torch tensors of labels  (targets)
    self.__getitem__ returns (PIL image, torch uint8 target) by indexing self.data

    Args:
        root (string): Root directory of dataset where ``GalaxyMNIST/raw/train_dataset.hdf5``
            and  ``GalaxyMNIST/raw/test_dataset.hdf5`` exist.
        train (bool, optional): If True, creates dataset from ``GalaxyMNIST/raw/train_dataset.hdf5``,
            otherwise from ``GalaxyMNIST/raw/test_dataset.hdf5``.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    # simply overrides resources, classes

    # mirrors option removed, use my own download func (very similar - see docstring)

    # check_integrity will skip md5 check if None
    # don't bother with md5 until dataset definitely not changing? just set None
    # https://github.com/pytorch/vision/blob/main/torchvision/datasets/utils.py#L416
    resources = [
        ('https://dl.dropboxusercontent.com/s/5a14de0o7slyif9/train_dataset.hdf5.gz', 'e408ae294e9b975482dc1abffeb373a6'),
        ('https://dl.dropboxusercontent.com/s/5rza12nn24cwd2k/test_dataset.hdf5.gz', '7a940e4cea64a8b7cb60339098f74490')
    ]

    classes = ["smooth_round", "smooth_cigar", "edge_on_disk", "unbarred_spiral"]


    def download(self) -> None:
        """
        Download the data if it doesn't exist already.

        Modified from MNIST to remove {mirror}{filename} setup which doesn't work well with Dropbox
        (dropbox urls like https://dl.dropboxusercontent.com/s/xenuo0ekgyi10ru/train_dataset.hdf5.gz
        will ignore the final part and always download the file matching that hash).
        I also use almost this exact code in pytorch-galaxy-datasets (pasted as lazy)
        """

        if self._check_exists():
            return

        # MNIST pattern expects {root}/raw directory
        os.makedirs(self.raw_folder, exist_ok=True)

        # download files
        for url, md5 in self.resources:
            filename = os.path.basename(url)
            try:
                logging.info(f"Downloading {url}")
                if url.endswith('.tar.gz') or url.endswith('.hdf5.gz') or url.endswith('.zip'):
                    download_and_extract_archive(
                        url, download_root=self.raw_folder, filename=filename, md5=md5)
                else:  # don't try to extract archive, just download
                    download_url(url, root=self.raw_folder, filename=filename, md5=md5)
            except URLError as error:
                logging.info(f"Failed to download (trying next):\n{error}")
                continue



    def _check_legacy_exist(self):
        # GalaxyMNIST has no legacy data (yet).
        # Function exists for potential backwards compatibility only
        return False


    def _load_legacy_data(self):
        raise NotImplementedError(
            """
            GalaxyMNIST has no legacy data (yet).
            Function exists for potential backwards compatibility only
            """
        )

    
    def _load_data(self):
        """
        Reads the extracted {train/test}_dataset.hdf5. 
        Each hdf5 includes both the images and labels - see read_dataset_file. 
        This defines the canonical dataset (MNIST-style, as a standard reference)
        To make your own tweaks (e.g. set a different train-test split, use ``load_custom_data``)       

        Returns:
            images: NHWC PIL images, 8000 train images or 2000 test images
            targets: torch uint64 tensor like N, 0-3 integer-encoded classes (see GalaxyMNIST.classes), similarly
        """
        dataset_file = f"{'train' if self.train else 'test'}_dataset.hdf5"
        images, targets = read_dataset_file(os.path.join(self.raw_folder, dataset_file))
        return images, targets


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Copied from MNIST, except mode='RGB' not 'L' as it's RGB colour
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) image is NHWC PIL image and target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])  # CHW convention

        # will return PIL image, as per mnist dataset. Transpose back to HWC before fromarray converts to PIL.
        img = Image.fromarray(img.numpy().transpose(1, 2, 0), mode='RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


    def load_custom_data(self, test_size=0.2, stratify=False, random_state=42):
        """
        Load GalaxyMNIST in a different way to the canonical GalaxyMNIST() (which is GalaxyMNIST._load_data())
        Note - has no effect on GalaxyMNIST class itself e.g. self.data, self.targets, which are always canonical.
        Use as a pure function only.

        Args:
            test_size (float, optional): Select test size/fraction as per sklearn.model_selection.train_test_split. Defaults to 0.2.
            stratify (bool, optional): Force exactly even number of classes between splits. Defaults to False.

        Returns:
            (torch.tensor, torch.tensor): train dataset of labels and images like (NCHW, N), same format as self._load_data()
            (torch.tensor, torch.tensor): test dataset of labels and images like (NCHW, N), same format as self._load_data()
        """

        # mnist init uses self.train to control _load_data, so need to modify self.train
        # need to do some bookkeeping to set it back afterwards
        prev_self_train_state = self.train  # bool is always ref-by-value

        self.train = True
        canonical_train_images, canonical_train_targets = self._load_data()
        self.train = False
        canonical_test_images, canonical_test_targets = self._load_data()

        all_images = torch.cat([canonical_train_images, canonical_test_images], axis=0)
        all_labels = torch.cat([canonical_train_targets, canonical_test_targets], axis=0)

        self.train = prev_self_train_state  # set self.train back how it was

        # print(all_images.shape, all_labels.shape)

        if stratify:
            split_indices = model_selection.StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state).split(X=all_images, y=all_labels)
            train_indices, test_indices = list(split_indices)[0]
            train_images, train_labels, test_images, test_labels = all_images[train_indices], all_labels[train_indices], all_images[test_indices], all_labels[test_indices]
        else:
            train_images, train_labels, test_images, test_labels = model_selection.train_test_split(all_images, all_labels, test_size=test_size)
        

        if self.train:
            self.data, self.targets = (train_images, train_labels)
        else:
            self.data, self.targets = (test_images, test_labels)

        return (train_images, train_labels), (test_images, test_labels)


class GalaxyMNISTHighrez(GalaxyMNIST):

    resources = [
        ("https://dl.dropboxusercontent.com/s/xenuo0ekgyi10ru/train_dataset.hdf5.gz", '3391dcddac14d5b4055db73fb600ae63'),
        ("https://dl.dropboxusercontent.com/s/lczri4sb4bbcgyh/test_dataset.hdf5.gz", 'fb272c4e94000b4d99a09d638977b0b9')
    ]
    # otherwise identical to GalaxyMNIST (should be exactly the same galaxies). 
    # Just pointing at hdf5's with differently resized images.


def read_dataset_file(path: str) -> torch.Tensor:
    """
    Read an hdf5 file containing galaxy images (under ``images`` and integer-encoded labels (under ``labels``)

    Args:
        path (str): path to read from (e.g. root/galaxyMNIST/train_dataset.hdf5)

    Returns:
        torch.Tensor: galaxy images, torch uint8 tensor like NCHW
        torch.Tensor: torch uint64 tensor like N, 0-3 integer-encoded classes (see GalaxyMNIST.classes), similarly
    """
    with h5py.File(path, 'r') as f:

        images = f['images'][:]
        # images are saved as NHWC convention
        # (numpy/matplotlib being the tiebreaker for pytorch vs tensorflow)
        images = torch.from_numpy(images).type(torch.uint8).permute(0, 3, 1, 2)
        assert images.ndimension() == 4

        targets = f['labels'][:]  # integer-encoded (from 0) according to GalaxyMNIST.classes
        targets = torch.from_numpy(targets).type(torch.int64) # dtype consistent with mnist (same as tensor.long())
        assert targets.ndimension() == 1

    return images, targets


# TODO optionally, download or write jpgs?
