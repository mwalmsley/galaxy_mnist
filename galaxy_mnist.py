import os

import torch
import h5py
from sklearn import model_selection

from torchvision.datasets.mnist import MNIST


class GalaxyMNIST(MNIST):
    """`GalaxyMNIST <https://github.com/mwalmsley/galaxy_mnist>`_ Dataset.

    Based on MNIST/FashionMNIST torchvision datasets.

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

    # simply overrides mirrors, resources, classes

    mirrors = ["http://www.jb.man.ac.uk/research/MiraBest/MiraBest_F/"]

    # check_integrity will skip md5 check if None
    # don't bother with md5 until dataset definitely not changing? just set None
    # https://github.com/pytorch/vision/blob/main/torchvision/datasets/utils.py#L416
    resources = [
        ("train_dataset.hdf5.gz", None),  # 'e408ae294e9b975482dc1abffeb373a6'
        ("test_dataset.hdf5.gz", None)
    ]

    classes = ["smooth_round", "smooth_cigar", "edge_on_disk", "unbarred_spiral"]


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
            images: torch uint8 tensor like NCHW, 8000 train images or 2000 test images
            targets: torch uint64 tensor like N, 0-3 integer-encoded classes (see GalaxyMNIST.classes), similarly
        """
        dataset_file = f"{'train' if self.train else 'test'}_dataset.hdf5"
        images, targets = read_dataset_file(os.path.join(self.raw_folder, dataset_file))
        return images, targets


    def load_custom_data(self, test_size=0.2, stratify=False, random_state=None):
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

        return (train_images, train_labels), (test_images, test_labels)



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
        # reorder axis to NCHW for pytorch convention
        images = torch.from_numpy(images).type(torch.uint8).permute(0, 3, 1, 2)
        assert images.ndimension() == 4

        targets = f['labels'][:]  # integer-encoded (from 0) according to GalaxyMNIST.classes
        targets = torch.from_numpy(targets).type(torch.int64) # dtype consistent with mnist (same as tensor.long())
        assert targets.ndimension() == 1

    return images, targets


# TODO optionally, download or write jpgs?
