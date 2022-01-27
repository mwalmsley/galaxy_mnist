# GalaxyMNIST

Galaxy images labelled by morphology (shape). Aimed at ML debugging and teaching.

Contains 10,000 images of galaxies (3x64x64), confidently labelled by Galaxy Zoo volunteers as belonging to one of four morphology classes.

## Installation

    git clone https://github.com/mwalmsley/galaxy_mnist
    pip install -e galaxy_mnist

The only dependencies are `pandas`, `scikit-learn`, and `h5py` (for .hdf5 support).
(py)`torch` is required but not specified as a dependency, because you likely already have it and may require a very specific version (e.g. from conda, AWS-optimised, etc).

## Use

Simply use as with MNIST:

    from galaxy_mnist import GalaxyMNIST

    dataset = GalaxyMNIST(
        root='/some/download/folder',
        download=True
    )

Access the images and labels - in a fixed "canonical" 80/20 train/test division - like so:

    images, labels = dataset.data, dataset.targets

You can also divide the data according to your own to your own preferences with `load_custom_data`:

    (custom_train_images, custom_train_labels), (custom_test_images, custom_test_labels) = dataset.load_custom_data(test_size=0.8, stratify=True) 

See `load_in_pytorch.py` for a working example.

## Dataset Details

GalaxyMNIST has four classes: smooth and round, smooth and cigar-shaped, edge-on-disk, and unbarred spiral (you can retrieve this as a list with `GalaxyMNIST.classes`).

The galaxies are selected from Galaxy Zoo DECaLS Campaign A (GZD-A), which classified images taken by DECaLS and released in DR1 and 2.
The images are as shown to volunteers on Galaxy Zoo, except for a 75% crop followed by a resize to 64x64 pixels.

At least 17 people must have been asked the necessary questions, and at least half of them must have answered with the given class.
The class labels are therefore much more confident than from, for example, simply labelling with the most common answer to some question.

The classes are balanced exactly equally across the whole dataset (2500 galaxies per class), but only approximately equally (by random sampling) in the canonical train/test split.
For a split with exactly equal classes on both sides, use `load_custom_data` with `stratify=True`.

You can see the exact choices made to select the galaxies and labels under the `reproduce` folder. This includes the notebook exploring and selecting choices for pruning the decision tree, and the script for saving the final dataset(s).

## Citations and Further Reading

If you use this dataset, please cite [Galaxy Zoo DECaLS](https://ui.adsabs.harvard.edu/abs/2022MNRAS.509.3966W/abstract), the data release paper from which the labels are drawn. Please also acknowledge the DECaLS survey (see the linked paper for an example).

You can find the original volunteer votes (and images) on Zenodo [here](https://doi.org/10.5281/zenodo.4196266).
