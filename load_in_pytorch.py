import pandas as pd

from galaxy_mnist import GalaxyMNIST


if __name__ == '__main__':

    dataset = GalaxyMNIST(
        root='/home/walml/repos/galaxy_mnist/download_root',
        download=True,
        train=True  # by default, or False for canonical test set
    )

    # this is always the canonical 80/20 train/test split
    images = dataset.data
    labels = dataset.targets

    print(images.shape, images.dtype)
    print(labels.shape, labels.dtype)


    # however, you can set your own split size and stratification if you like
    (custom_train_images, custom_train_labels), (custom_test_images, custom_test_labels) = dataset.load_custom_data(test_size=0.8, stratify=True) 
    print(custom_train_images.shape, custom_test_images.shape)
    print(pd.value_counts(custom_train_labels.numpy()), pd.value_counts(custom_test_labels.numpy()))
    # this does NOT change the always-canonical values from dataset.data, dataset.targets
