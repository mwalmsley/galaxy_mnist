import os
import gzip
from random import random  # part of the standard library!
import shutil
import hashlib

import h5py
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def load_image(loc, cropped_size=318, resized_size=64):
    # decals models use resize 424->300, tfrecord, then crop->224 with no resize
    # effective crop factor is 224/300=0.75
    # therefore consistent crop from 424 = 318
    original_size = 424
    pixels_to_crop_per_side = (original_size - cropped_size)//2

    im = Image.open(loc)

    im = im.crop((
        pixels_to_crop_per_side,
        pixels_to_crop_per_side,
        original_size - pixels_to_crop_per_side,
        original_size - pixels_to_crop_per_side,
    ))

    if resized_size != cropped_size:

        im = im.resize(
            (resized_size, resized_size),
            resample=Image.LANCZOS  # else you get grid-like resampling artifacts
        )

    return np.array(im)


def construct_integer_labels(names_of_classes):
    name_to_int = {
        "smooth_round": 0,
        "smooth_cigar": 1,
        "edge_on_disk": 2,
        "unbarred_spiral": 3
    }
    return np.array([name_to_int[name] for name in names_of_classes]).astype(np.uint8)


def save_galaxy_dataset(df_name, df, save_dir):
        dataset_hdf5_loc = os.path.join(save_dir, f'{df_name}_dataset.hdf5')
        dataset_gz_loc = dataset_hdf5_loc + '.gz'

        # NHWC format
        images = np.stack([load_image(loc) for loc in df['jpeg_loc']], axis=0)
        print(images.shape, images.dtype)  # should be uint8

        # write to .hdf5
        with h5py.File(dataset_hdf5_loc, "w") as f:
            _ = f.create_dataset("images", data=images)  # automatically preserves shape, dtype of input ndarray
            # add attributes?
            _ = f.create_dataset("labels", data=df['integer_label'].astype(np.uint8)) 

        # compress to .gz
        # https://docs.python.org/3/library/gzip.html
        with open(dataset_hdf5_loc, 'rb') as f_in:
            with gzip.open(dataset_gz_loc, 'wb', compresslevel=9) as f_out:
                shutil.copyfileobj(f_in, f_out)

        # print hash
        with open(dataset_gz_loc, 'rb') as f:
            md5_checksum = hashlib.md5(f.read()).hexdigest()
        

        print(f'Saved {df_name} ({len(df)}) to f{dataset_gz_loc}, md5_checksum {md5_checksum}')


if __name__ == '__main__':

    df = pd.read_parquet(
        '/home/walml/repos/zoobot_private/gz_decals_volunteers_1_and_2_internal.parquet'
    )
    print('DR1/2 volunteer catalog: ', len(df))
    label_df = pd.read_parquet('/home/walml/repos/galaxy_mnist/reproduce/latest_labels.parquet')
    df = pd.merge(df, label_df, on='iauname', how='inner')  # implicitly: min 34 votes
    print('Might have label: ', len(df))

    df = df[df['summary'].isin(["smooth_round", "unbarred_spiral", "smooth_cigar", "edge_on_disk"])]
    print('Has final label: ', len(df))

    save_dir = '/home/walml/repos/galaxy_mnist/hidden'

    df['integer_label'] = construct_integer_labels(df['summary'])
    print(df['integer_label'].value_counts())

    # balance dataset
    min_subset_size = 2500
    subsets = []
    for an_integer in df['integer_label'].unique():
        subset = df.query(f'integer_label == {an_integer}')
        assert len(subset) > min_subset_size
        subsets.append(subset[:2500])

    df = pd.concat(subsets, axis=0)
    df = df.sample(len(df), replace=False, random_state=42)

    print('After balancing:')
    print(df['integer_label'].value_counts())

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    print('Train df:')
    print(train_df['integer_label'].value_counts())
    print('Test df:')
    print(test_df['integer_label'].value_counts())


    # temp dev
    # train_df = train_df.sample(200)
    # test_df = test_df.sample(50)

    for df_name, df in [('train', train_df), ('test', test_df)]:

        save_galaxy_dataset(df_name, df, save_dir)
