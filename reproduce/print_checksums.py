import os
import hashlib

if __name__ == '__main__':

    download_folder_loc = '/home/walml/repos/galaxy_mnist/download_root/GalaxyMNIST/raw'
    for dataset_gz_loc in [os.path.join(download_folder_loc, x) for x in ['train_dataset.hdf5.gz', 'test_dataset.hdf5.gz']]:

        with open(dataset_gz_loc, 'rb') as f:
            md5_checksum = hashlib.md5(f.read()).hexdigest()
            print(md5_checksum)
        