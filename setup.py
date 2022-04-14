from setuptools import setup

setup(
    name='galaxyMNIST',
    version='0.1.0',
    py_modules=['galaxy_mnist'],
    install_requires=[
        'pandas', 'scikit-learn', 'h5py'
    ],  # does not require torch/vision, as you may well already have it or want a very specific version. Tested on torch 1.10.2, torchvision 0.11.3.
    entry_points='''
        [console_scripts]
        example=example:example
    ''',
)
