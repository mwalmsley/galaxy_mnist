from setuptools import setup

setup(
    name='galaxyMNIST',
    version='0.1.0',
    py_modules=['galaxy_mnist'],
    install_requires=[
        'pandas', 'scikit-learn', 'h5py'
    ],  # does not require torch, as you may well already have it or want a very specific version
    entry_points='''
        [console_scripts]
        example=example:example
    ''',
)
