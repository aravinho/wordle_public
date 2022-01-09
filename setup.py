from setuptools import find_packages, setup

setup(
    name="wordle",
    version="0.0.0",
    packages=find_packages(),
    install_requires=[
        "fire",
        "gym",
        "ipdb",
        "torch",
        "tensorboard",
        "termcolor",
        "tqdm",
    ],
    license="MIT License",
)
