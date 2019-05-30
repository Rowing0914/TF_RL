# followed this tutorial
# https://packaging.python.org/tutorials/packaging-projects/

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="TF_RL",
    version="0.1.14",
    author="Norio Kosaka",
    author_email="kosakaboat@gmail.com",
    description="Reinforcement Learning for Tensorflow",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Rowing0914/TF_RL",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)