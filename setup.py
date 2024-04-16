from setuptools import setup, find_packages
import os
import re

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def find_version():
    path_to_init = os.path.join(ROOT_DIR, "brainimagetools", "__init__.py")
    with open(path_to_init, "r", encoding="utf-8") as f:
        content = f.read()
        version_match = re.search(r"^__version__ = ['\"](.*?)['\"]$", content, re.M)
        if version_match:
            return version_match.group(1)
        raise RuntimeError("version cannot be found!")

def read_file_to_list(filepath):
    """
    Read a text file and return a list of lines.

    Parameters
    ----------
    filepath : str
        The path to the text file.

    Returns
    -------
    list of str
        A list where each element is a line from the file.
    """
    with open(filepath, "r") as file:
        lines = file.readlines()
    return [re.split(r"[=>]", line)[0].strip() for line in lines]

with open(os.path.join(ROOT_DIR, "README.md"), "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="brainimagetools",
    version=find_version(),
    author="Leon Martin",
    author_email="leon.martin@bih-charite.de",
    description="Plotting utilities for brain visualization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://git.bihealth.org/martinl/brainimagetools",
    packages=find_packages(include=["brainimagetools", "brainimagetools.*"]),
    include_package_data=True,
    project_urls={
        "Bug Tracker": "https://git.bihealth.org/martinl/brainimagetools/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    # package_dir={"": "tvbase"},
    python_requires=">=3.6",
    install_requires=read_file_to_list("requirements.txt"),
)
