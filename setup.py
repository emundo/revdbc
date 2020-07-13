# pylint: disable=exec-used
import os

from setuptools import setup, find_packages

source_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "revdbc")

version_scope = {}
with open(os.path.join(source_root, "version.py")) as f:
    exec(f.read(), version_scope)
version = version_scope["__version__"]

project_scope = {}
with open(os.path.join(source_root, "project.py")) as f:
    exec(f.read(), project_scope)
project = project_scope["project"]

with open("README.md") as f:
    long_description = f.read()

classifiers = [
    "Intended Audience :: Developers",
    "Intended Audience :: Education",
    "Intended Audience :: Information Technology",
    "Intended Audience :: Science/Research",

    "License :: OSI Approved :: Apache Software License",

    "Programming Language :: Python :: 2",
    "Programming Language :: Python :: 2.7",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.4",
    "Programming Language :: Python :: 3.5",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8"
]

classifiers.extend(project["categories"])

if version["tag"] == "alpha":
    classifiers.append("Development Status :: 3 - Alpha")

if version["tag"] == "beta":
    classifiers.append("Development Status :: 4 - Beta")

if version["tag"] == "stable":
    classifiers.append("Development Status :: 5 - Production/Stable")

del project["categories"]
del project["year"]

setup(
    version = version["short"],
    long_description = long_description,
    long_description_content_type = "text/markdown",
    license = "Apache 2.0",
    packages = find_packages(),
    entry_points = {
        "console_scripts": [
            "revdbc=revdbc.__main__:main"
        ],
    },
    install_requires = [
        "candumpgen>=0.0.1,<0.1",
        "enum34>=1.1.10,<2",
        "future>=0.18.2,<0.19",
        "scikit-learn>=0.23.0,<0.24"
    ],
    python_requires = ">=2.7, !=3.0, !=3.1, !=3.2, !=3.3",
    zip_safe = False,
    classifiers = classifiers,
    **project
)
