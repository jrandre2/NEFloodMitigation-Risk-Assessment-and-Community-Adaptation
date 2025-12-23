#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Setup script for Parcel Owner Classification Project
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="parcel-owner-classification",
    version="1.0.0",
    author="Jesse Andrews",
    author_email="jesse.andrews@unl.edu",
    description="Machine learning pipeline for classifying property owner types using BERT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jesseandrews/parcel-owner-classification",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: GIS",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "classify-parcels=scripts.production.production_parcel_classification:main",
        ],
    },
)
