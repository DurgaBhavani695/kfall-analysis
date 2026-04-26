"""
setup.py
========
Package installation configuration for kfall-analysis.

Install in development mode:
    pip install -e .

Install normally:
    pip install .
"""

from setuptools import setup, find_packages
from pathlib import Path

HERE = Path(__file__).parent
LONG_DESCRIPTION = (HERE / "README.md").read_text(encoding="utf-8")

setup(
    name="kfall-analysis",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description=(
        "Fall detection and ADL classification using a Conv-Recurrent HopField "
        "Neural Network (CRHNN) with a custom FireHawks Optimizer on the KFall dataset."
    ),
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/kfall-analysis",
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.8",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "tensorflow==2.10.0",
        "keras==2.10.0",
        "numpy>=1.23,<2.0",
        "pandas>=1.5,<3.0",
        "openpyxl>=3.0",
        "scikit-learn>=1.1",
        "tqdm>=4.64",
        "matplotlib>=3.6",
        "seaborn>=0.12",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "kfall-train=scripts.train:main",
            "kfall-evaluate=scripts.evaluate:main",
            "kfall-predict=scripts.predict:main",
        ]
    },
    include_package_data=True,
    zip_safe=False,
)
