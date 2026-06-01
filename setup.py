"""Editable install: `pip install -e .` (or `pip install -e ".[dev]"` for pytest)."""
from setuptools import find_packages, setup

setup(
    name="vit_cnn_rae",
    version="0.4.0",
    description="Attention-guided local reversible adversarial examples",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0",
        "torchvision>=0.15",
        "timm>=0.9",
        "scikit-image>=0.19",
        "matplotlib",
        "pandas",
        "tqdm",
        "pillow",
    ],
    extras_require={
        "dev": ["pytest>=8.0"],
    },
)
