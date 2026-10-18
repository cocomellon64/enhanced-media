from setuptools import setup, find_packages

setup(
    name="video-image-enhancer",
    version="1.0.0",
    description="Conservative, auditable video and image enhancer to 4K with confidence maps",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "opencv-python>=4.7.0",
        "Pillow>=9.5.0",
    ],
    extras_require={
        "api": ["fastapi>=0.100.0", "uvicorn[standard]>=0.23.0", "python-multipart>=0.0.6"],
        "ui": ["gradio>=4.0.0"],
        "metrics": ["scikit-image>=0.21.0", "lpips>=0.1.4"],
        "all": [
            "fastapi>=0.100.0", "uvicorn[standard]>=0.23.0", "python-multipart>=0.0.6",
            "gradio>=4.0.0", "scikit-image>=0.21.0", "lpips>=0.1.4",
        ],
    },
    entry_points={
        "console_scripts": [
            "enhance=scripts.enhance_cli:main",
        ],
    },
)
