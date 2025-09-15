from setuptools import setup, find_packages

setup(
    name="sci-tutor",
    version="0.1.0",
    description="AI tutoring system for processing academic PDF books with citation lookup capabilities",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.13",
    install_requires=[
        "transformers==4.30.2",
        "datasets==2.14.0",
        "accelerate==0.21.0",
        "bitsandbytes==0.41.0",
        "peft==0.4.0",
        "PyMuPDF==1.23.0",
        "pytesseract==0.3.10",
        "requests==2.31.0",
        "beautifulsoup4==4.12.2",
        "chromadb==0.4.0",
        "sentence-transformers==2.2.2",
        "Pillow==10.0.0",
        "tqdm==4.65.0",
        "torch>=2.0.1",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)