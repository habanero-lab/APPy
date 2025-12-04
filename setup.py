import os
import re
from setuptools import setup, find_packages

def read_version():
    with open(os.path.join("appy", "__version__.py")) as f:
        match = re.search(r'__version__\s*=\s*["\'](.+?)["\']', f.read())
        if match:
            return match.group(1)
        raise RuntimeError("Version not found.")
    
setup(
    name="appyc",                 
    version=read_version(),            
    author="Tong Zhou",           
    author_email="zt9465@gmail.com", 
    description="APPy (Annotated Parallelism for Python) enables users to easily run Python loops on GPUs.", # Short description
    long_description=open("README.md").read(), # Long description from README file
    long_description_content_type="text/markdown", # Type of the long description
    url="https://github.com/habanero-lab/APPy", # URL of your project repository
    packages=find_packages(),           # Automatically find the packages in your project
    classifiers=[
        "Programming Language :: Python :: 3",  # Classifiers help users find your project
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',            # Minimum Python version requirement
    install_requires=[
        'ast_comments', 
        'ast_transforms',        
    ],
    extras_require={
        "triton": [
            "torch",  # Triton depends on torch
            "triton",
        ],
        "metal": [
            "metalcompute"
        ],
        "cuda": [
            "pycuda",
        ],
        "dev": [
            "pytest",
            "pytest-regressions",
            "black",
            "numpy",
            "scipy"
        ],
    },
)