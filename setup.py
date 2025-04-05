from setuptools import setup, find_packages

setup(
    name="appyc",                 # Name of the package
    version="0.1.13",                    # Version
    author="Tong Zhou",                 # Your name
    author_email="zt9465@gmail.com", # Your email
    description="APPy (Annotated Parallelism for Python) enables users to annotate loops and tensor expressions in Python with compiler directives akin to OpenMP, and automatically compiles the annotated code to GPU kernels.", # Short description
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
    install_requires=['ast_comments', 'black', 'sympy', 'ast_transforms'],
)