from setuptools import setup

setup(
    name='slap',
    version='0.1',
    packages=['slap'],
    install_requires=[
        'torch', 'triton==2.0.0', 'ast-comments', 
    ],
)
