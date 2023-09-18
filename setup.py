from setuptools import setup

setup(
    name='appy',
    version='0.1',
    packages=['appy'],
    install_requires=[
        'torch', 'triton=2.1.0', 'ast-comments', 
    ],
)
