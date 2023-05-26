from setuptools import setup

setup(
    name='bmp',
    version='0.1',
    packages=['bmp'],
    install_requires=[
        'torch', 'triton', 'typed_ast'
    ],
)
