from setuptools import setup

setup(
    name='loft',
    version='0.1',
    packages=['loft'],
    install_requires=[
        'torch', 'triton', 'typed_ast'
    ],
)
