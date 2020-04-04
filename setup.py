from setuptools import setup, find_namespace_packages
setup(

    name="structs",
    version="0.0.1",
    author="Oliver Batchelor",
    author_email="saulzar@gmail.com",
    description="Python structs and tables using dot notation",
    url="https://github.com/saulzar/structs",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
