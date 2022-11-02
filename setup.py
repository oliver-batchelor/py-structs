from setuptools import find_namespace_packages, setup

setup(
    name='py-structs',
    version='1.1.0',
    author='Oliver Batchelor',
    author_email='saulzar@gmail.com',
    description='Python structs and tables using dot notation',
    url='https://github.com/oliver_batchelor/py_structs',
    packages=find_namespace_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],

    install_requires=[
      'numpy',
      'immutables'
    ],

    python_requires='>=3.8',
)
