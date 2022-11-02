"""Test import of all files to check syntax, path"""

from ucvision_utility.test_utils import test_import


def test_imports():
  test_import('py_structs')


if __name__ == '__main__':
  test_imports()
