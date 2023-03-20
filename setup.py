from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['tensorflow-addons==0.19.0', 'torch==1.11.0', 'dgl', 'deepchem', 'fsspec', 'gcsfs']

setup(
  name='molecular-toxicity-prediction',
  version='0.1',
  author='Demetrios Fassois',
  install_requires=REQUIRED_PACKAGES,
  packages=find_packages(),
  include_package_data=True,
  description="Project for Columbia's EECS6895 Advanced Big Data & AI.")
