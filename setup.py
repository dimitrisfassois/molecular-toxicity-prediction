from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = []

setup(
  name='molecular-toxicity-prediction',
  version='0.1',
  author='Demetrios Fassois',
  install_requires=REQUIRED_PACKAGES,
  packages=find_packages(),
  description="Project for Columbia's EECS6895 Advanced Big Data & AI.")