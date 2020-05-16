from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(name='gym_combrf',
      version='0.0.1',
      install_requires=['gym', 'numpy']#And any other dependencies required
)
