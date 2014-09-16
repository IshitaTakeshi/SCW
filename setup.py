import os
from setuptools import setup

def read(filename):
    return open(os.path.join(os.path.dirname(__file__), filename)).read()

setup(
    name='scw',
    py_modules=['scw'],
    version='1.0',
    author='Ishita Takeshi',
    description=("An implementation of " 
                 "Exact Soft Confidence-Weighted Learning"),
    license='MIT',
    keywords = "Exact Soft Confidence-Weighted Learning",
    url='https://pypi.python.org/pypi/scw',
    long_description=read('README.txt'),
    author_email='ishitah.takeshi@gmail.com'
)
