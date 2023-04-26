from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy

ext_modules = [
    Extension(
        "util",
        ["util.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
    )
]

setup(ext_modules=cythonize("util.pyx", language_level=3), include_dirs=[numpy.get_include()])