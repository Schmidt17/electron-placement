from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

ext_modules = [
    Extension(
        "cython_utils",
        ["cython_utils.pyx"],
        libraries=["m"],  # Unix-like specific
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
    )
]

setup(
    name='cython_utils',
    ext_modules=cythonize(ext_modules),
)