from setuptools import Extension, setup
from Cython.Build import cythonize
from numpy import get_include

ext_modules = [
    Extension(
        "fdr",
        ["fdr.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
    )
]

setup(
    ext_modules=cythonize(['fdr.pyx'], annotate=True),
    include_dirs=[get_include()],
    zip_safe=False,
)
