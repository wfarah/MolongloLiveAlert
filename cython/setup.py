from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy
"""
ext_modules = [
    Extension("name", ["name.pyx"]),
]
"""
ext_modules = [
    Extension("wrapper", ["wrapper.pyx"],include_dirs=[numpy.get_include()],extra_compile_args=["-O3"])
]

setup(
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules,
  include_dirs=[numpy.get_include()],
)

#$python setup.py build_ext --inplace
#--inplace is for the 
