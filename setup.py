from distutils.core import setup
from Cython.Build import cythonize

setup(name='whistprb_common',ext_modules=cythonize('common_c.pyx'))
