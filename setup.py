#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION"

module1 = Extension(name='func',
                   sources = ['func.pyx'],
                   define_macros = [('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
                   include_dirs = [np.get_include(), '.'],
                   )

module2 = Extension(name='main',
                   sources = ['main_cy.pyx'],
                   define_macros = [('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
                   include_dirs = [np.get_include(), '.'],
                   )

setup(name='Helper functions to AIS',
      ext_modules=cythonize(module1))
setup(name='Helper functions to AIS',
      ext_modules=cythonize(module2))
  #    include_dirs=[np.get_include()],
   #   extra_compile_args=['-Wno-#warnings'])


# =============================================================================
# filename = 'agents3.pyx'
# 
# agents_module = Extension(
#     'Agents',
#     sources = [filename],
#     define_macros = [('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
#     include_dirs = [numpy.get_include()],
# )
# 
# setup (name = 'Agents',
#     ext_modules = cythonize(agents_module)
# )
# =============================================================================
