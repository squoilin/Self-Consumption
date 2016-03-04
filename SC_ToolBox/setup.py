# -*- coding: utf-8 -*-
"""
Setup File for the cython builder. The file to compile is:
yearly_simulation.pyx

Created on Mon Jan 11 20:03:12 2016

@author: Sylvain Quoilin, JRC
"""

# Cython setup file

# python setup.py build_ext --inplace

from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("yearly_simulation.pyx"),
)