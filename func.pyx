#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
# "cimport" is used to import special compile-time information
# about the numpy module (this is stored in a file numpy.pxd which is
# currently part of the Cython distribution).
cimport numpy as np
# =============================================================================
# DTYPE_int = np.int
# # "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# # every type in the numpy module there's a corresponding compile-time
# # type with a _t-suffix.
# ctypedef np.int_t DTYPE_t
# =============================================================================

DTYPE = np.float32
ctypedef np.float32_t DTYPE_t

def say_hello_to(name):
    print("Hello %s!" % name)


cdef double cy_affinity(np.ndarray[DTYPE_t, ndim=1] vector1,
                        np.ndarray[DTYPE_t, ndim=1] vector2):
#def cy_affinity(vector1,
#                         vector2):    
    """
    Compute the affinity (Normalized!! distance) between two features
    vectors.
    
    Parameters
    --------------
    vector1: list
        First features vector
    vector2: list
        Second features vector
    
    Returns
    --------------
        The affinity between the two vectors [0-1]
    """
    cdef double dist
    cdef Py_ssize_t x_max = vector1.shape[0]
    cdef Py_ssize_t x
    cdef double tmp = 0.0
    
    #dist = (np.square(vector1 - vector2).sum())**0.5
    for x in range(x_max):
        tmp += (vector1[x] - vector2[x])**2.0   
    
    dist = tmp**0.5
    
    return dist/(1.0 + dist)
