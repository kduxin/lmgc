cdef extern from "Python.h":
    char* PyUnicode_AsUTF8(object unicode)

cimport cython
import numpy as np
cimport numpy as np
from libc.stdlib cimport atof
from cython.parallel cimport prange
from libc.stdlib cimport free

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float32_t, ndim=1] fromstring(str s, int count, str sep):
    cdef list segs = s.split(sep)
    cdef int length = len(segs)
    cdef int n = length if count < 0 else count
    cdef np.ndarray[np.float32_t, ndim=1] floats = np.empty(n, dtype=np.float32)
    
    cdef int[::1] starts = np.zeros(n, dtype=np.int32)
    cdef int start = 0
    cdef int i
    for i in range(n):
        starts[i] = start
        start += len(segs[i]) + 1
    
    cdef char *s1 = PyUnicode_AsUTF8(s)

    with nogil:
        for i in prange(n, schedule='static', num_threads=8):
            floats[i] = atof(&s1[starts[i]])
    
    # free(s1)
    return floats