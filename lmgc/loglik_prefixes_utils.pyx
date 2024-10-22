cimport cython
import numpy as np
cimport numpy as np
from libc.stdlib cimport atof, atoi, malloc, free
from libc.stdio cimport *
from libc.string cimport memcpy, strcmp
from cython.parallel cimport prange, parallel
import tqdm.auto as tqdm

cdef extern from "Python.h":
    char* PyUnicode_AsUTF8(object unicode)
 
cdef extern from "stdio.h":
    #FILE * fopen ( const char * filename, const char * mode )
    FILE *fopen(const char *, const char *)
    #int fclose ( FILE * stream )
    int fclose(FILE *)
    #ssize_t getline(char **lineptr, size_t *n, FILE *stream);
    ssize_t getline(char **, size_t *, FILE *)

@cython.boundscheck(False)
@cython.wraparound(False)
def extract_whole_query_loglik(filename, n_queries, skip, max_len):
    filename_byte_string = filename.encode("UTF-8")
    cdef char* fname = filename_byte_string
    docid2logprobs = {}
 
    cdef FILE* cfile
    cfile = fopen(fname, "rb")
    if cfile == NULL:
        raise FileNotFoundError(2, "No such file or directory: '%s'" % filename)
 
    cdef char * line = NULL
    cdef size_t l = 0
    cdef ssize_t read
    
    cdef size_t i = 0
    cdef size_t j
    cdef char* docid = <char*>malloc(256 * sizeof(char))
    cdef size_t queryid = 0;
    cdef np.ndarray[np.float32_t, ndim=1] logprobs = np.ones(n_queries, dtype=np.float32)
    
    for _ in range(skip):
        read = getline(&line, &l, cfile)
        if read == -1: break
 
    pbar = tqdm.tqdm(total=n_queries, desc="Processing lines")
    i = 0
    while True:
        read = getline(&line, &l, cfile)
        if read == -1: break

        # find docid
        j = 0
        while line[j] != '\t':
            if j >= read:
                raise ValueError("No docid found in the line")
            j = j + 1

        # initialize logprobs if new docid
        if i % n_queries == 0:
            if i > 0:
                docid2logprobs[docid.decode()] = logprobs
                pbar.update(n_queries)
            memcpy(docid, line, j * sizeof(char))
            docid[j] = '\0'
            logprobs = np.ones(n_queries, dtype=np.float32)
        
        # check docid & queryid
        assert strcmp(docid, line[:j]) == 0, f"{docid} ----vs----- {line[:j]}"
        queryid = atoi(&line[j+1])
        assert queryid == i % n_queries, f"{queryid} ----vs----- {i % n_queries}"
        
        # find logprob
        if max_len < 0:
            j = read - 1
            while line[j] != '|':
                if j < 0:
                    raise ValueError("No | found in the line")
                j = j - 1
            logprobs[queryid] = atof(&line[j+1])
        else:
            while line[j] != '\t':
                j += 1
            j += 1
            k = 0
            while j < read:
                if line[j] == '|':
                    k += 1
                    if k == max_len:
                        break
                j += 1

            j -= 1
            while line[j] != '|' and line[j] != '\t':
                if j < 0:
                    raise ValueError("No | found in the line")
                j = j - 1
            logprobs[queryid] = atof(&line[j+1])

        # increment i
        i = i + 1

    # process the last docid
    assert i % n_queries == 0
    docid2logprobs[docid.decode()] = logprobs
 
    fclose(cfile)
    return docid2logprobs
