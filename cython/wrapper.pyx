from libc.stdlib cimport malloc,free
import numpy as np
cimport numpy as np
np.import_array()

cdef extern from "numpy/arrayobject.h":
    void PyArray_ENABLEFLAGS(np.ndarray arr, int flags) # To python to take control of memory block

cdef extern from "functions.c":
    void dm_delay(float f0, float df, float dm, int nchans, int *d_list)
    void read_filfile(char * file_direc, unsigned long long skip, int nsamp_read, unsigned char * data)
    int index_transform(int old_index, int rows, int cols)
    void get_time_series(char * file_direc, float dm, unsigned long nsamp_skip,int nsamp_read,long * time_series)
    void convolve(float * time_series, int ndat, int width, float * ans)
    void norm_baseline(long * time_series, int size, float med, float mad, float * time_series_norm)
    void MAD_1D(long time_series[], int n, float *md, float *mad)
    float median_float(float x[], int n)
    float median(long x[], int n)

cdef pointer_to_numpy_array(void * ptr, np.npy_intp size):
    '''Convert c pointer to numpy array.
    The memory will be freed as soon as the ndarray is deallocated.
    '''
    cdef np.ndarray[float, ndim=1] arr = \
            np.PyArray_SimpleNewFromData(1, &size, np.NPY_FLOAT, ptr)
    PyArray_ENABLEFLAGS(arr, np.NPY_OWNDATA)
    return arr


def wrapper(char * file_direc, unsigned long nsamp_skip, int nsamp_read,int width, float dm):
    cdef long * time_series
    time_series = <long*>malloc(nsamp_read * sizeof(long))
    get_time_series(file_direc,dm,nsamp_skip,nsamp_read,time_series)    #Loads filterbank, dedisp and frequency crunch
    cdef float med,mad
    MAD_1D(time_series, nsamp_read, &med, &mad) #Computes the median and MAD of time series
    cdef float * time_series_norm
    time_series_norm = <float *>malloc(nsamp_read * sizeof(float))
    norm_baseline(time_series, nsamp_read, med, mad, time_series_norm)  #Normalizes and removes baseline, output is time_series_norm casted to float
    free(time_series)   #ALWAYS FREE MEMORY
    cdef float * convolved
    convolved = <float*>malloc(nsamp_read * sizeof(float))
    convolve(time_series_norm, nsamp_read, width, convolved)    #Convolves with box car of width=width, outputs convolved
    free(time_series_norm)
    to_python = pointer_to_numpy_array(convolved, nsamp_read)    #No need to free this, python takes control of it
    return to_python
