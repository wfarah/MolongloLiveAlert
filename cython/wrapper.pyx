from libc.stdlib cimport malloc,free
import numpy as np
cimport numpy as np
np.import_array()

cdef extern from "numpy/arrayobject.h":
    void PyArray_ENABLEFLAGS(np.ndarray arr, int flags) # To python to take control of memory block

cdef extern from "functions.c":
    void MAD_1D(long time_series[], int n, float *md, float *mad)
    void norm_baseline(long * time_series, int size, float med, float mad, float * time_series_norm)
    void convolve(float * time_series, int ndat, int width, float * ans)
    void convolve_sn0(float * time_series, int ndat, int width,int nbin_start, int nbin_search, float * ans)
    void read_block(char * file_direc, unsigned long  nsamp_skip, int nsamp_read, unsigned char * data)
    void get_time_series(unsigned char * data, float dm, long * time_series, int nsamp_read)
    int sampling_bits
    int nchans

cdef unsigned char * block #Block of Filterbank data. Visible in the c wrapper and c functions scope
cdef unsigned long nsamp_read

cdef pointer_to_numpy_array(void * ptr, np.npy_intp size):
    '''Convert c pointer to numpy array.
    The memory will be freed as soon as the ndarray is deallocated.
    '''
    cdef np.ndarray[float, ndim=1] arr = \
            np.PyArray_SimpleNewFromData(1, &size, np.NPY_FLOAT, ptr)
    PyArray_ENABLEFLAGS(arr, np.NPY_OWNDATA)
    return arr

cdef pointer_to_numpy_array_long(void * ptr, np.npy_intp size):
    cdef np.ndarray[np.long_t, ndim=1] arr =\
            np.PyArray_SimpleNewFromData(1, &size, np.NPY_LONG, ptr)
    PyArray_ENABLEFLAGS(arr,np.NPY_OWNDATA)
    return arr



def load_block(char * file_direc, unsigned long nsamp_skip, int samples):
    """ Function that loads block of filterbank data unto memory 
        Warning: BLOCK SHOULD BE DELETED EXPLICITLY, USING delete_block()
    """
    global block,nsamp_read
    nsamp_read = samples

    block = <unsigned char *>malloc(sampling_bits*nchans*samples/8)
    read_block(file_direc,nsamp_skip,nsamp_read,block)

def delete_block():
    """ Function that deletes the filterbank block """
    free(block)


def dedisp_norm_base_conv(int width, float dm):
    """ Function that dedisperses filterbank block, which is already loaded unto memory with load_block(),
    into a time series. Time series is normalized, and baseline removed, then convolved with
    a boxcar of particular width
    NOTE: LOAD BLOCK FIRST WITH load_block()
    """
    cdef long * time_series
    cdef float med,mad
    cdef float * convolved

    time_series = <long*>malloc(nsamp_read * sizeof(long))
    get_time_series(block,dm,time_series,nsamp_read) #Dedisp and frequency crunch.

    MAD_1D(time_series,nsamp_read,&med,&mad) #Computes the median and MAD of time series

    time_series_norm = <float *>malloc(nsamp_read * sizeof(float))
    norm_baseline(time_series,nsamp_read, med, mad, time_series_norm) #Normalizes, removes baseline, output is time_series_norm casted to float
    free(time_series) #ALWAYS FREE MEMORY

    convolved = <float*>malloc(nsamp_read * sizeof(float))
    convolve(time_series_norm,nsamp_read,width,convolved)  #Convolves with box car of width=width

    free(time_series_norm)

    to_python = pointer_to_numpy_array(convolved,nsamp_read)   #No need to free this, python takes control of it
    return to_python


def norm_base_conv_sn0(int width, int backstep, int disp_delay):
    """Function that dedispwerese at dm=0, computes the SN=0. same as the dedisp_norm_base_conv(), but take in
    the value of the dispersion delay in samples
    NOTE: for this function, block should have been already loaded
    """
    cdef long * time_series
    cdef float med,mad
    cdef float * convolved

    time_series = <long*>malloc(nsamp_read * sizeof(long))
    get_time_series(block,0,time_series,nsamp_read) #Dedisp and frequency crunch.

    MAD_1D(time_series,nsamp_read,&med,&mad) #Computes the median and MAD of time series

    time_series_norm = <float *>malloc(nsamp_read * sizeof(float))
    norm_baseline(time_series,nsamp_read, med, mad, time_series_norm) #Normalizes, removes baseline, output is time_series_norm casted to float
    free(time_series) #ALWAYS FREE MEMORY

    convolved = <float*>malloc(nsamp_read * sizeof(float))
    convolve_sn0(time_series_norm,nsamp_read,width,backstep,disp_delay,convolved)

    free(time_series_norm)

    to_python = pointer_to_numpy_array(convolved,nsamp_read)   #No need to free this, python takes control of it
    return to_python


def dedisp(float dm):
    """
    Wrapper function that returns dedispersed time_series
    NOTE: LOAD BLOCK FIRST WITH load_block()
    """
    cdef long * time_series
    time_series = <long*>malloc(nsamp_read * sizeof(long))
    get_time_series(block, dm, time_series, nsamp_read)
    to_python = pointer_to_numpy_array_long(time_series,nsamp_read)
    return to_python


