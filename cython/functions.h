#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "molongloheader.h"

float median(long x[], int n);
float median_float(float x[], int n);
void MAD_1D(long time_series[], int n, float *md, float *mad);
void norm_baseline(long * time_series, int size, float med, float mad, float * time_series_norm);
void convolve(float * time_series, int ndat, int width, float * ans);
void dm_delay(float f0, float df, float dm, int nchans, int *d_list);
void read_block(char * file_direc, unsigned long  nsamp_skip, int nsamp_read, unsigned char * data);
int index_transform(int old_index, int rows, int cols);
void get_time_series(unsigned char * data, float dm, long * time_series, int nsamp_read);
