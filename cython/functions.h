#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "molongloheader.h"

void dm_delay(float f0, float df, float dm, int nchans, int *d_list);
void read_filfile(char * file_direc, unsigned long long skip, int nsamp_read, unsigned char * data);
int index_transform(int old_index, int rows, int cols);
void get_time_series(char * file_direc, float dm, unsigned long nsamp_skip,int nsamp_read,long * time_series);
