#include "functions.h"


/* Function to be implemented to update the variables in the
 * header files */
/*
void update_header(){
}
*/

/* Function that returns the median of a array of long */
float median(long x[], int n) {
    int i,j;
    long * temp_arr;
    temp_arr = malloc(n*sizeof(long));
    for (i=0;i<n;i++) temp_arr[i]=x[i];
    float temp;
    // the following two loops sort the array x in ascending order
    for(i=0; i<n-1; i++){
        for(j=i+1; j<n; j++) {
            if(temp_arr[j] < temp_arr[i]) {
                // swap elements
                temp = temp_arr[i];
                temp_arr[i] = temp_arr[j];
                temp_arr[j] = temp;
            }
        }
    }

    if(n%2==0) {
        // if there is an even number of elements, return mean of the two elements in the middle
        temp = (temp_arr[n/2] + temp_arr[n/2 - 1]) / 2.0;
        free(temp_arr);
        return(temp);
    } else {
        // else return the element in the middle
        temp = x[n/2];
        free(temp_arr);
        return temp;
    }
}

/* Function that returns the median of an array of floats */
float median_float(float x[], int n){
    int i,j;
    float * temp_arr;
    temp_arr = malloc(n*sizeof(float));
    for (i=0;i<n;i++) temp_arr[i]=x[i];
    float temp;
    // the following two loops sort the array x in ascending order
    for(i=0; i<n-1; i++){
        for(j=i+1; j<n; j++) {
            if(temp_arr[j] < temp_arr[i]) {
                // swap elements
                temp = temp_arr[i];
                temp_arr[i] = temp_arr[j];
                temp_arr[j] = temp;
            }
        }
    }

    if(n%2==0) {
        // if there is an even number of elements, return mean of the two elements in the middle
        temp = (temp_arr[n/2] + temp_arr[n/2 - 1]) / 2.0;
        free(temp_arr);
        return(temp);
    } else {
        // else return the element in the middle
        temp = x[n/2];
        free(temp_arr);
        return temp;
    }
}

/*  Returns the median and the MAD of 1D array */
void MAD_1D(long time_series[], int n, float *md, float *mad){
    int i;
    float t_median = median(time_series,n);
    float * temp;
    temp = malloc(n * sizeof(float));
    for (i=0;i<n;i++){
        temp[i] = abs(t_median - time_series[i]);
    }
    *md = t_median;
    *mad = median_float(temp,n);
    free(temp);
}

/* Removes baseline of time_series and normalizes it, time_series_norm is casted to float */
void norm_baseline(long * time_series, int size, float med, float mad, float * time_series_norm){
    int i;
    for (i=0;i<size;i++) time_series_norm[i] = (float)((time_series[i] - med)/(1.4826*mad));
}


/* Convolves 1d array with boxcar of width = width, from start till start + bin search */
void convolve(float * time_series, int ndat, int width, float * ans){
    int i,j;
    for (i=0;i<ndat;i++) ans[i]=0.0;
    for (i=start;i<start+bin_search;i++){
        float ss=0.0;
        for (j=0;j<width;j++) ss+=time_series[i+j];
        ans[i]=ss;
    }
}

/* Convolves 1d array with boxcar, given start and number of samples to
 * convolve. Used for sn at dm = 0 in offline mode */
void convolve_sn0(float * time_series, int ndat, int width,
        int nbin_start, int nbin_search, float * ans){
    int i,j;
    for (i=0;i<ndat;i++) ans[i]=0.0;
    for (i=nbin_start;i<nbin_start+nbin_search;i++){
    float ss=0.0;
    for (j=0;j<width;j++) ss+=time_series[i+j];
    ans[i]=ss;
    }
}

/* Function that returns list of dispersion delays */
void dm_delay(float f0, float df, float dm, int nchans, int *d_list){
  float nu1,nu2,nu1_2,nu2_2,shift;
  int i;
  for (i=0; i<nchans; i++){
      nu1 = f0;
      nu2 = f0 + i*df; // df usually -ve
      nu1_2 = 1.0e6/(nu1*nu1);
      nu2_2 = 1.0e6/(nu2*nu2);
      shift = 4.148808e-3 * dm * (nu2_2-nu1_2);  // secs
      d_list[i] = round(shift / sampling_time);
  }
  return ;
}

/* Function that reads filterbank files, skips nsamp_skip bytes and reads nsamp_read number 
 * samples. data is the pointer to the 1D data, and is column major.
 */
void read_block(char * file_direc, unsigned long  nsamp_skip, int nsamp_read, unsigned char * data){
    unsigned long long bytes_skip = header_bytes + nsamp_skip*nchans;
    FILE * fptr = fopen(file_direc,"rb");
    if (fptr==NULL){
        printf("Error opening file \n");
        exit(-1);
    }
    fseek(fptr,bytes_skip,SEEK_CUR);
    fread((unsigned char *) data,sampling_bits/8,nchans*nsamp_read,fptr);
    fclose(fptr);
    return;
}



/* Performs a transformation of index from row-major to column-major */ 
int index_transform(int old_index, int rows, int cols){
    int real_index, old_col, old_row;
    old_row = old_index % cols;
    old_col = old_index / cols;
    real_index = old_row * rows + old_col;
    return real_index;
}

/* Computes the time_series */
void get_time_series(unsigned char * data, float dm, long * time_series, int nsamp_read){
    int i,j;
    unsigned long index,real_index,sum;
    int d_list[nchans];

    dm_delay(f0,df,dm,nchans,d_list);
    

    for (i=0;i<nsamp_read;i++){
        sum = 0;
        for (j=0;j<nchans;j++){
            index = j*nsamp_read + (i+d_list[j])%nsamp_read;
            if (index >= nsamp_read * nchans){
                printf("ERROR, READING OUT OF BOUNDS");
                exit(-1);
            }
            real_index = index_transform(index,nchans,nsamp_read);
            sum += data[real_index];
        }
        time_series[i] = sum;
    }
   return;
}
/*
int main(int argc, char **argv){
    char *file_direc = "/home/wfarah/disp_test/2016-08-05-07:48:39.fil";
    float dm = 50;
    unsigned long nsamp_skip = 50972;
    int nsamp_read = 400;
    long * time_series;
    time_series = malloc(nsamp_read * sizeof(long));
    unsigned char * block;
    get_time_series(block, dm, time_series, nsamp_skip);
    free(block);
    free(time_series);

    return 1;
}
*/


int main(int argc, char **argv){
    char *file_direc = "/home/wfarah/disp_test/2016-08-05-07:48:39.fil";
    float dm = 50;
    unsigned long nsamp_skip = 50972;
    int nsamp_read = 400;
    long * time_series;
    time_series = malloc(nsamp_read * sizeof(long));
    unsigned char * block;
    block = malloc(sampling_bits*nchans*nsamp_read/8);
    read_block(file_direc,nsamp_skip,nsamp_read,block);
    
    get_time_series(block, dm, time_series, nsamp_read);
    free(block);
    float med,mad;
    MAD_1D(time_series,nsamp_read,&med,&mad);
    float * time_series_norm;
    time_series_norm = malloc(nsamp_read * sizeof(float));
    norm_baseline(time_series, nsamp_read, med, mad, time_series_norm);
    free(time_series);
    float * convolved;
    convolved = malloc(nsamp_read * sizeof(float));
    int width = 10;
    convolve(time_series_norm, nsamp_read, width, convolved);
    int i;
    for (i=0;i<nsamp_read;i++) printf("%f ",convolved[i]);

    free(time_series_norm);
    return 1;
}

