#ifndef RESIZE_H
#define RESIZE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>


// undistort_map_t points to a big array of these, 1 per pixel
typedef struct bilinear_lookup_t{
    int16_t		I[2]; // backwards map to top left corner of 2x2 square
    uint8_t		F[4]; // 4 coefficients for 4 corners of square
} bilinear_lookup_t;


typedef struct undistort_map_t{
    int w_in;			    // input image width
    int h_in;			    // input image height
    int w_out;              // output image width
    int h_out;              // output image height
    bilinear_lookup_t* L;   // lookup table
} undistort_map_t;

// takes the input and output dimensions and generates a lookup table
int mcv_init_resize_map(int w_in, int h_in, int w_out, int h_out, undistort_map_t* map);

// resizes the image using the lookup table created by mcv_init_resize_map
int mcv_resize_image(const uint8_t* input, uint8_t* output, undistort_map_t* map);
// "" but with 3 channels
int mcv_resize_8uc3_image(const uint8_t* rgb_input, uint8_t* output, undistort_map_t* map);

#ifdef __cplusplus
}
#endif

#endif // RESIZE_H
