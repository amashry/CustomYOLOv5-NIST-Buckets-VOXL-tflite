#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include "resize.h"


// AVG 1.4ms min 1.20 ms for vga image on VOXL1 fastest core
int mcv_resize_image(const uint8_t* input, uint8_t* output, undistort_map_t* map)
{
    // shortcut variables to make code cleaner
    int  height = map->h_out;
    int   width = map->w_out;
    int n_pix = width*height;
    bilinear_lookup_t* L = map->L;

    // go through every pixel in output image
    for(int pix=0; pix<n_pix; pix++){

        // check for invalid (blank) pixels
        if(L[pix].I[0]<0){
            output[pix] = 0;
            printf("INVALID PIXEL\n");
            continue;
        }

        // get indices from index lookup I
        uint16_t x1 = L[pix].I[0];
        uint16_t y1 = L[pix].I[1];

        // don't worry about all the index algebra, the compiler optimizes this
        uint16_t p0 = input[map->w_in*y1 + x1];
        uint16_t p1 = input[map->w_in*y1 + x1 + 1];
        uint16_t p2 = input[map->w_in*(y1+1) + x1];
        uint16_t p3 = input[map->w_in*(y1+1) + x1 + 1];

        // multiply add each pixel with weighting
        output[pix] = (	p0*L[pix].F[0] +
                        p1*L[pix].F[1] +
                        p2*L[pix].F[2] +
                        p3*L[pix].F[3]) /256;
    }
    return 0;
}

// AVG TODOms min TODOms for vga image on VOXL1 fastest core
// expects rgb_input as contiguous memory chunk 8bits-R|8bits-G|8bits-B ...
int mcv_resize_8uc3_image(const uint8_t* rgb_input, uint8_t* output, undistort_map_t* map)
{
    // shortcut variables to make code cleaner
    int  height = map->h_out;
    int   width = map->w_out;
    int n_pix = width*height;
    bilinear_lookup_t* L = map->L;

    // go through every pixel in output image
    int out_pix = 0;
    for(int pix=0; pix<n_pix; pix++){
        out_pix = pix * 3;
        // check for invalid (blank) pixels
        if(L[pix].I[0]<0){
            output[out_pix] = 0;
            output[out_pix+1] = 0;
            output[out_pix+2] = 0;
            continue;
        }

        // get indices from index lookup I - same for all 3 components of pixel
        uint16_t x1 = L[pix].I[0];
        uint16_t y1 = L[pix].I[1];

        /**
         * rgb input is arranged as: 8bits-R|8bits-G|8bits-B
         * these indices will be relative to a grayscale image
         * conversion is as follows:
         * mapped_index * 3 = r (start of pixel), + 1 = g, + 2 = b
         *
         * don't worry about all the index algebra, the compiler optimizes this
         */

        /// R ///
        uint16_t p0 = rgb_input[(map->w_in*y1 + x1) * 3];
        /// G ///
        uint16_t p4 = rgb_input[(map->w_in*y1 + x1) * 3 + 1];
        /// B ///
        uint16_t p8 = rgb_input[(map->w_in*y1 + x1) * 3 + 2];
        /// R ///
        uint16_t p1 = rgb_input[(map->w_in*y1 + x1 + 1) * 3 ];
        /// G ///
        uint16_t p5 = rgb_input[(map->w_in*y1 + x1 + 1) * 3 + 1];
        /// B ///
        uint16_t p9 = rgb_input[(map->w_in*y1 + x1 + 1) * 3 + 2];
        /// R ///
        uint16_t p2 = rgb_input[(map->w_in*(y1+1) + x1) * 3];
        /// G ///
        uint16_t p6 = rgb_input[(map->w_in*(y1+1) + x1) * 3 + 1];
        /// B ///
        uint16_t p7 = rgb_input[(map->w_in*(y1+1) + x1) * 3 + 2];
        /// R ///
        uint16_t p3 = rgb_input[(map->w_in*(y1+1) + x1 + 1) * 3];
        /// G ///
        uint16_t p10 = rgb_input[(map->w_in*(y1+1) + x1 + 1) * 3 + 1];
        /// B ///
        uint16_t p11 = rgb_input[(map->w_in*(y1+1) + x1 + 1) * 3 + 2];


        // multiply add each pixel with weighting
        output[out_pix] = (	p0*L[pix].F[0] +
                                p1*L[pix].F[1] +
                                p2*L[pix].F[2] +
                                p3*L[pix].F[3]) /256;

        // multiply add each pixel with weighting
        output[out_pix+1] = (	p4*L[pix].F[0] +
                                p5*L[pix].F[1] +
                                p6*L[pix].F[2] +
                                p7*L[pix].F[3]) /256;

        // multiply add each pixel with weighting
        output[out_pix+2] = (	p8*L[pix].F[0]  +
                                p9*L[pix].F[1]  +
                                p10*L[pix].F[2] +
                                p11*L[pix].F[3]) /256;

    }
    return 0;
}

int mcv_init_resize_map(int w_in, int h_in, int w_out, int h_out, undistort_map_t* map)
{
    map->h_out = h_out;
    map->w_out = w_out;
    map->h_in = h_in;
    map->w_in = w_in;

    // allocate new map
    // TODO some sanity and error checking here
    map->L = (bilinear_lookup_t*)malloc(w_out*h_out*sizeof(bilinear_lookup_t));
    if(map->L==NULL){
        perror("failed to allocate memory for lookup table");
        return -1;
    }
    bilinear_lookup_t* L = map->L;

    float x_r = ((float)(w_in - 1)/(float)(w_out));
    float y_r = ((float)(h_in - 1)/(float)(h_out));

    for(int v=0; v<h_out; ++v){
        for(int u=0; u<w_out; ++u){
            int x_l = (x_r * u);
            int y_l = (y_r * v);
            int x2 = x_l + 1;
            int y2 = y_l + 1;
            // x and y difference for top left point
            float x_w = (x_r * u) - x_l;
            float y_w = (y_r * v) - y_l;

            int pix = w_out*v + u;

            if((x_l < 0 || x2 > (w_in-1)) || (y_l < 0 || y2 > (h_in-1))){
                L[pix].I[0] = -1;
                L[pix].I[1] = -1;
                continue;
            }

            // populate lookup table with top left corner pixel
            L[pix].I[0] = x_l;
            L[pix].I[1] = y_l;

            // integer weightings for 4 pixels. Due to truncation, these 4 ints
            // should sum to no more than 255
            L[pix].F[0] = (1-x_w)*(1-y_w)*256;
            L[pix].F[1] = (x_w)*(1-y_w)*256;
            L[pix].F[2] = (y_w)*(1-x_w)*256;
            L[pix].F[3] = (x_w)*(y_w)*256;
        }
    }
    return 0;
}
