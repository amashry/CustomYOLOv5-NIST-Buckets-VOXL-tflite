/*******************************************************************************
 * Copyright 2021 ModalAI Inc.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * 4. The Software is used solely in conjunction with devices provided by
 *    ModalAI Inc.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 ******************************************************************************/

#ifndef CONFIG_FILE_H
#define CONFIG_FILE_H

#include <modal_json.h>

#define CHAR_BUF_SIZE 128
#define CONFIG_FILE "/etc/modalai/voxl-tflite-server.conf"

#ifdef BUILD_QRB5165
#define CONFIG_FILE_HEADER "\
/**\n\
 * This file contains configuration that's specific to voxl-tflite-server.\n\
 *\n\
 * skip_n_frames       - how many frames to skip between processed frames. For 30Hz\n\
 *                         input frame rate, we recommend skipping 5 frame resulting\n\
 *                         in 5hz model output. For 30Hz/maximum output, set to 0.\n\
 * model               - which model to use. Currently support mobilenet, fastdepth,\n\
 *                         posenet, deeplab, and yolov5.\n\
 * input_pipe          - which camera to use (tracking, hires, or stereo).\n\
 * delegate            - optional hardware acceleration: gpu, cpu, or nnapi. If\n\
 *                         the selection is invalid for the current model/hardware, \n\
 *                         will silently fall back to base cpu delegate.\n\
 * allow_multiple      - remove process handling and allow multiple instances\n\
 *                         of voxl-tflite-server to run. Enables the ability\n\
 *                         to run multiples models simultaneously.\n\
 * output_pipe_prefix  - if allow_multiple is set, create output pipes using default\n\
 *                         names (tflite, tflite_data) with added prefix.\n\
 *                         ONLY USED IF allow_multiple is set to true.\n\
 */\n"
#endif

#ifndef BUILD_QRB5165
#define CONFIG_FILE_HEADER "\
/**\n\
 * This file contains configuration that's specific to voxl-tflite-server.\n\
 *\n\
 * skip_n_frames      - how many frames to skip between processed frames. For 30Hz\n\
 *                        input frame rate, we recommend skipping 5 frame resulting\n\
 *                        in 5hz model output. For 30Hz/maximum output, set to 0.\n\
 * model               - which model to use. Currently support mobilenet, fastdepth,\n\
 *                         posenet, deeplab, and yolov5.\n\
 * input_pipe         - which camera to use (tracking, hires, or stereo).\n\
 * delegate           - optional hardware acceleration: gpu or cpu. If\n\
 *                        the selection is invalid for the current model/hardware, \n\
 *                        will silently fall back to base cpu delegate.\n\
 */\n"
#endif

static int skip_n_frames;
static char model[CHAR_BUF_SIZE];
static char input_pipe[CHAR_BUF_SIZE];
static char delegate[CHAR_BUF_SIZE];
static bool allow_multiple;
static char output_pipe_prefix[CHAR_BUF_SIZE];

static inline void config_file_print(void) {
    printf("=================================================================\n");
    printf("skip_n_frames:                    %d\n", skip_n_frames);
    printf("=================================================================\n");
    printf("model:                            %s\n", model);
    printf("=================================================================\n");
    printf("input_pipe:                       %s\n", input_pipe);
    printf("=================================================================\n");
    printf("delegate:                         %s\n", delegate);
    printf("=================================================================\n");
    #ifdef BUILD_QRB5165
    printf("allow_multiple:                   %s\n", allow_multiple ? "true" : "false");
    printf("=================================================================\n");
    printf("output_pipe_prefix:               %s\n", output_pipe_prefix);
    printf("=================================================================\n");
    #endif
    return;
}

static inline int config_file_read(void) {
    int ret = json_make_empty_file_with_header_if_missing(CONFIG_FILE, CONFIG_FILE_HEADER);
    if (ret < 0)
        return -1;
    else if (ret > 0)
        fprintf(stderr, "Creating new config file: %s\n", CONFIG_FILE);

    cJSON* parent = json_read_file(CONFIG_FILE);
    if (parent == NULL) return -1;

    // actually parse values
    json_fetch_int_with_default(parent, "skip_n_frames", &skip_n_frames, 0);
    json_fetch_string_with_default(parent, "model", model, CHAR_BUF_SIZE, "/usr/bin/dnn/ssdlite_mobilenet_v2_coco.tflite");
    json_fetch_string_with_default(parent, "input_pipe", input_pipe, CHAR_BUF_SIZE, "/run/mpa/hires/");
    json_fetch_string_with_default(parent, "delegate", delegate, CHAR_BUF_SIZE, "gpu");

    #ifdef BUILD_QRB5165
    json_fetch_bool_with_default(parent, "allow_multiple", (int*)&allow_multiple, 0);
    json_fetch_string_with_default(parent, "output_pipe_prefix", output_pipe_prefix, CHAR_BUF_SIZE, "mobilenet");
    #endif

    if (json_get_parse_error_flag()) {
        fprintf(stderr, "failed to parse config file %s\n", CONFIG_FILE);
        cJSON_Delete(parent);
        return -1;
    }

    // write modified data to disk if neccessary
    if (json_get_modified_flag()) {
        printf("The config file was modified during parsing, saving the changes to disk\n");
        json_write_to_file_with_header(CONFIG_FILE, parent, CONFIG_FILE_HEADER);
    }
    cJSON_Delete(parent);
    return 0;
}
#endif  // end CONFIG_FILE_H
