/*******************************************************************************
 * Copyright 2022 ModalAI Inc.
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

#include "inference_helper.h"
#include <string>
#include <fstream>

// generic template for TensorData*
template<typename T>
T* TensorData(TfLiteTensor* tensor, int batch_index);

// Gets the float tensor data pointer
template<>
float* TensorData(TfLiteTensor* tensor, int batch_index)
{
    int nelems = 1;

    for (int i = 1; i < tensor->dims->size; i++){
        nelems *= tensor->dims->data[i];
    }

    switch (tensor->type){
        case kTfLiteFloat32:
            return tensor->data.f + nelems * batch_index;
        default:
            fprintf(stderr, "Error in %s: should not reach here\n", __FUNCTION__);
    }

    return nullptr;
}

// Gets the int32_t tensor data pointer
template<>
int32_t* TensorData(TfLiteTensor* tensor, int batch_index)
{
    int nelems = 1;

    for (int i = 1; i < tensor->dims->size; i++){
        nelems *= tensor->dims->data[i];
    }

    switch (tensor->type){
        case kTfLiteInt32:
            return tensor->data.i32 + nelems * batch_index;
        default:
            fprintf(stderr, "Error in %s: should not reach here\n", __FUNCTION__);
    }

    return nullptr;
}

// Gets the int64_t tensor data pointer
template<>
int64_t* TensorData(TfLiteTensor* tensor, int batch_index)
{
    int nelems = 1;

    for (int i = 1; i < tensor->dims->size; i++){
        nelems *= tensor->dims->data[i];
    }

    switch (tensor->type){
        case kTfLiteInt64:
            return tensor->data.i64 + nelems * batch_index;
        default:
            fprintf(stderr, "Error in %s: should not reach here\n", __FUNCTION__);
    }

    return nullptr;
}

// Gets the int8_t tensor data pointer
template<>
int8_t* TensorData(TfLiteTensor* tensor, int batch_index)
{
    int nelems = 1;

    for (int i = 1; i < tensor->dims->size; i++){
        nelems *= tensor->dims->data[i];
    }

    switch (tensor->type){
        case kTfLiteInt8:
            return tensor->data.int8 + nelems * batch_index;
        default:
            fprintf(stderr, "Error in %s: should not reach here\n", __FUNCTION__);
    }

    return nullptr;
}

// Gets the uint8_t tensor data pointer
template<>
uint8_t* TensorData(TfLiteTensor* tensor, int batch_index)
{
    int nelems = 1;

    for (int i = 1; i < tensor->dims->size; i++){
        nelems *= tensor->dims->data[i];
    }

    switch (tensor->type){
        case kTfLiteUInt8:
            return tensor->data.uint8 + nelems * batch_index;
        default:
            fprintf(stderr, "Error in %s: should not reach here\n", __FUNCTION__);
    }

    return nullptr;
}

// loads labels, pads so the length is a multiple of 16 for tflite
TfLiteStatus ReadLabelsFile(char* file_name, std::vector<std::string>* result, size_t* found_label_count)
{
    std::ifstream file(file_name);

    if (!file){
        fprintf(stderr, "Labels file %s not found\n", file_name);
        return kTfLiteError;
    }

    result->clear();
    std::string line;
    while (std::getline(file, line)){
        result->push_back(line);
    }

    *found_label_count = result->size();
    const int padding = 16;

    while (result->size() % padding){
        result->emplace_back();
    }
    return kTfLiteOk;
}

// timing helper
static uint64_t rc_nanos_monotonic_time()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ((uint64_t)ts.tv_sec*1000000000)+ts.tv_nsec;
}

InferenceHelper::InferenceHelper(char* model_file, char* labels_file, DelegateOpt delegate_choice, bool _en_debug, bool _en_timing, NormalizationType _do_normalize)
{
    // set our helper varss
    en_debug = _en_debug;
    en_timing = _en_timing;
    do_normalize = _do_normalize;
    hardware_selection = delegate_choice;
    labels_location = labels_file;

    // build model
    model = tflite::FlatBufferModel::BuildFromFile(model_file);
    if (!model){
        fprintf(stderr, "FATAL: Failed to mmap model %s\n", model_file);
        exit(-1);
    }

    if (en_debug) printf("Loaded model %s\n", model_file);
    model->error_reporter();
    if (en_debug) printf("Resolved reporter\n");

    // build interpreter
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);

    if (!interpreter){
        fprintf(stderr, "Failed to construct interpreter\n");
        exit(-1);
    }

    // setting multi-threaded
    #ifdef BUILD_QRB5165
    interpreter->SetNumThreads(8);
    #endif

    #ifndef BUILD_QRB5165
    interpreter->SetNumThreads(4);
    #endif
    
    // allow fp precision loss for faster inference
    interpreter->SetAllowFp16PrecisionForFp32(1);

    // setup optional hardware delegate
    switch(hardware_selection){

        case XNNPACK:{
            #ifdef BUILD_QRB5165
            TfLiteXNNPackDelegateOptions xnnpack_options = TfLiteXNNPackDelegateOptionsDefault();
            xnnpack_options.num_threads = 8;
            xnnpack_delegate = TfLiteXNNPackDelegateCreate(&xnnpack_options);
            if (interpreter->ModifyGraphWithDelegate(xnnpack_delegate) != kTfLiteOk) fprintf(stderr, "Failed to apply XNNPACK delegate\n");
            #endif
        }
            break;

        case GPU: {
            TfLiteGpuDelegateOptionsV2 gpu_opts = TfLiteGpuDelegateOptionsV2Default();
            gpu_opts.inference_preference = TFLITE_GPU_INFERENCE_PREFERENCE_SUSTAINED_SPEED;
            gpu_opts.inference_priority1  = TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY;
            gpu_delegate = TfLiteGpuDelegateV2Create(&gpu_opts);
            if (interpreter->ModifyGraphWithDelegate(gpu_delegate) != kTfLiteOk) fprintf(stderr, "Failed to apply GPU delegate\n");
        }
            break;

        case NNAPI: {
            #ifdef BUILD_QRB5165
            const auto* nnapi_impl = NnApiImplementation();
            std::string temp = tflite::nnapi::GetStringDeviceNamesList(nnapi_impl);
            tflite::StatefulNnApiDelegate::Options nnapi_opts;
            nnapi_opts.execution_preference = tflite::StatefulNnApiDelegate::Options::ExecutionPreference::kSustainedSpeed;
            nnapi_opts.allow_fp16 = true;
            nnapi_opts.execution_priority = ANEURALNETWORKS_PRIORITY_HIGH;
            nnapi_opts.use_burst_computation = true;
            nnapi_opts.disallow_nnapi_cpu = false;
            nnapi_opts.max_number_delegated_partitions = -1;
            
            // can manually specificy, the HTA/NPU is shown below
            printf("Manually selecting NPU for nnapi delegation\n");
            nnapi_opts.accelerator_name = "libunifiedhal-driver.so2"; 

            nnapi_delegate = new tflite::StatefulNnApiDelegate(nnapi_opts);            
            if (interpreter->ModifyGraphWithDelegate(nnapi_delegate) != kTfLiteOk) fprintf(stderr, "Failed to apply NNAPI delegate\n");
            #endif
        }
            break;
    }

    if (interpreter->AllocateTensors() != kTfLiteOk){
        fprintf(stderr, "Failed to allocate tensors!\n");
        exit(-1);
    }

    // grab model specific params and use to setup resize map
    TfLiteIntArray* dims = interpreter->tensor(interpreter->inputs()[0])->dims;

    model_height   = dims->data[1];
    model_width    = dims->data[2];
    model_channels = dims->data[3];
}

bool InferenceHelper::preprocess_image(camera_image_metadata_t &meta, char* frame, cv::Mat &preprocessed_image, cv::Mat &output_image){
    start_time = rc_nanos_monotonic_time();
    num_frames_processed++;

    // initialize the resize map on first frame recieved only
    if (num_frames_processed == 1){
        mcv_init_resize_map(meta.width, meta.height, model_width, model_height, &map);
        input_height = meta.height;
        input_width = meta.width;
        
        if (meta.format == IMAGE_FORMAT_RAW8){
            resize_output = (uint8_t*)malloc(model_height * model_width * sizeof(uint8_t));
        }
        else {
            resize_output = (uint8_t*)malloc(model_height * model_width * sizeof(uint8_t) * 3);
        }
        return false;
    }
    // if color input provided, make sure that is reflected in output image
    switch (meta.format){
        case IMAGE_FORMAT_STEREO_NV12:
            meta.format = IMAGE_FORMAT_NV12;
        case IMAGE_FORMAT_NV12:{
            cv::Mat yuv(input_height+input_height/2, input_width, CV_8UC1, (uchar*)frame);
            cv::cvtColor(yuv, output_image, CV_YUV2RGB_NV12);
            mcv_resize_8uc3_image(output_image.data, resize_output, &map);
            cv::Mat holder(model_height, model_width, CV_8UC3, (uchar*)resize_output);

            preprocessed_image = holder;
            meta.format = IMAGE_FORMAT_RGB;
            meta.size_bytes     = (meta.height * meta.width * 3);
            meta.stride         = (meta.width * 3);
        }
            break;
        case IMAGE_FORMAT_STEREO_NV21:
            meta.format = IMAGE_FORMAT_NV21;
        case IMAGE_FORMAT_NV21:{
            cv::Mat yuv(input_height+input_height/2, input_width, CV_8UC1, (uchar*)frame);
            cv::cvtColor(yuv, output_image, CV_YUV2RGB_NV21);
            mcv_resize_8uc3_image(output_image.data, resize_output, &map);
            cv::Mat holder(model_height, model_width, CV_8UC3, (uchar*)resize_output);

            preprocessed_image = holder;
            meta.format = IMAGE_FORMAT_RGB;
            meta.size_bytes     = (meta.height * meta.width * 3);
            meta.stride         = (meta.width * 3);
        }
            break;

        case IMAGE_FORMAT_STEREO_RAW8:
            meta.format = IMAGE_FORMAT_RAW8;
        case IMAGE_FORMAT_RAW8: {
            output_image = cv::Mat(input_height, input_width, CV_8UC1, (uchar*)frame);

            // resize to model input dims
            mcv_resize_image(output_image.data, resize_output, &map);

            // stack resized input to make "3 channel" grayscale input
            cv::Mat holder(model_height, model_width, CV_8UC1, (uchar*)resize_output);
            cv::Mat in[] = {holder, holder, holder};
            cv::merge(in, 3, preprocessed_image);
        }
            break;

        default:
            fprintf(stderr, "Unexpected image format %d received! Exiting now.\n", meta.format);
            return false;
    }

    if (en_timing) total_preprocess_time += ((rc_nanos_monotonic_time() - start_time)/1000000.);

    return true;
}

#define NORMALIZATION_CONST 255.0f
#define PIXEL_MEAN_GUESS 127.0f

bool InferenceHelper::run_inference(cv::Mat preprocessed_image, double* last_inference_time){
    start_time = rc_nanos_monotonic_time();

    // Get input dimension from the input tensor metadata assuming one input only
    int input = interpreter->inputs()[0];

    // manually fill tensor with image data, specific to input format
    switch (interpreter->tensor(input)->type){
        case kTfLiteFloat32:{
            float* dst = TensorData<float>(interpreter->tensor(input), 0);
            const int row_elems = model_width * model_channels;
            for (int row = 0; row < model_height; row++) {
                const uchar* row_ptr = preprocessed_image.ptr(row);
                for (int i = 0; i < row_elems; i++) {
                    if (do_normalize == HARD_DIVISION) 
                        dst[i] = row_ptr[i] / NORMALIZATION_CONST;
                    else if (do_normalize == PIXEL_MEAN)
                        dst[i] = (row_ptr[i] - PIXEL_MEAN_GUESS) / PIXEL_MEAN_GUESS;
                    else 
                        dst[i] = (row_ptr[i]);
                }
                dst += row_elems;
            }
        }
            break;

        case kTfLiteInt8:{
            int8_t* dst = TensorData<int8_t>(interpreter->tensor(input), 0);
            const int row_elems = model_width * model_channels;
            for (int row = 0; row < model_height; row++) {
                const uchar* row_ptr = preprocessed_image.ptr(row);
                for (int i = 0; i < row_elems; i++) {
                    dst[i] = row_ptr[i];
                }
                dst += row_elems;
            }
        }
            break;

        case kTfLiteUInt8:{
            uint8_t* dst = TensorData<uint8_t>(interpreter->tensor(input), 0);
            int row_elems = model_width * model_channels;
            for (int row = 0; row < model_height; row++) {
                uchar* row_ptr = preprocessed_image.ptr(row);
                for (int i = 0; i < row_elems; i++) {
                    dst[i] = row_ptr[i];
                }
                dst += row_elems;
            }
        }
            break;

        default:
            fprintf(stderr, "FATAL: Unsupported model input type!");
            return false;
    }

    if (interpreter->Invoke() != kTfLiteOk){
        fprintf(stderr, "FATAL: Failed to invoke tflite!\n");
        return false;
    }

    int64_t end_time = rc_nanos_monotonic_time();

    if (en_timing) total_inference_time += ((end_time - start_time)/1000000.);
    if (last_inference_time != nullptr) *last_inference_time = ((double)(end_time - start_time)/1000000.);

    return true;
}

// gets some nice randomly generated colors for ids
// when grayscale, will be shades of gray
static cv::Scalar get_color_from_id(int32_t id)
{
    static constexpr int32_t kMaxNum = 100;
    static std::vector<cv::Scalar> color_list;
    if (color_list.empty()) {
        std::srand(123);
        for (int32_t i = 0; i < kMaxNum; i++) {
            color_list.push_back(cv::Scalar(std::rand() % 255, std::rand() % 255, std::rand() % 255));
        }
    }
    return color_list[id % kMaxNum];
}

static void draw_text(cv::Mat& mat, const std::string& text, cv::Point pos, double font_scale, int32_t thickness, cv::Scalar color_front, cv::Scalar color_back, bool is_text_on_rect)
{
    int32_t baseline = 0;
    cv::Size textSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, font_scale, thickness, &baseline);
    baseline += thickness;
    pos.y += textSize.height;
    if (is_text_on_rect) {
        cv::rectangle(mat, pos + cv::Point(0, baseline), pos + cv::Point(textSize.width, -textSize.height), color_back, -1);
        cv::putText(mat, text, pos, cv::FONT_HERSHEY_SIMPLEX, font_scale, color_front, thickness);
    } else {
        cv::putText(mat, text, pos, cv::FONT_HERSHEY_SIMPLEX, font_scale, color_back, thickness * 3);
        cv::putText(mat, text, pos, cv::FONT_HERSHEY_SIMPLEX, font_scale, color_front, thickness);
    }
}

static void draw_fps(cv::Mat& mat, double time_inference, cv::Point pos, double font_scale, int32_t thickness, cv::Scalar color_front, cv::Scalar color_back, bool is_text_on_rect = true)
{
    char text[64];
    static auto time_previous = std::chrono::steady_clock::now();
    auto time_now = std::chrono::steady_clock::now();
    double fps = 1e9 / (time_now - time_previous).count();
    time_previous = time_now;
    snprintf(text, sizeof(text), "FPS: %.1f, Inference: %.1f [ms]", fps, time_inference);
    draw_text(mat, text, cv::Point(0, 0), 0.5, 2, color_front, color_back, true);
}

bool InferenceHelper::postprocess_object_detect(cv::Mat& output_image, std::vector<ai_detection_t>& detections_vector, double last_inference_time){
    start_time = rc_nanos_monotonic_time();
    
    static std::vector<std::string> labels;
    static size_t label_count;

    if (labels.empty()){
        if (ReadLabelsFile(labels_location, &labels, &label_count) != kTfLiteOk){
            fprintf(stderr, "ERROR: Unable to read labels file\n");
            return false;
        }
    }

    // https://www.tensorflow.org/lite/models/object_detection/overview#starter_model
    TfLiteTensor* output_locations    = interpreter->tensor(interpreter->outputs()[0]);
    TfLiteTensor* output_classes      = interpreter->tensor(interpreter->outputs()[1]);
    TfLiteTensor* output_scores       = interpreter->tensor(interpreter->outputs()[2]);
    TfLiteTensor* output_detections   = interpreter->tensor(interpreter->outputs()[3]);
    const float*  detected_locations  = TensorData<float>(output_locations, 0);
    const float*  detected_classes    = TensorData<float>(output_classes, 0);
    const float*  detected_scores     = TensorData<float>(output_scores, 0);
    const int     detected_numclasses = (int)(*TensorData<float>(output_detections, 0));

     for (int i = 0; i < detected_numclasses; i++){
        const float score  = detected_scores[i];
        // scale bboxes back to input resolution
        const int   top    = detected_locations[4 * i + 0] * input_height;
        const int   left   = detected_locations[4 * i + 1] * input_width;
        const int   bottom = detected_locations[4 * i + 2] * input_height;
        const int   right  = detected_locations[4 * i + 3] * input_width;

        // Check for object detection confidence of 60% or more
        if (score > 0.6f){
            if (en_debug){
                printf("Detected: %s, Confidence: %6.2f\n", labels[detected_classes[i]].c_str(), (double)score);
            }
            int height = bottom - top;
            int width  = right - left;

            cv::Rect rect(left, top, width, height);
            cv::Point pt(left, top-10);

            cv::rectangle(output_image, rect, get_color_from_id(detected_classes[i]), 2);
            cv::putText(output_image, labels[detected_classes[i]], pt, cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0), 2);

            // setup ai detection for this detection
            ai_detection_t curr_detection;
            curr_detection.magic_number = AI_DETECTION_MAGIC_NUMBER;
            curr_detection.timestamp_ns = rc_nanos_monotonic_time();
            curr_detection.class_id = detected_classes[i];
            curr_detection.frame_id = num_frames_processed;

            std::string class_holder = labels[detected_classes[i]].substr(labels[detected_classes[i]].find(" ")+1);
            class_holder.erase(remove_if(class_holder.begin(), class_holder.end(), isspace), class_holder.end());
            strcpy(curr_detection.class_name, class_holder.c_str());

            strcpy(curr_detection.cam, cam_name.c_str());
            curr_detection.class_confidence = score;
            curr_detection.detection_confidence = -1; // UNKNOWN for ssd model architecture
            curr_detection.x_min = left;
            curr_detection.y_min = top;
            curr_detection.x_max = right;
            curr_detection.y_max = bottom;

            // fill the vector
            detections_vector.push_back(curr_detection);
        }
    }

    draw_fps(output_image, last_inference_time, cv::Point(0, 0), 0.5, 2, cv::Scalar(0, 0, 0), cv::Scalar(180, 180, 180), true);
    
    if (en_timing) total_postprocess_time += ((rc_nanos_monotonic_time() - start_time)/1000000.);

    return true;
}

bool InferenceHelper::postprocess_mono_depth(camera_image_metadata_t &meta, cv::Mat &output_image, double last_inference_time){
    start_time = rc_nanos_monotonic_time();

    TfLiteTensor* output_locations    = interpreter->tensor(interpreter->outputs()[0]);
    float* depth  = TensorData<float>(output_locations, 0);

    // actual depth image if desired
    cv::Mat depthImage(model_height, model_width, CV_32FC1, depth);

    // setup output metadata
    meta.height = model_height;
    meta.width = model_width;
    meta.size_bytes = meta.width * meta.height * 3;
    meta.stride = meta.width * 3;
    meta.format = IMAGE_FORMAT_RGB;

    // create a pretty colored depth image from the data
    double min_val, max_val;
    cv::Mat depthmap_visual;
    cv::minMaxLoc(depthImage, &min_val, &max_val);
    depthmap_visual = 255 * (depthImage - min_val) / (max_val - min_val); // * 255 for "scaled" disparity
    depthmap_visual.convertTo(depthmap_visual, CV_8U);
    cv::applyColorMap(depthmap_visual, output_image, 4); // opencv COLORMAP_JET

    if (en_timing) total_postprocess_time += ((rc_nanos_monotonic_time() - start_time)/1000000.);

    draw_fps(output_image, last_inference_time, cv::Point(0, 0), 0.5, 2, cv::Scalar(0, 0, 0), cv::Scalar(180, 180, 180), true);

    return true;
}

// pre-defined color map for each class, corresponds to cityscapes_labels.txt
constexpr uint8_t color_map[57] = {
    139,0,0,                    
    255,0,0,
    255,99,71,
    250,128,114,
    255,140,0,
    255,255,0,
    189,183,107,
    154,205,50,
    0,255,0,
    0,100,0,
    0,250,154,
    0,128,128,
    30,144,255,
    25,25,112,
    138,43,226,
    75,0,130,
    139,0,139,
    238,130,238,
    255,20,147
}; 

#define RIGHT_PIXEL_BORDER 110

bool InferenceHelper::postprocess_segmentation(camera_image_metadata_t &meta, cv::Mat &output_image, double last_inference_time){
    start_time = rc_nanos_monotonic_time();

    static std::vector<std::string> labels;
    static size_t label_count;

    if (labels.empty()){
        if (ReadLabelsFile(labels_location, &labels, &label_count) != kTfLiteOk){
            fprintf(stderr, "ERROR: Unable to read labels file\n");
            return false;
        }
    }

    TfLiteTensor* output_locations    = interpreter->tensor(interpreter->outputs()[0]);
    int64_t* classes  = TensorData<int64_t>(output_locations, 0);

    cv::Mat temp(model_height, model_width, CV_8UC3, cv::Scalar(0,0,0));

    for (int i = 0; i < model_width; i++){
        for (int j = 0; j < model_height; j++){
            cv::Vec3b color = temp.at<cv::Vec3b>(cv::Point(j,i));
            color[0] = color_map[classes[(i*model_width) + j]*3];
            color[1] = color_map[classes[(i*model_width) + j]*3 + 1];
            color[2] = color_map[classes[(i*model_width) + j]*3 + 2];
            temp.at<cv::Vec3b>(cv::Point(j,i)) = color;
        }
    }

    // now blend the model input and output
    cv::addWeighted(output_image, 0.75, temp, 0.25, 0, output_image);
    // add key overlay
    cv::copyMakeBorder(output_image, output_image, 0, 0, 0, RIGHT_PIXEL_BORDER, cv::BORDER_CONSTANT);

    for (unsigned int i = 0; i < labels.size(); i++){
        cv::putText(output_image, labels[i], cv::Point(325, 16 * (i+1)), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(color_map[(i*3)],color_map[(i*3)+1],color_map[(i*3)+2]), 1);
    }
    
    // now, setup metadata since we modified the output image
    meta.format = IMAGE_FORMAT_RGB;
    meta.width = model_width + RIGHT_PIXEL_BORDER;
    meta.height = model_height;
    meta.stride = meta.width * 3;
    meta.size_bytes = meta.height * meta.width * 3; 

    draw_fps(output_image, last_inference_time, cv::Point(0, 0), 0.25, 0.4, cv::Scalar(0, 0, 0), cv::Scalar(180, 180, 180), true);

    if (en_timing) total_postprocess_time += ((rc_nanos_monotonic_time() - start_time)/1000000.);

    return true;
}

#define CLASSIFICATION_CLASSES 1000

bool InferenceHelper::postprocess_classification(cv::Mat &output_image, double last_inference_time, int tensor_offset){
    start_time = rc_nanos_monotonic_time();

    static std::vector<std::string> labels;
    static size_t label_count;

    if (labels.empty()){
        if (ReadLabelsFile(labels_location, &labels, &label_count) != kTfLiteOk){
            fprintf(stderr, "ERROR: Unable to read labels file\n");
            return false;
        }
    }

    TfLiteTensor* output_locations    = interpreter->tensor(interpreter->outputs()[0]);
    uint8_t* confidence_tensor  = TensorData<uint8_t>(output_locations, 0);

    std::vector<uint8_t> confidences;
    confidences.assign(confidence_tensor+tensor_offset, confidence_tensor+CLASSIFICATION_CLASSES+tensor_offset);
 
    uint8_t best_prob = *std::max_element(confidences.begin(),confidences.end());
    int best_class = std::max_element(confidences.begin(),confidences.end()) - confidences.begin();

    fprintf(stderr, "class: %s, prob: %d\n", labels[best_class].c_str(), best_prob);
    cv::putText(output_image, labels[best_class], cv::Point(input_width/3, 25), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 1);

    draw_fps(output_image, last_inference_time, cv::Point(0, 0), 0.5, 2, cv::Scalar(0, 0, 0), cv::Scalar(180, 180, 180), true);

    if (en_timing) total_postprocess_time += ((rc_nanos_monotonic_time() - start_time)/1000000.);

    return true;
}

static const std::vector<std::pair<int32_t, int32_t>> kJointLineList {
    /* face */
    {0, 2},
    {2, 4},
    {0, 1},
    {1, 3},
    /* body */
    {6, 5},
    {5, 11},
    {11, 12},
    {12, 6},
    /* arm */
    {6, 8},
    {8, 10},
    {5, 7},
    {7, 9},
    /* leg */
    {12, 14},
    {14, 16},
    {11, 13},
    {13, 15},
};

bool InferenceHelper::postprocess_posenet(cv::Mat &output_image, double last_inference_time){
    start_time = rc_nanos_monotonic_time();

    static float confidence_threshold = 0.2;

    TfLiteTensor* output_locations    = interpreter->tensor(interpreter->outputs()[0]);
    float* pose_tensor  = TensorData<float>(output_locations, 0);

    std::vector<int32_t> x_coords;
    std::vector<int32_t> y_coords;
    std::vector<float> confidences;

    for (int i = 0; i < 17; i++){
        x_coords.push_back(static_cast<int32_t>(pose_tensor[i * 3 + 1] * input_width));
        y_coords.push_back(static_cast<int32_t>(pose_tensor[i * 3] * input_height));
        confidences.push_back(pose_tensor[i * 3 + 2]);
    }

    for (const auto& jointLine : kJointLineList) {
        if (confidences[jointLine.first] >= confidence_threshold && confidences[jointLine.second] >= confidence_threshold) {
            int32_t x0 = x_coords[jointLine.first];
            int32_t y0 = y_coords[jointLine.first];
            int32_t x1 = x_coords[jointLine.second];
            int32_t y1 = y_coords[jointLine.second];
            cv::line(output_image, cv::Point(x0, y0), cv::Point(x1, y1), cv::Scalar(200, 200, 200), 2);
        }
    }

    for (unsigned int i = 0; i < x_coords.size(); i++){
        if (confidences[i] > confidence_threshold){
            cv::circle(output_image, cv::Point(x_coords[i], y_coords[i]), 4, cv::Scalar( 255, 255, 0 ), cv::FILLED);
        }
    }

    draw_fps(output_image, last_inference_time, cv::Point(0, 0), 0.5, 2, cv::Scalar(0, 0, 0), cv::Scalar(180, 180, 180), true);

    return true;
}

// straight up stolen from https://github.com/iwatake2222/play_with_tflite/blob/master/pj_tflite_det_yolov5/image_processor/detection_engine.cpp
static constexpr int32_t kGridScaleList[] = { 8, 16, 32 };
static constexpr int32_t kGridChannel = 3;
static constexpr int32_t kNumberOfClass = 80;
static constexpr int32_t kElementNumOfAnchor = kNumberOfClass + 5;    // x, y, w, h, bbox confidence, [class confidence]
static constexpr float threshold_box_confidence_   = 0.40;    // not sure if this is too low or high yet
static constexpr float threshold_class_confidence_ = 0.20;    // not sure if this is too low or high yet
static constexpr float threshold_nms_iou_ = 0.50;    // not sure if this is too low or high yet

typedef struct b_box{
    int32_t class_id;
    std::string label;
    float class_conf;
    float detection_conf;   // these two are for ai_detection_t
    float score;            // this is for actually sorting bboxes
    int32_t x;
    int32_t y;
    int32_t w;
    int32_t h;
} b_box;

static void get_bbox(const float* data, float scale_x, float  scale_y, int32_t grid_w, int32_t grid_h, std::vector<b_box>& bbox_list)
{
    int actual_loops = 0;
    int n_skipped = 0;
    int32_t index = 0;
    for (int32_t grid_y = 0; grid_y < grid_h; grid_y++) {
        for (int32_t grid_x = 0; grid_x < grid_w; grid_x++) {
            for (int32_t grid_c = 0; grid_c < kGridChannel; grid_c++) {
                actual_loops++;
                float box_confidence = data[index + 4];

                if (box_confidence >= threshold_box_confidence_) {
                    int32_t class_id = 0;
                    float confidence = 0;
                    float confidence_of_class = 0;
                    for (int32_t class_index = 0; class_index < kNumberOfClass; class_index++) {
                        confidence_of_class = data[index + 5 + class_index];
                        if (confidence_of_class > confidence) {
                            confidence = confidence_of_class;
                            class_id = class_index;
                        }
                    }

                    if (confidence >= threshold_class_confidence_) {
                        int32_t cx = static_cast<int32_t>((data[index + 0] + 0) * scale_x);     // no need to + grid_x
                        int32_t cy = static_cast<int32_t>((data[index + 1] + 0) * scale_y);     // no need to + grid_y
                        int32_t w = static_cast<int32_t>(data[index + 2] * scale_x);            // no need to exp
                        int32_t h = static_cast<int32_t>(data[index + 3] * scale_y);            // no need to exp
                        int32_t x = cx - w / 2;
                        int32_t y = cy - h / 2;
                        b_box bbox = {class_id, "", confidence_of_class, box_confidence, confidence, x, y, w, h};
                        bbox_list.push_back(bbox);
                    }
                }
                else n_skipped++;
                index += kElementNumOfAnchor;
            }
        }
    }
}

static float calc_iou(const b_box& obj0, const b_box& obj1)
{
    int32_t interx0 = (std::max)(obj0.x, obj1.x);
    int32_t intery0 = (std::max)(obj0.y, obj1.y);
    int32_t interx1 = (std::min)(obj0.x + obj0.w, obj1.x + obj1.w);
    int32_t intery1 = (std::min)(obj0.y + obj0.h, obj1.y + obj1.h);
    if (interx1 < interx0 || intery1 < intery0) return 0;

    int32_t area0 = obj0.w * obj0.h;
    int32_t area1 = obj1.w * obj1.h;
    int32_t areaInter = (interx1 - interx0) * (intery1 - intery0);
    int32_t areaSum = area0 + area1 - areaInter;

    return static_cast<float>(areaInter) / areaSum;
}

static void nms(std::vector<b_box>& bbox_list, std::vector<b_box>& bbox_nms_list, float threshold_nms_iou, bool check_class_id)
{
    std::sort(bbox_list.begin(), bbox_list.end(), [](b_box const& lhs, b_box const& rhs) {
        if (lhs.score > rhs.score) return true;
        return false;
        });

    std::unique_ptr<bool[]> is_merged(new bool[bbox_list.size()]);
    for (size_t i = 0; i < bbox_list.size(); i++) is_merged[i] = false;
    for (size_t index_high_score = 0; index_high_score < bbox_list.size(); index_high_score++) {
        std::vector<b_box> candidates;
        if (is_merged[index_high_score]) continue;
        candidates.push_back(bbox_list[index_high_score]);
        for (size_t index_low_score = index_high_score + 1; index_low_score < bbox_list.size(); index_low_score++) {
            if (is_merged[index_low_score]) continue;
            if (check_class_id && bbox_list[index_high_score].class_id != bbox_list[index_low_score].class_id) continue;
            if (calc_iou(bbox_list[index_high_score], bbox_list[index_low_score]) > threshold_nms_iou) {
                candidates.push_back(bbox_list[index_low_score]);
                is_merged[index_low_score] = true;
            }
        }
        bbox_nms_list.push_back(candidates[0]);
    }
}

bool InferenceHelper::postprocess_yolov5(cv::Mat &output_image, std::vector<ai_detection_t>& detections_vector, double last_inference_time){
    start_time = rc_nanos_monotonic_time();

    // yolo has just one fat float output tensor
    TfLiteTensor* output_locations    = interpreter->tensor(interpreter->outputs()[0]);
    float* output_tensor  = TensorData<float>(output_locations, 0);

    std::vector<b_box> bbox_list;

    for (const auto& scale : kGridScaleList) {
        int32_t grid_w = model_width / scale;
        int32_t grid_h = model_height / scale;
        float scale_x = static_cast<float>(input_width);
        float scale_y = static_cast<float>(input_height);
        get_bbox(output_tensor, scale_x, scale_y, grid_w, grid_h, bbox_list);
        output_tensor += grid_w * grid_h * kGridChannel * kElementNumOfAnchor; 
    }

    std::vector<b_box> bbox_nms_list;
    nms(bbox_list, bbox_nms_list, threshold_nms_iou_, false);

    static std::vector<std::string> labels;
    static size_t label_count;

    if (labels.empty()){
        if (ReadLabelsFile(labels_location, &labels, &label_count) != kTfLiteOk){
            fprintf(stderr, "ERROR: Unable to read labels file\n");
            return false;
        }
    }

    for (const auto& bbox : bbox_nms_list) {
        cv::putText(output_image, labels[bbox.class_id], cv::Point(bbox.x, bbox.y), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0), 2);
        cv::rectangle(output_image, cv::Rect(bbox.x, bbox.y, bbox.w, bbox.h), get_color_from_id(bbox.class_id), 2);
         // setup ai detection for this detection
        ai_detection_t curr_detection;
        curr_detection.magic_number = AI_DETECTION_MAGIC_NUMBER;
        curr_detection.timestamp_ns = rc_nanos_monotonic_time();
        curr_detection.class_id = bbox.class_id;
        curr_detection.frame_id = num_frames_processed;

        strcpy(curr_detection.class_name, labels[bbox.class_id].c_str());
        strcpy(curr_detection.cam, cam_name.c_str());

        curr_detection.class_confidence = bbox.class_conf;
        curr_detection.detection_confidence = bbox.detection_conf;
        curr_detection.x_min = bbox.x;
        curr_detection.y_min = bbox.y;
        curr_detection.x_max = bbox.x + bbox.w;
        curr_detection.y_max = bbox.y + bbox.h;

        // fill the vector
        detections_vector.push_back(curr_detection);
    }

    draw_fps(output_image, last_inference_time, cv::Point(0, 0), 0.5, 2, cv::Scalar(0, 0, 0), cv::Scalar(180, 180, 180), true);

    if (en_timing) total_postprocess_time += ((rc_nanos_monotonic_time() - start_time)/1000000.);

    return true;
}


///////////////// ADDING CUSTOM YOLOv5 MODEL - NIST YOLOv5 MODEL /////////////////////////////////////////////////////////////////////////////////////
static constexpr int32_t kNumberOfClass_NIST = 8; // Changed this to 8 number of classes representing 8 buckets to be detected
static constexpr int32_t kElementNumOfAnchor_NITS = kNumberOfClass_NIST + 5;    // x, y, w, h, bbox confidence, [class confidence]




bool InferenceHelper::postprocess_yolo_Nist(cv::Mat &output_image, std::vector<ai_detection_t>& detections_vector, double last_inference_time){
    start_time = rc_nanos_monotonic_time();

    // yolo has just one fat float output tensor
    TfLiteTensor* output_locations    = interpreter->tensor(interpreter->outputs()[0]);
    float* output_tensor  = TensorData<float>(output_locations, 0);

    std::vector<b_box> bbox_list;

    for (const auto& scale : kGridScaleList) {
        int32_t grid_w = model_width / scale;
        int32_t grid_h = model_height / scale;
        float scale_x = static_cast<float>(input_width);
        float scale_y = static_cast<float>(input_height);
        get_bbox(output_tensor, scale_x, scale_y, grid_w, grid_h, bbox_list);
        output_tensor += grid_w * grid_h * kGridChannel * kElementNumOfAnchor_NITS; 
    }

    std::vector<b_box> bbox_nms_list;
    nms(bbox_list, bbox_nms_list, threshold_nms_iou_, false);

    static std::vector<std::string> labels;
    static size_t label_count;

    if (labels.empty()){
        if (ReadLabelsFile(labels_location, &labels, &label_count) != kTfLiteOk){
            fprintf(stderr, "ERROR: Unable to read labels file\n");
            return false;
        }
    }

    for (const auto& bbox : bbox_nms_list) {
        cv::putText(output_image, labels[bbox.class_id], cv::Point(bbox.x, bbox.y), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0), 2);
        cv::rectangle(output_image, cv::Rect(bbox.x, bbox.y, bbox.w, bbox.h), get_color_from_id(bbox.class_id), 2);
         // setup ai detection for this detection
        ai_detection_t curr_detection;
        curr_detection.magic_number = AI_DETECTION_MAGIC_NUMBER;
        curr_detection.timestamp_ns = rc_nanos_monotonic_time();
        curr_detection.class_id = bbox.class_id;
        curr_detection.frame_id = num_frames_processed;

        strcpy(curr_detection.class_name, labels[bbox.class_id].c_str());
        strcpy(curr_detection.cam, cam_name.c_str());

        curr_detection.class_confidence = bbox.class_conf;
        curr_detection.detection_confidence = bbox.detection_conf;
        curr_detection.x_min = bbox.x;
        curr_detection.y_min = bbox.y;
        curr_detection.x_max = bbox.x + bbox.w;
        curr_detection.y_max = bbox.y + bbox.h;

        // fill the vector
        detections_vector.push_back(curr_detection);
    }

    draw_fps(output_image, last_inference_time, cv::Point(0, 0), 0.5, 2, cv::Scalar(0, 0, 0), cv::Scalar(180, 180, 180), true);

    if (en_timing) total_postprocess_time += ((rc_nanos_monotonic_time() - start_time)/1000000.);

    return true;
}

//////////////////////// END OF CUSTOM MODEL ///////////////////////////////////////////////////////

void InferenceHelper::print_summary_stats(){
    if (en_timing){
        fprintf(stderr, "\n------------------------------------------\n");
        fprintf(stderr, "TIMING STATS (on %d processed frames)\n", num_frames_processed);
        fprintf(stderr, "------------------------------------------\n");
        fprintf(stderr, "Preprocessing Time  -> Total: %6.2fms, Average: %6.2fms\n", (double)(total_preprocess_time), (double)((total_preprocess_time/(num_frames_processed))));
        fprintf(stderr, "Inference Time      -> Total: %6.2fms, Average: %6.2fms\n", (double)(total_inference_time), (double)((total_inference_time/(num_frames_processed))));
        fprintf(stderr, "Postprocessing Time -> Total: %6.2fms, Average: %6.2fms\n", (double)(total_postprocess_time), (double)((total_postprocess_time/(num_frames_processed))));
        fprintf(stderr, "------------------------------------------\n");
    } 
}

InferenceHelper::~InferenceHelper(){    
    free(resize_output);

    #ifdef BUILD_QRB5165
    if (gpu_delegate) TfLiteGpuDelegateV2Delete(gpu_delegate);
    if (xnnpack_delegate) TfLiteXNNPackDelegateDelete(xnnpack_delegate);
    if (nnapi_delegate) delete(nnapi_delegate);
    #endif    

    print_summary_stats();
}
