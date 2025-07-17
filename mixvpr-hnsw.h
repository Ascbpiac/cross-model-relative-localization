
#ifndef VINS_FUSION_DEEP_NET_H
#define VINS_FUSION_DEEP_NET_H

#pragma once
#include "tensorrt_utils.h"

//sys
#include "iostream"
#include "vector"

#include "opencv2/opencv.hpp"

//eigen
#include "eigen3/Eigen/Dense"
#include "eigen3/Eigen/Core"
#include "eigen3/Eigen/Geometry"

//ros
#include "ros/ros.h"
#include <ros/time.h>
#include "sensor_msgs/CompressedImage.h"

// #include "../keyframe.h"

// #include "tensorrt_tools/preprocess_kernel.cuh"

#define mix_engine_path "/docker-ros/ws/MixVPR/LOGS/mix_512FP16.engine"


class frame{
public:
    frame()=default;

    std::string frame_id;
    ros::Time timestamp;

    float extract_time;

    int image_width;
    int image_height;

    cv::String filename;
    cv::Mat raw_image;

    //dim = 512
    std::vector<float> img_global_des_vec;

    std::vector<float> local_descriptor;
    std::vector<float> kpoints;
    std::vector<float> similarity;
    std::vector<int> top_k_ind;
    
    std::vector<std::pair<float,float>> landmarks;
    void init_top_k(int k) {
        top_k_ind.resize(k);
    }
};

class MixVPR{
public:

    TRTLogger Logger;


    explicit MixVPR(const std::string engine_path);
    explicit MixVPR();
    void inference(const cv::Mat& image,
                   std::vector<frame> &frame_set,
                   std::vector<float> &des_db,
                   const char *engine_path,
                   cudaStream_t stream
    );

    void test_in_datasets(const std::string filepath,
                          std::vector<cv::String > &namearray,
                          std::vector<frame> &frame_set,
                          std::map<int,frame> &frame_des_dataset,
                          std::vector<float> &des_db,
                          cudaStream_t stream
    );

    void img_callback(const sensor_msgs::CompressedImage &msg,
                      frame &_frame,
                      std::vector<frame> &frame_set,
                      std::map<int,frame> &frame_des_dataset,
                      std::vector<float> &des_db
    );

    void sort_vec_hnsw(std::vector<frame> &frame_set, std::vector<float> &des_db ,
                        std::vector<frame> &frame_set_q, std::vector<float> &des_q);

    void run(std::string &datapath,std::string &querypath);
private:
    cudaStream_t stream; // 将流作为类成员变量
};



#endif //VINS_FUSION_DEEP_NET_H
