#include "mixvpr-hnsw.h"
//opencv
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include "NvInfer.h"
#include "opencv2/core.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include "opencv2/highgui.hpp"
#include "cv_bridge/cv_bridge.h"
//hnswlib
#include <hnswlib/hnswlib.h>
#include <vector>
#include <iostream>
// 64-bit int
// using idx_t = faiss::Index::idx_t;
using namespace std;
int width_adj ,height_adj;

MixVPR::MixVPR( const std::string engine_path) {
    TRTLogger logger;
    // 创建两个事件，一个开始事件，一个结束事件
    // cudaEvent_t start, stop;
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);
    // 在默认的stream记录开始事件
    // cudaEventRecord(start);

    //-------------------------- 1.加载模型 ----------------------------
    static auto engine_data = load_file(engine_path);

    //------------------------ 2.创建 runtime --------------------------
    static auto runtime = make_nvshared(nvinfer1::createInferRuntime(Logger));

    //-------------------------- 3.反序列化 ---------------------------
    static auto engine = make_nvshared(runtime->deserializeCudaEngine(engine_data.data(), engine_data.size()));

    if (engine == nullptr) {
        printf("Deserialize cuda engine failed.\n");
        runtime->destroy();
        return ;
    }

    //-------------------- 4.从engine上创建 执行上下文
    static auto execution_context = make_nvshared(engine->createExecutionContext());


    //------------------------- 5.创建CUDA 流 ------------------------
    // cudaStream_t stream = nullptr;
    // checkRuntime(cudaStreamCreate(&stream));

    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    // float time;
    // cudaEventElapsedTime(&time, start, stop);
    // printf("Model loading time = %f\n", time);
}


void MixVPR::inference(const cv::Mat& image,
                       std::vector<frame> &frame_set,
                       std::vector<float> &des_db,
                       const char *engine_path,
                       cudaStream_t stream)
                       
 {

    float time;
    frame frame;
    
    TRTLogger logger;
    // 创建两个事件，一个开始事件，一个结束事件
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // 在默认的stream记录开始事件
    cudaEventRecord(start);

    //-------------------------- 1.加载模型 ----------------------------
    static bool is_first = true;
    static vector<unsigned char> engine_data;
    if (is_first) {
        engine_data =  load_file(engine_path);
        is_first = false;
    }
   // static auto engine_data = load_file(engine_path);

    //------------------------ 2.创建 runtime --------------------------
    static auto runtime = make_nvshared(nvinfer1::createInferRuntime(logger));

    //-------------------------- 3.反序列化 ---------------------------
    static auto engine = make_nvshared(runtime->deserializeCudaEngine(engine_data.data(), engine_data.size()));

    if (engine == nullptr) {
        printf("Deserialize cuda engine failed.\n");
        runtime->destroy();
        return ;
    }

    //-------------------- 4.从engine上创建 执行上下文
    static auto execution_context = make_nvshared(engine->createExecutionContext());

    // //------------------------- 5.创建CUDA 流 ------------------------
    // cudaStream_t stream = nullptr;
    // checkRuntime(cudaStreamCreate(&stream));

    //-------------------------6.准备数据----------------------------
    //图像格式
    int input_batch= 1;
    int input_channel = 3;
    int input_height = 320;
    int input_width = 320;
    int input_numel = input_batch * input_channel * input_height * input_width;
    int image_num=1;

    float* input_data_host= nullptr;
    float* input_data_device = nullptr;
    checkRuntime(cudaMallocHost(&input_data_host, image_num*input_numel * sizeof(float)));
    checkRuntime(cudaMalloc(&input_data_device, image_num*input_numel * sizeof(float)));

    float mean[] = {0.406, 0.456, 0.485};
    float std[]= {0.225, 0.224, 0.229};

    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(input_width, input_height));
    int image_area = resized_image.cols * resized_image.rows;
    unsigned char* pimage = resized_image.data; //BGRBGRBGR TO BBBGGGRRR   用地址偏移做索引

    //------------------使用多张图片做测试------------------------
    for(int i =0;i<image_num;i++)
    {
        float* phost_b = input_data_host + image_area *(3*i);
        float* phost_g = input_data_host + image_area *(3*i+1) ;
        float* phost_r = input_data_host + image_area *(3*i+2) ;
        pimage = image.data;  //执行完一张图片后，下一个循环重新指向图片的收个像素的指针

        for(int j = 0; j < image_area; ++j, pimage += 3){
            //注意这里的顺序 rgb 调换了

            *phost_r++ = (pimage[0] / 255.0f - mean[0]) / std[0];
            *phost_g++ = (pimage[1] / 255.0f - mean[1]) / std[1];
            *phost_b++ = (pimage[2] / 255.0f - mean[2]) / std[2];
        }

    }

    checkRuntime(cudaMemcpyAsync(input_data_device, input_data_host,
                                 image_num*input_numel * sizeof(float),
                                 cudaMemcpyHostToDevice, stream));

    const int des_dim = 512;
    float output_data_host[des_dim];
    float* output_data_device = nullptr;

    checkRuntime(cudaMalloc(&output_data_device, des_dim * sizeof(float)));

    //明确当前推理时，使用的数据输入大小
    
    auto input_dims = execution_context->getBindingDimensions(0);
    input_dims.d[0] = input_batch;
    execution_context->setBindingDimensions(0, input_dims);


    //用一个指针数组指定 input 和 output 在 GPU 内存中的指针
    float* bindings[] = {input_data_device, output_data_device};


    //------------------------ 7.推理 --------------------------
    bool success = execution_context->enqueueV2((void**)bindings, stream,
                                                nullptr);


    checkRuntime(cudaMemcpyAsync(output_data_host, output_data_device,
                                 sizeof(output_data_host), cudaMemcpyDeviceToHost, stream));

    checkRuntime(cudaStreamSynchronize(stream));

    //解算描述子
    frame.img_global_des_vec.insert(frame.img_global_des_vec.begin(),output_data_host, output_data_host + 512);
    des_db.insert(des_db.end(),output_data_host,output_data_host+512);

//--------------------8.按照创建相反的顺序释放内存 ----------------------

    // 推理完成后，记录结束事件
    // cudaEventRecord(stop);
    // 等待结束事件完成
    // cudaEventSynchronize(stop);

    // 计算两个event之间的间隔时间
    // cudaEventElapsedTime(&time, start, stop);
    // printf("MIX time = %f ms\n",time);

    frame.extract_time = time;
    // std::cout << "frame size = " << sizeof(frame) << std::endl;
    frame_set.push_back(frame);


    // 销毁事件
    // cudaEventDestroy(start);
    // cudaEventDestroy(stop);

    // checkRuntime(cudaStreamDestroy(stream));
    checkRuntime(cudaFreeHost(input_data_host));
    checkRuntime(cudaFree(input_data_device));
    checkRuntime(cudaFree(output_data_device));

}



void MixVPR::test_in_datasets(const std::string filepath,
                              std::vector<cv::String > &namearray,
                              std::vector<frame> &frame_set,
                              std::map<int,frame> &frame_des_dataset,
                              std::vector<float> &des_db,
                              cudaStream_t stream)
{

    cv::glob(filepath,namearray);

    std::sort(namearray.begin(), namearray.end());

    for(size_t i=0 ; i < namearray.size();i++) {
        auto image = cv::imread(namearray[i]);
        
        inference(image,frame_set,des_db,"/docker-ros/ws/MixVPR/LOGS/mix_512FP16.engine",stream);
//        cv::waitKey(0);
        frame_set.at(i).raw_image = image;
        frame_set.at(i).frame_id = std::to_string(i);
        frame_set.at(i).filename = namearray[i];
        frame_des_dataset[i] = frame_set[i];
    }

}

void MixVPR::sort_vec_hnsw(std::vector<frame> &frame_set, std::vector<float> &des_db,
                           std::vector<frame> &frame_set_q, std::vector<float> &des_q)
{
    
    int dim = 512;   // dimension
    int max_elements = frame_set.size();
    int M = 32;
    int ef_construction = 400;
    int ef_search = 200;  
    int k = 5;
    
    hnswlib::L2Space space(dim);

    hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);

    int nb = frame_set.size();

    // 1. 向HNSW索引中添加所有特征向量
    for (int i = 0; i < nb; i++) {
        // 获取当前帧的描述子
        float* descriptor = frame_set[i].img_global_des_vec.data();
        // 将描述子插入HNSW索引
        alg_hnsw->addPoint(descriptor, i); // 每个描述子与该帧的索引相关联
    }

    // 设置搜索时的邻居数量
    alg_hnsw->setEf(ef_search);

    // 对每个查询图像执行 k-NN 搜索
    for (size_t i = 0; i < frame_set_q.size(); i++) {
        // 获取查询图像的描述子
        float* query_descriptor = frame_set_q[i].img_global_des_vec.data();

        // 执行 k-NN 搜索，返回最相似的 3 个邻居
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnn(query_descriptor, k);

        // 输出最相似的 k 个结果
        std::cout << "----------- Query Frame " << frame_set_q[i].frame_id << " -----------" << std::endl;
        std::cout << "Top " << k << " nearest neighbors:" << std::endl;

        // 打印最相似的 3 张图像
        while (!result.empty()) {
            std::cout << "Neighbor " << k - result.size() + 1 << 
            ": ID = " << result.top().second << 
            // ": timestamp = "<< frame_set[result.top().second].filename<< 
            ", Distance = " << result.top().first << std::endl;

            frame_set_q[i].top_k_ind.push_back(result.top().second); // 保存结果
            result.pop();
        }
    }
    // std::cout << "for done" << std::endl;

    // 删除HNSW索引对象
    delete alg_hnsw;
}


void MixVPR::run(std::string &datapath,std::string &querypath)
{
    //加载数据集
    cudaStream_t stream = nullptr;
    checkRuntime(cudaStreamCreate(&stream));

    std::vector<cv::String > namearray,namearray_q;
    std::vector<frame> frame_set,frame_set_q;
    std::map<int,frame> frame_des_dataset,frame_des_q;
    std::vector<float> des_db,des_q;


    auto startTime = std::chrono::high_resolution_clock::now();

    //数据集中进行测试
    test_in_datasets(datapath,namearray,frame_set,frame_des_dataset,des_db,stream);
    test_in_datasets(querypath,namearray_q,frame_set_q,frame_des_q,des_q,stream);
    checkRuntime(cudaStreamDestroy(stream));
    auto endTime = std::chrono::high_resolution_clock::now();
    float cpu_time = std::chrono::duration<float, std::milli>(endTime - startTime).count();

    //--------------------打印数据-------------------
    int round = frame_set.size();

    float stream_time =0 ;
    for(int i =0;i<frame_set.size();i++ )
    {
        stream_time += frame_set.at(i).extract_time;
    }

    // printf("time in stream = %f ms\n",(stream_time)/(round));
    // printf("初始化和数据库构建时间: %f ms\n",cpu_time);

    

    auto startTime2 = std::chrono::high_resolution_clock::now();
    sort_vec_hnsw(frame_set,des_db,frame_set_q,des_q);

    auto endTime2 = std::chrono::high_resolution_clock::now();
    float faiss_sort_time = std::chrono::duration<float, std::milli>(endTime2 - startTime2).count();
    cout<<"初始化和数据库构建时间: "<< cpu_time <<" ms"<<endl;
    cout<<"平均查询时间: "<<faiss_sort_time/round<<" ms/帧"<<endl;


    static int correct = 0;  // 正确匹配计数
    static int wrong = 0;    // 错误匹配计数
    for(int i =0;i<frame_set_q.size();i++)
    {
        
        cv::Mat img0 = cv::imread(namearray_q[i], -1);
        cv::Mat img1 = cv::imread(namearray[frame_set_q[i].top_k_ind[4]], -1);
        cv::Mat img2;
        cv::hconcat(img0,img1,img2);

        cv::putText(img2, "Query: "+ std::to_string(i+1),
                    cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, 
                    cv::Scalar(0,255,0), 2);
        cv::putText(img2, "Match: " + std::to_string(frame_set_q[i].top_k_ind[2]), 
                    cv::Point(650, 30), cv::FONT_HERSHEY_SIMPLEX, 1, 
                    cv::Scalar(0,255,0), 2);
        cv::imshow("result",img2);
        // cv::waitKey(0);
        int key = cv::waitKey(0);

        if(key == 27) { // ESC键退出
            break;
        } else if(key == 'y' || key == 'Y') {
            correct++;
        } else if(key == 'n' || key == 'N') {
            wrong++;
        }
    
        // 实时显示统计
        std::cout << "\n----- Progress -----" << std::endl;
        std::cout << "Current frame: " << i+1 << "/" << frame_set_q.size() << std::endl;
        std::cout << "Correct: " << correct << "  Wrong: " << wrong << std::endl;
        if(correct + wrong > 0) {
            float acc = (static_cast<float>(correct) / (correct + wrong)) * 100;
            std::cout << "Accuracy: " << std::fixed << std::setprecision(1) 
                     << acc << "%" << std::endl;
        }
    }
}


int main(int argc, char** argv) {

    // 获取图像数据集路径
    // std::string datapath = "/docker-ros/ws/dataset/EuRoC-MAV/database";
    // std::string querypath = "/docker-ros/ws/dataset/EuRoC-MAV/query";
    std::string datapath = "/docker-ros/ws/dataset/loop/database";
    std::string querypath = "/docker-ros/ws/dataset/loop/query";

    // 创建 MixVPR 对象，传入模型文件路径
    MixVPR mixvpr("/docker-ros/ws/MixVPR/LOGS/mix_512FP16.engine");

    // 执行推理任务
    mixvpr.run(datapath,querypath);

    // 打印处理时间等信息
    // std::cout << "Inference completed!" << std::endl;

    return 0;
}