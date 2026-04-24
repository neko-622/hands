#ifndef __HAND_LANDMARKS_HPP__
#define __HAND_LANDMARKS_HPP__

#include <stdio.h>
#include <vector>
#include <array>
#include <string>
#include <math.h>
#include "smartsoc/ssne_api.h"

/*
*手部关键点检测结果
*/
struct HandLandMarksResult
{
    //四个输出头
    
    
    std::vector<std::array<float, 63>> landmarks_local; //Id0
    std::vector<float> confidence; //Id1
    std::vector<float> flag; //Id2
    std::vector<std::array<float, 63>> landmarks_world; //Id3


    HandLandMarksResult() {}
    HandLandMarksResult(const HandLandMarksResult& res);
    // 清空所有变量
    void Clear();

    // 释放内存
    void Free();
    
    // 提前为结构体保留一定的空间 
    void Reserve(int size);
    
    // 修改结构体大小，取前size个元素
    void Resize(int size);
};


class HLMD_MODEL
{
public:
    std::string ModelName() const { return "HLMD_MODEL"; }
    /*
    *   传入图像，进行推理
    *   img_in 传入图像 NCHW形式
    *   result 输出结果
    */
    void Predict(ssne_tensor_t* img_in, HandLandMarksResult* result);
    /*
    *   模型初始化
    *   model_path onnx模型路径，字符串类型。
    *   in_img_shape 输入图像尺寸(w, h)
    *   in_det_shape 检测图像尺寸(w, h)。
    */
    void Initialize(std::string& model_path, std::array<int, 2>* in_img_shape, 
                    std::array<int, 2>* in_det_shape); 
    // 释放资源
    void Release();

    //debug
    void saveImageBin(const void* data, int w, int h, const char* filename);
    void saveFloatBin(const float* data, int length, const char* filename);



private:
    uint16_t model_id = 0; //模型id
    ssne_tensor_t inputs[1]; //输入张量
    ssne_tensor_t outputs[4]; //输出张量
    AiPreprocessPipe pipe_offline = GetAIPreprocessPipe(); //预处理句柄
    // 前处理时，模型推理输入的原始待检测图像尺寸，（width，height）
    std::array<int, 2> img_shape;
    // 前处理时，模型推理需要的待检测图像尺寸，（width，height）
    std::array<int, 2> det_shape;
};



#endif