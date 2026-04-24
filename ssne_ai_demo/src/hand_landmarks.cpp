#include "../include/hand_landmarks.hpp"
#include <cstring>
#include <iostream>

/**
 * @brief 清空HandLandMarksResult的内容
 * @description 清空所有变量内容，但保留内存分配
 */
void HandLandMarksResult::Clear() {
    landmarks_local.clear();
    landmarks_world.clear();
    confidence.clear();
    flag.clear();
}

/**
 * @brief 预分配内存空间
 * @param size 要保留的元素数量
 * @description 为所有变量预分配内存，提高性能
 */
void HandLandMarksResult::Reserve(int size) {
    landmarks_local.reserve(size);
    landmarks_world.reserve(size);
    confidence.reserve(size);
    flag.reserve(size);
}

/**
 * @brief 调整HandLandMarksResult的大小
 * @param size 新的元素数量
 * @description 调整数量
 */
void HandLandMarksResult::Resize(int size) {
    landmarks_local.reserve(size);
    landmarks_world.reserve(size);
    confidence.reserve(size);
    flag.reserve(size);
}

/**
 * @brief 释放FaceDetectionResult的内存
 * @description 使用swap技巧释放vector占用的内存
 */
void HandLandMarksResult::Free() {
  std::vector<std::array<float, 63>>().swap(landmarks_local);
  std::vector<float>().swap(confidence);
  std::vector<float>().swap(flag);
  std::vector<std::array<float, 63>>().swap(landmarks_world);
}

/**
 * @brief HandLandMarksResult的拷贝构造函数
 * @param res 要拷贝的HandLandMarksResult对象
 * @description 深拷贝检测结果的所有数据
 */
HandLandMarksResult::HandLandMarksResult(const HandLandMarksResult& res) {
    landmarks_local.assign(res.landmarks_local.begin(), res.landmarks_local.end());
    landmarks_world.assign(res.landmarks_world.begin(), res.landmarks_world.end());
    confidence.assign(res.confidence.begin(), res.confidence.end());
    flag.assign(res.flag.begin(), res.flag.end());
}




// 用于保存最后一帧的全局静态变量
static ssne_tensor_t g_last_img;
static ssne_tensor_t g_last_pipe_input;
static bool g_has_frame = false;

/*
*   传入图像，进行推理，完整流程实现 预处理、推理、后处理
*   img_in 传入图像 NCHW形式
*   result 输出结果
*/
void HLMD_MODEL::Predict(ssne_tensor_t* img_in, HandLandMarksResult* result)
{
    // offline图像tensor初始化：对输入图像进行预处理（resize、归一化等）
    int ret = RunAiPreprocessPipe(pipe_offline, *img_in, inputs[0]);
    // printf("ret: %d\n", ret);
    if (ret != 0) {
        printf("[ERROR] Failed to run AI preprocess pipe!\n");
        printf("ret: %d\n", ret);
        return;
    }

    // 保存当前帧的图像（每次调用都更新，最终保留最后一帧）
    g_last_img = *img_in;
    g_last_pipe_input = inputs[0];
    g_has_frame = true;

    // 前向推理：在NPU上执行模型推理
    if (ssne_inference(model_id, 1, inputs))
    {
        fprintf(stderr, "ssne inference fail!\n");
    }

    // 获取模型输出：4个输出tensor（1+1+1+1）
    ssne_getoutput(model_id, 4, outputs);

    float *Id0 = (float*)get_data(outputs[0]);  // 2.5D输出
    float *Id1 = (float*)get_data(outputs[1]);  // 置信度 
    float *Id2 = (float*)get_data(outputs[2]);  // 手性
    float *Id3 = (float*)get_data(outputs[3]);  // 世界坐标系

    // std::vector<std::array<float, 63>> landmarks;
    // std::vector<float> confidence;
    // std::vector<float> flag;
    // std::vector<std::array<float, 63>> landmarks_w;

    //执行后处理过程
    //...

    //保存结果
    result->Clear();
    result->Reserve(1);

    std::array<float, 63> lm;
    memcpy(lm.data(), Id0, sizeof(float) * 63);
    result->landmarks_local.emplace_back(lm);

    result->confidence.emplace_back(Id1[0]);

    result->flag.emplace_back(Id2[0]);

    // std::array<float, 63> lmw;
    memcpy(lm.data(), Id3, sizeof(float) * 63);
    result->landmarks_world.emplace_back(lm);

}

/**
 * @brief 释放资源
 * @description 释放所有tensor、预处理管道和计时器资源
 */
void HLMD_MODEL::Release()
{   
    // 保存最后一帧的图像（如果有的话）
    if (g_has_frame) {
        printf("[INFO] Saving last frame images...\n");
        save_tensor(g_last_img, "dbg_in.raw");
        save_tensor(g_last_pipe_input, "dbg_in_pipe.raw");
        printf("[INFO] Last frame saved successfully!\n");
    }

    // 释放输入和输出tensor
    release_tensor(inputs[0]);
    release_tensor(outputs[0]);
    release_tensor(outputs[1]);
    release_tensor(outputs[2]);
    release_tensor(outputs[3]);
    // 释放预处理管道
    ReleaseAIPreprocessPipe(pipe_offline);
}

/* debug */
/**
 * @brief 保存图像数据到二进制文件（调试用）
 * @param data 图像数据指针
 * @param w 图像宽度
 * @param h 图像高度
 * @param filename 输出文件名
 * @description 将图像数据保存为二进制文件，用于调试和验证
 */
void HLMD_MODEL::saveImageBin(const void* data, int w, int h, const char* filename) {
    FILE* file = fopen(filename, "wb");
    if (file != nullptr) {
        fwrite(&w, sizeof(int), 1, file);      // 写入宽度
        fwrite(&h, sizeof(int), 1, file);      // 写入高度
        fwrite(data, sizeof(char), w * h, file);  // 写入图像数据
        fclose(file);
        std::cout << "write file " << filename << " successfully!" << std::endl;
    }
    else {
        std::cerr << "failed to write " << filename << std::endl;
    }
}
/* debug */
/**
 * @brief 保存浮点数组到二进制文件（调试用）
 * @param data 浮点数组指针
 * @param length 数组长度
 * @param filename 输出文件名
 * @description 将浮点数组保存为二进制文件，用于调试和验证
 */
void HLMD_MODEL::saveFloatBin(const float* data, int length, const char* filename) {
    FILE* file = fopen(filename, "wb");
    if (file != nullptr) {
        fwrite(&length, sizeof(int), 1, file);        // 写入数组长度
        fwrite(data, sizeof(float), length, file);     // 写入浮点数据
        fclose(file);
        std::cout << "write file " << filename << " successfully!" << std::endl;
    }
    else {
        std::cerr << "failed to write " << filename << std::endl;
    }
}

/**
 * @brief 初始化SCRFD检测器
 * @param model_path 模型文件路径
 * @param in_img_shape 裁剪后图像尺寸 [宽度, 高度]
 * @param in_det_shape 检测输入尺寸 [宽度, 高度]
 * @param in_use_kps 是否使用关键点检测
 * @param in_box_len 检测框长度
 * @description 初始化检测器的各种参数，加载模型，生成anchor boxes
 */
void HLMD_MODEL::Initialize(std::string& model_path, std::array<int, 2>* in_img_shape, std::array<int, 2>* in_det_shape) 
{
    img_shape = *in_img_shape;  // 原始图像尺寸
    det_shape = *in_det_shape;  // 检测输入尺寸

    // 加载模型
    char* model_path_char = const_cast<char*>(model_path.c_str());
    model_id = ssne_loadmodel(model_path_char, SSNE_STATIC_ALLOC);

    // 创建模型输入tensor
    uint32_t det_width = static_cast<uint32_t>(det_shape[0]);
    uint32_t det_height = static_cast<uint32_t>(det_shape[1]);
    inputs[0] = create_tensor(det_width, det_height, SSNE_Y_8, SSNE_BUF_AI);
}