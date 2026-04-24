/*
 * @Filename: demo_hand.cpp
 * @Author: Hongying He
 * @Email: hongying.he@smartsenstech.com
 * @Date: 2025-12-30 14-57-47
 * @Copyright (c) 2025 SmartSens
 */
#include <fstream>
#include <iostream>
#include <cstring>
#include <thread>
#include <mutex>
#include <fcntl.h>
#include <regex>
#include <dirent.h>
#include <unistd.h>
#include "include/utils.hpp"
#include "include/hand_landmarks.hpp"

using namespace std;

// 全局退出标志（线程安全）
bool g_exit_flag = false;
// 保护退出标志的互斥锁
std::mutex g_mtx;

// OSD 贴图结构体
struct osdInfo {
    std::string filename; // OSD 文件名
    uint16_t x;           // 起始坐标 x
    uint16_t y;           // 起始坐标 y
};

/**
 * @brief 键盘监听程序，用于结束demo
 */
void keyboard_listener() {
    std::string input;
    std::cout << "键盘监听线程已启动，输入 'q' 退出程序..." << std::endl;

    while (true) {
        // 读取键盘输入（会阻塞直到有输入）
        std::cin >> input;

        // 加锁修改退出标志
        std::lock_guard<std::mutex> lock(g_mtx);
        if (input == "q" || input == "Q") {
            g_exit_flag = true;
            std::cout << "检测到退出指令，通知主线程退出..." << std::endl;
            break;
        } else {
            std::cout << "输入无效（仅 'q' 有效），请重新输入：" << std::endl;
        }
    }
}

/**
 * @brief 检查退出标志的辅助函数（线程安全）
 * @return 是否需要退出
 */
bool check_exit_flag() {
    std::lock_guard<std::mutex> lock(g_mtx);
    return g_exit_flag;
}

/**
 * @brief 手部关键点检测演示程序主函数
 * @return 执行结果，0表示成功
 */
int main() {
    /******************************************************************************************
     * 1. 参数配置
     ******************************************************************************************/

    // 图像尺寸配置（根据镜头参数修改）
    int img_width = 720;    // 输入图像宽度
    int img_height = 1280;  // 输入图像高度

    // 模型配置参数
    array<int, 2> det_shape = {224, 224};  // 检测模型输入尺寸
    string path_hand = "/app_demo/app_assets/models/hand_landmarks.m1model";  // 手部关键点模型路径（修改）

    // OSD 信息
    static osdInfo osds[3] = {
        {"si.ssbmp", 10, 10},
        {"te.ssbmp", 90, 10},
        {"wei.ssbmp", 170, 10}
    };

    /******************************************************************************************
     * 2. 系统初始化
     ******************************************************************************************/

    // SSNE初始化
    if (ssne_initial()) {
        fprintf(stderr, "SSNE initialization failed!\n");
    }

    // 图像处理器初始化
    array<int, 2> img_shape = {img_width, img_height};  // 原始图像尺寸
    array<int, 2> crop_shape = {720,720}//裁剪尺寸（保持图像resize后比例不变）
    const int crop_offset_y = 280;  // 裁剪时Y方向的偏移量
    // 原图: 720×1280, 模型输入图：640×480
    // 为了保证模型输入图经过resize后比例不变，需要先将原图裁剪为crop图: 720×540 (上下各裁370px)
    IMAGEPROCESSOR processor;
    processor.Initialize(&img_shape);  // 初始化图像处理器（配置原图尺寸）

    // 手部关键点模型初始化（修改：替换人脸检测器为手部检测器）
    HLMD_MODEL model;  // 假设SDK提供的手部关键点检测器类名
    model.Initialize(path_det, &crop_shape, &det_shape);
    // 手部关键点结果初始化（修改：替换人脸结果为手部关键点结果）
    HandLandMarksResult* det_result = new HandLandMarksResult;
       
    // OSD可视化器初始化（用于绘制关键点）
    VISUALIZER visualizer;
    visualizer.Initialize(img_shape, "shared_colorLUT.sscl");  // 初始化可视化器（配置图像尺寸和位图LUT）

    // 系统稳定等待
    cout << "sleep for 0.2 second!" << endl;
    sleep(0.2);  // 等待系统稳定

    // OSD 贴图
    visualizer.DrawBitmap(osds[0].filename, "shared_colorLUT.sscl", osds[0].x, osds[0].y, 2);

    uint16_t num_frames = 0;  // 帧计数器
    uint8_t osd_index = 0; // osd 贴图 index
    ssne_tensor_t img_sensor;  // 图像tensor定义

    // 创建键盘监听线程
    std::thread listener_thread(keyboard_listener);

    /******************************************************************************************
     * 3. 主处理循环
     ******************************************************************************************/
    while (!check_exit_flag()) {

        // 从sensor获取图像（单通道灰度图 720x540）
        processor.GetImage(&img_sensor);

        // ===================== 单通道灰度图 → 3通道数据 =====================
        const int img_w = 720;
        const int img_h = 720;
        uint8_t* gray_data = (uint8_t*)img_sensor.data;  // 原始单通道数据指针

        // 创建3通道RGB数据缓冲区（R/G/B三个通道值相同）
        int rgb_data_size = img_w * img_h * 3;
        uint8_t* rgb_data = new uint8_t[rgb_data_size];

        // 逐像素复制：灰度值填充三个通道
        for (int i = 0; i < img_w * img_h; i++) {
            rgb_data[i*3 + 0] = gray_data[i];  // R通道
            rgb_data[i*3 + 1] = gray_data[i];  // G通道
            rgb_data[i*3 + 2] = gray_data[i];  // B通道
        }

        // 构造临时tensor，仅替换数据指针
        ssne_tensor_t img_rgb = img_sensor;
        img_rgb.data = rgb_data;  // 仅替换为3通道数据，其他成员保持不变

        // 手部关键点模型推理（修改：调用手部检测器Predict）
        hand_detector.Predict(&img_rgb, hand_result, 0.4f);  // 0.4f为置信度阈值

        // 释放临时内存（防止内存泄漏）
        delete[] rgb_data;

        /**********************************************************************************
         * 3.1 判断是否有检测到手部关键点
         **********************************************************************************/
           
        // 存储用于绘图的坐标 (注意：这里假设 visualizer 需要的是 2D 点 [x, y])
        std::vector<std::vector<std::array<float, 2>>> landmarks_to_draw;

        // 遍历每一只手
        // hand_idx: 第几只手
        for (size_t hand_idx = 0; hand_idx < hand_result->landmarks_local.size(); ++hand_idx) {
            
            // 获取当前手的 63 个浮点数据引用
            const std::array<float, 63>& hand_data = hand_result->landmarks_local[hand_idx];
            
            // 获取当前手的置信度 (可选，用于过滤)
            float conf = hand_result->confidence[hand_idx];
            if (conf < 0.5f) continue; // 低置信度过滤

            std::vector<std::array<float, 2>> single_hand_2d;

            // 遍历 21 个关键点
            for (int kp_idx = 0; kp_idx < 21; ++kp_idx) {
                // 从 63 维数组中取出 x, y
                // 索引逻辑: kp_idx * 3 + 0 (x), +1 (y), +2 (z)
                float x_raw = hand_data[kp_idx * 3 + 0];
                float y_raw = hand_data[kp_idx * 3 + 1];
                // z 坐标通常用于深度，这里绘图可能不需要
                
                // -------------------------------------------------------------
                // 注意：这里需要确认 SDK 输出的 x_raw 是归一化坐标还是像素坐标
                // 如果是归一化坐标 (0~1)，需要乘以图像宽度和高度
                // float x_pixel = x_raw * img_width;
                // float y_pixel = y_raw * img_height;
                //
                // 结合原代码注释，这里假设输出已经是基于 Crop 图 (720x540) 的像素坐标
                // -------------------------------------------------------------

                // 坐标转换 (加上裁剪偏移量 370)
                float x_orig = x_raw;
                float y_orig = y_raw + crop_offset_y;

                single_hand_2d.push_back({x_orig, y_orig});
            }
            
            landmarks_to_draw.push_back(single_hand_2d);
        }

        // 5. OSD 绘图
        visualizer.DrawLandmarks(landmarks_to_draw);
    
   
        num_frames += 1;  // 帧计数器递增

        // OSD 贴图
        osd_index = (num_frames / 10) % 3;
        visualizer.DrawBitmap(osds[osd_index].filename, "shared_colorLUT.sscl", osds[osd_index].x, osds[osd_index].y, 2);
    }

    // 等待监听线程退出，释放资源
    if (listener_thread.joinable()) {
        listener_thread.join();
    }

    /******************************************************************************************
     * 4. 资源释放
     ******************************************************************************************/

    delete hand_result;  // 释放手部关键点结果
    hand_detector.Release();  // 释放手部检测器资源
    processor.Release();  // 释放图像处理器资源
    visualizer.Release();  // 释放可视化器资源

    if (ssne_release()) {
        fprintf(stderr, "SSNE release failed!\n");
        return -1;
    }

    return 0;
}