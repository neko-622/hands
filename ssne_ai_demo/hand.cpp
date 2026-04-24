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
	uint16_t x; // 起始坐标 x
	uint16_t y; // 起始坐标 y
};

/**
 * @brief 键盘监听程序，用于结束demo
 */
void keyboard_listener()
{
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
bool check_exit_flag()
{
	std::lock_guard<std::mutex> lock(g_mtx);
	return g_exit_flag;
}

/**
 * @brief 手部关键点检测演示程序主函数
 * @return 执行结果，0表示成功
 */
int main()
{
	/******************************************************************************************
     * 1. 参数配置
     ******************************************************************************************/

	// 图像尺寸配置（根据镜头参数修改）
	int img_width = 720; // 输入图像宽度
	int img_height = 1280; // 输入图像高度

	// 模型配置参数
	array<int, 2> det_shape = { 224, 224 }; // 检测模型输入尺寸
	string path_det = "/app_demo/app_assets/models/hand_landmarks.m1model"; // 手部关键点模型路径

	// OSD 信息
	static osdInfo osds[3] = { { "si.ssbmp", 10, 10 }, { "te.ssbmp", 90, 10 }, { "wei.ssbmp", 170, 10 } };

	/******************************************************************************************
     * 2. 系统初始化
     ******************************************************************************************/

	// SSNE初始化
	if (ssne_initial()) {
		fprintf(stderr, "SSNE initialization failed!\n");
	}

	// 图像处理器初始化
	array<int, 2> img_shape = { img_width, img_height }; // 原始图像尺寸
	array<int, 2> crop_shape = { 720, 720 }; //裁剪尺寸
	const int crop_offset_y = 280; // 裁剪时Y方向的偏移量
	IMAGEPROCESSOR processor;
	processor.Initialize(&img_shape); // 初始化图像处理器

	// 手部关键点模型初始化
	HLMD_MODEL model;
	model.Initialize(path_det, &crop_shape, &det_shape);
	HandLandMarksResult *det_result = new HandLandMarksResult;

	// OSD可视化器初始化
	VISUALIZER visualizer;
	visualizer.Initialize(img_shape, "shared_colorLUT.sscl");

	// 系统稳定等待
	cout << "sleep for 0.2 second!" << endl;
	sleep(0.2);

	// OSD 贴图
	visualizer.DrawBitmap(osds[0].filename, "shared_colorLUT.sscl", osds[0].x, osds[0].y, 2);

	uint16_t num_frames = 0; // 帧计数器
	uint8_t osd_index = 0; // osd 贴图 index
	ssne_tensor_t img_sensor; // 图像tensor定义

	// 创建键盘监听线程
	std::thread listener_thread(keyboard_listener);

	/******************************************************************************************
     * 3. 主处理循环
     ******************************************************************************************/
	while (!check_exit_flag()) {
		// 从sensor获取图像
		int ret = processor.GetImage(&img_sensor);
		if (ret != 0) {
			fprintf(stderr, "Failed to get image from sensor!\n");
			continue;
		}

		// ===================== 单通道灰度图直接使用 =====================
		// 直接使用灰度图像进行推理，不需要转换为RGB
		// 推理
		model.Predict(&img_sensor, det_result);

		// ===================== 手部关键点解析 =====================
		std::vector<std::array<float, 4>> single_hand_2d;

		if (!det_result->landmarks_local.empty()) {
			std::array<float, 63> hand_data = det_result->landmarks_local.at(0);
			const int POINT_OFFSET = 2;

			// 计算缩放因子：从模型输入尺寸(224x224)到裁剪图像尺寸(720x720)
			float scale_x = static_cast<float>(crop_shape[0]) / det_shape[0];
			float scale_y = static_cast<float>(crop_shape[1]) / det_shape[1];

			// 21个关键点
			for (int kp_idx = 0; kp_idx < 21; ++kp_idx) {
				float x_raw = hand_data[kp_idx * 3 + 0];
				float y_raw = hand_data[kp_idx * 3 + 1];

				// 从模型输入尺寸映射到裁剪图像尺寸
				float x_crop = x_raw * scale_x;
				float y_crop = y_raw * scale_y;

				// 从裁剪图像尺寸映射到原始图像尺寸
				float x_orig = x_crop;
				float y_orig = y_crop + crop_offset_y;

				float x1 = x_orig - POINT_OFFSET;
				float y1 = y_orig - POINT_OFFSET;
				float x2 = x_orig + POINT_OFFSET;
				float y2 = y_orig + POINT_OFFSET;

				single_hand_2d.push_back({x1, y1, x2, y2});
			}
		}

		// 绘制关键点
		visualizer.Draw(single_hand_2d);

		// 帧计数 + OSD切换
		num_frames++;
		osd_index = (num_frames / 10) % 3;
		visualizer.DrawBitmap(osds[osd_index].filename, "shared_colorLUT.sscl", osds[osd_index].x, osds[osd_index].y, 2);
	}

	/******************************************************************************************
     * 4. 资源释放
     ******************************************************************************************/
	// 等待线程退出
	if (listener_thread.joinable()) {
		listener_thread.join();
	}

	// 释放资源
	delete det_result;
	model.Release();
	processor.Release();
	visualizer.Release();

	if (ssne_release()) {
		fprintf(stderr, "SSNE release failed!\n");
		return -1;
	}

	return 0;
}