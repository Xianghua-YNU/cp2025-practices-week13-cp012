#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
傅立叶滤波和平滑 - 道琼斯工业平均指数分析

本模块实现了对道琼斯工业平均指数数据的傅立叶分析和滤波处理。
通过保留不同的频率成分，我们可以观察信号的变化。
"""

import numpy as np  # 用于数学计算和傅立叶变换
import matplotlib.pyplot as plt  # 用于绘制图表
from matplotlib import rcParams  # 用于配置 matplotlib

# 配置 matplotlib 以支持中文显示
rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为 SimHei
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def load_data(filename):
    """
    加载道琼斯工业平均指数数据

    参数:
        filename (str): 数据文件路径，例如 'dow.txt'

    返回:
        numpy.ndarray: 包含指数值的数组
    """
    try:
        data = np.loadtxt(filename)
        print(f"成功加载文件 {filename}，数据长度为 {len(data)}")
    except FileNotFoundError:
        print(f"错误：文件 {filename} 未找到，请确保文件在当前目录下！")
        return None
    except Exception as e:
        print(f"加载数据时发生错误: {e}")
        return None

    return data


def plot_data(data, title="道琼斯工业平均指数"):
    """
    绘制时间序列数据

    参数:
        data (numpy.ndarray): 输入数据数组
        title (str): 图表标题，默认为“道琼斯工业平均指数”

    返回:
        None
    """
    if data is None:
        print("错误：数据为空，无法绘制图表！")
        return

    plt.figure()
    plt.plot(data, label='原始数据', color='blue')
    plt.title(title)
    plt.xlabel('交易日')
    plt.ylabel('指数值')
    plt.legend()
    plt.grid(True)
    plt.show()


def fourier_filter(data, keep_fraction=0.1):
    """
    执行傅立叶变换并滤波

    参数:
        data (numpy.ndarray): 输入数据数组
        keep_fraction (float): 保留的傅立叶系数比例，例如 0.1 表示保留前10%

    返回:
        tuple: (滤波后的数据数组, 原始傅立叶系数数组)
    """
    if data is None:
        print("错误：数据为空，无法执行傅立叶滤波！")
        return None, None

    coeff = np.fft.rfft(data)
    cutoff = int(len(coeff) * keep_fraction)
    print(f"保留前 {keep_fraction * 100}% 的系数，cutoff = {cutoff}/{len(coeff)}")

    filtered_coeff = np.copy(coeff)
    filtered_coeff[cutoff:] = 0

    filtered_data = np.fft.irfft(filtered_coeff)

    return filtered_data, coeff


def plot_comparison(original, filtered, title="傅立叶滤波结果"):
    """
    绘制原始数据和滤波结果的比较

    参数:
        original (numpy.ndarray): 原始数据数组
        filtered (numpy.ndarray): 滤波后的数据数组
        title (str): 图表标题，默认为“傅立叶滤波结果”

    返回:
        None
    """
    if original is None or filtered is None:
        print("错误：数据为空，无法绘制比较图表！")
        return

    plt.figure()
    plt.plot(original, label='原始数据', color='blue')
    plt.plot(filtered, label='滤波数据', color='red')
    plt.title(title)
    plt.xlabel('交易日')
    plt.ylabel('指数值')
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    print("任务1：加载数据并绘制原始数据图")
    data = load_data('dow.txt')
    plot_data(data, "道琼斯工业平均指数 - 原始数据")

    print("\n任务2：傅立叶滤波，保留前10%的系数")
    filtered_10, coeff = fourier_filter(data, 0.1)
    plot_comparison(data, filtered_10, "傅立叶滤波（保留前10%系数）")

    print("\n任务3：傅立叶滤波，保留前2%的系数")
    filtered_2, _ = fourier_filter(data, 0.02)
    plot_comparison(data, filtered_2, "傅立叶滤波（保留前2%系数）")


if __name__ == "__main__":
    main()
