#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
白炽灯温度优化

通过数值计算方法，基于普朗克辐射定律，研究白炽灯发光效率与灯丝温度的关系，
寻找使效率最大化的最优温度，并分析其可行性。
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.optimize import minimize_scalar

# 物理常数
H = 6.62607015e-34  # 普朗克常数 (J·s)
C = 299792458       # 光速 (m/s)
K_B = 1.380649e-23  # 玻尔兹曼常数 (J/K)

# 可见光波长范围 (m)
VISIBLE_LIGHT_MIN = 380e-9  # 380 nm
VISIBLE_LIGHT_MAX = 780e-9  # 780 nm


def planck_law(wavelength, temperature):
    """
    计算普朗克黑体辐射公式
    
    参数:
        wavelength (float or numpy.ndarray): 波长，单位为米
        temperature (float): 温度，单位为开尔文
    
    返回:
        float or numpy.ndarray: 给定波长和温度下的辐射强度 (W/(m²·m))
    """
    # 普朗克常数 * 光速 / (波长 * 玻尔兹曼常数 * 温度)
    x = H * C / (wavelength * K_B * temperature)
    intensity = (2 * H * C**2) / (wavelength**5 * (np.exp(x) - 1))
    return intensity


def calculate_visible_power_ratio(temperature):
    """
    计算给定温度下可见光功率与总辐射功率的比值
    
    参数:
        temperature (float): 温度，单位为开尔文
    
    返回:
        float: 可见光效率（可见光功率/总功率）
    """
    # 定义普朗克函数
    def planck_func(wavelength):
        return planck_law(wavelength, temperature)
    
    # 计算可见光区域的积分
    visible_power, _ = integrate.quad(planck_func, VISIBLE_LIGHT_MIN, VISIBLE_LIGHT_MAX)
    
    # 计算总辐射功率（全波长范围）
    total_power, _ = integrate.quad(planck_func, 1e-20, np.inf)  # 从接近0开始积分
    
    return visible_power / total_power


def plot_efficiency_vs_temperature(temp_range):
    """
    绘制效率-温度关系曲线
    
    参数:
        temp_range (numpy.ndarray): 温度范围，单位为开尔文
    
    返回:
        tuple: (matplotlib.figure.Figure, numpy.ndarray, numpy.ndarray) 图形对象、温度数组、效率数组
    """
    efficiencies = np.array([calculate_visible_power_ratio(T) for T in temp_range])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(temp_range, efficiencies, 'b-', linewidth=2)
    ax.set_xlabel('Temperature (K)', fontsize=12)
    ax.set_ylabel('Visible Light Efficiency', fontsize=12)
    ax.set_title('Incandescent Lamp Efficiency vs Temperature', fontsize=14)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    return fig, temp_range, efficiencies


def find_optimal_temperature():
    """
    寻找使白炽灯效率最大的最优温度
    
    返回:
        tuple: (float, float) 最优温度和对应的效率
    """
    # 定义目标函数（负效率，用于最小化算法）
    def objective(T):
        return -calculate_visible_power_ratio(T)
    
    # 使用黄金分割法寻找最优温度
    result = minimize_scalar(
        objective,
        bounds=(1000, 10000),
        method='bounded',
        options={'xatol': 1.0}  # 精度设置为1K
    )
    
    optimal_temp = result.x
    optimal_efficiency = -result.fun  # 转换回正效率值
    
    return optimal_temp, optimal_efficiency


def main():
    """
    主函数，计算并可视化最优温度
    """
    # 绘制效率-温度曲线 (1000K-10000K)
    temp_range = np.linspace(1000, 10000, 100)
    fig_efficiency, temps, effs = plot_efficiency_vs_temperature(temp_range)
    plt.savefig('efficiency_vs_temperature.png', dpi=300)
    plt.show()
    
    # 计算最优温度
    optimal_temp, optimal_efficiency = find_optimal_temperature()
    print(f"\n最优温度: {optimal_temp:.1f} K")
    print(f"最大效率: {optimal_efficiency:.4f} ({optimal_efficiency*100:.2f}%)")
    
    # 与实际白炽灯温度比较
    actual_temp = 2700
    actual_efficiency = calculate_visible_power_ratio(actual_temp)
    print(f"\n实际灯丝温度: {actual_temp} K")
    print(f"实际效率: {actual_efficiency:.4f} ({actual_efficiency*100:.2f}%)")
    print(f"效率差异: {(optimal_efficiency - actual_efficiency)*100:.2f}%")
    
    # 标记最优和实际温度点
    plt.figure(figsize=(10, 6))
    plt.plot(temps, effs, 'b-', label='Efficiency Curve')
    plt.plot(optimal_temp, optimal_efficiency, 'ro', markersize=8, 
             label=f'Optimal: {optimal_temp:.1f} K ({optimal_efficiency*100:.1f}%)')
    plt.plot(actual_temp, actual_efficiency, 'go', markersize=8, 
             label=f'Actual: {actual_temp} K ({actual_efficiency*100:.1f}%)')
    plt.xlabel('Temperature (K)')
    plt.ylabel('Visible Light Efficiency')
    plt.title('Incandescent Lamp Efficiency vs Temperature')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('optimal_temperature.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
