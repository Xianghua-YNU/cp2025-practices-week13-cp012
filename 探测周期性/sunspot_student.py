import numpy as np
import matplotlib.pyplot as plt


def load_sunspot_data(url):
    """
    从本地文件读取太阳黑子数据

    参数:
        url (str): 本地文件路径

    返回:
        tuple: (years, sunspots) 年份和太阳黑子数
    """
    # 只读取第2列(年份)和第3列(太阳黑子数)，忽略其他列
    data = np.loadtxt(url, usecols=(1, 2))  # 列索引从0开始
    years = data[:, 0]
    sunspots = data[:, 1]
    return years, sunspots


def plot_sunspot_data(years, sunspots):
    """
    绘制太阳黑子数据随时间变化图

    参数:
        years (numpy.ndarray): 年份数组
        sunspots (numpy.ndarray): 太阳黑子数数组
    """
    plt.figure(figsize=(12, 6))
    plt.plot(years, sunspots)
    plt.title('太阳黑子数随时间变化')
    plt.xlabel('年份')
    plt.ylabel('太阳黑子数')
    plt.grid(True)
    plt.show()


def compute_power_spectrum(sunspots):
    """
    计算太阳黑子数据的功率谱

    参数:
        sunspots (numpy.ndarray): 太阳黑子数数组

    返回:
        tuple: (frequencies, power) 频率数组和功率谱
    """
    n = len(sunspots)
    fft_result = np.fft.rfft(sunspots)  # 使用rfft只计算正频率部分
    power = np.abs(fft_result) ** 2 / n ** 2  # 正确归一化
    frequencies = np.fft.rfftfreq(n, d=1.0)  # 采样间隔为1个月
    return frequencies, power  # 直接返回正频率部分


def plot_power_spectrum(frequencies, power):
    """
    绘制功率谱图

    参数:
        frequencies (numpy.ndarray): 频率数组
        power (numpy.ndarray): 功率谱数组
    """
    plt.figure(figsize=(12, 6))
    plt.plot(frequencies, power)
    plt.title('太阳黑子数据功率谱')
    plt.xlabel('频率 (1/月)')
    plt.ylabel('功率')
    plt.grid(True)
    plt.xlim(0, 0.05)  # 只显示低频部分，便于观察
    plt.show()


def find_main_period(frequencies, power):
    """
    找出功率谱中的主周期

    参数:
        frequencies (numpy.ndarray): 频率数组
        power (numpy.ndarray): 功率谱数组

    返回:
        float: 主周期（月）
    """
    # 排除频率为0的直流分量
    valid_indices = np.where(frequencies > 0.0001)  # 更小的阈值
    valid_freq = frequencies[valid_indices]
    valid_power = power[valid_indices]

    # 限制频率范围，聚焦在可能的太阳黑子周期(约8-15年)
    min_freq = 1 / (12 * 12)  # 15年对应的频率
    max_freq = 1 / (10 * 12)  # 8年对应的频率
    periodic_indices = np.where((valid_freq >= min_freq) & (valid_freq <= max_freq))

    # 如果找到符合条件的频率，在其中找最大功率对应的频率
    if len(periodic_indices[0]) > 0:
        filtered_freq = valid_freq[periodic_indices]
        filtered_power = valid_power[periodic_indices]
        max_power_index = np.argmax(filtered_power)
        main_freq = filtered_freq[max_power_index]
    else:
        # 如果没有找到符合条件的频率，使用所有有效频率
        max_power_index = np.argmax(valid_power)
        main_freq = valid_freq[max_power_index]

    # 计算周期（月）
    main_period = 1 / main_freq
    return main_period


def main():
    # 数据文件路径
    data = "sunspot_data.txt"

    # 1. 加载并可视化数据
    years, sunspots = load_sunspot_data(data)
    plot_sunspot_data(years, sunspots)

    # 2. 傅里叶变换分析
    frequencies, power = compute_power_spectrum(sunspots)
    plot_power_spectrum(frequencies, power)

    # 3. 确定主周期
    main_period = find_main_period(frequencies, power)
    print(f"\nMain period of sunspot cycle: {main_period:.2f} months")
    print(f"Approximately {main_period / 12:.2f} years")


if __name__ == "__main__":
    main()
