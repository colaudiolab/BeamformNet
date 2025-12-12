import pandas as pd
import matplotlib.pyplot as plt

# 读取 Excel 文件
file_path = 'results.xlsx'
xls = pd.ExcelFile(file_path)
sheet_names = xls.sheet_names

plt.rcParams.update({
    'font.size': 14,               # 默认字体大小（影响 legend、tick labels 等）
    'axes.labelsize': 16,          # x 和 y 轴标签字体大小
    'axes.titlesize': 18,          # 图标题字体大小
    'legend.fontsize': 12,         # 图例字体大小
    'xtick.labelsize': 14,         # x 轴刻度数字字体大小
    'ytick.labelsize': 14,       # y 轴刻度数字字体大小
    'axes.prop_cycle': plt.cycler('color', plt.cm.tab20.colors)  # 添加颜色循环

})


# 遍历每个 sheet
for sheet in sheet_names:
    df = pd.read_excel(xls, sheet_name=sheet)

    # 假设第一列是 x，其余是 y 数据
    if df.shape[1] < 2:
        print(f"Sheet '{sheet}' has less than 2 columns. Skipping.")
        continue

    # 清洗数据：将无法转换为数字的值替换为 NaN
    df = df.apply(pd.to_numeric, errors='coerce')

    x = df.iloc[:, 0]
    y_columns = df.columns[1:]

    plt.figure(figsize=(10, 4))

    for col in y_columns:
        if sheet == 'E':
            if col == 'BeamformNet':
                plt.plot(x, df[col], label=col, marker='o', markersize=3)
            else:
                plt.plot(x, df[col], label=col)
        else:
            plt.plot(x, df[col], label=col, marker='o', markersize=3)

    if sheet != 'accuracy':
        plt.ylabel('RMSPE [rad]')
    else:
        plt.ylabel('Accuracy [%]')
    # plt.title(f'Sheet: {sheet}')

    if sheet == 'spst_robust' or sheet == 'pre' or sheet == 'm':
        plt.legend(ncol=3)
    else:
        plt.legend()
    plt.grid(True)

    # 设置 Y 轴为 log scale
    if sheet != 'accuracy':
        plt.yscale('log')

    # 根据 sheet 名称设置 X 轴
    if sheet == 'spst':
        plt.xscale('log')  # spst sheet 的 X 轴为 log scale
        plt.xlabel('number of snapshots T')
    elif sheet == 'm':
        plt.xticks(x)  # m sheet 显示具体的 x 列数值
        plt.xlabel('number of array elements M')
    elif sheet == 'E':
        plt.xticks(x)  # m sheet 显示具体的 x 列数值
        plt.xlabel('value of hyperparameter E')
    elif sheet == 'd':
        plt.xlabel('number of signals K')
    elif sheet == 'snr':
        plt.xlabel('SNR[dB]')
    elif sheet == 'd_robust':
        plt.xlabel('number of signals K')
    elif sheet == 'snr_robust':
        plt.xlabel('SNR[dB]')
    elif sheet == 'spst_robust':
        plt.xlabel('number of snapshots T')
    elif sheet == 'pre':
        plt.xlabel('source distance(Δθ)[rad]')
    elif sheet == 'p_err':
        plt.xlabel(r'value of $\rho_{\mathrm{err}}$')
    elif sheet == 'accuracy':
        plt.xlabel(r'number of signals K')
    plt.tight_layout()
    plt.savefig(f'{sheet}_plot.png')  # 可选：保存图像
    plt.show()
