import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
file_path='D:/PycharmProjects\HRAN-main\HRAN/results_weight1\HRAN_kinship_05-19-05-23-57.txt'
file_path1='D:/PycharmProjects\HRAN-main\HRAN/results_weight1/59.txt'
def predata(file_path):
    # 初始化数据存储结构
    data = {
        "Iteration": [],
        "Loss": [],
        "MR": [],
        "MRR": [],
        "Hits1": [],
        "Hits3": [],
        "Hits10": [],
        "MR_BEST": [],
        "MRR_BEST": [],
        "Hits1_BEST": [],
        "Hits3_BEST": [],
        "Hits10_BEST": []
    }

    # 读取并解析文件
    iteration_tracker = []
    with open(file_path, 'r') as file:
        for line in file:
            if "Iteration:" in line and "epoch loss:" in line:
                parts = line.split()
                iteration = int(parts[0].split(':')[1])
                loss = float(parts[3])
                data["Iteration"].append(iteration)
                data["Loss"].append(loss)
                iteration_tracker.append(iteration)  # 追踪迭代次数
            elif len(line.split()) == 11 and line[0].isdigit():
                metrics = list(map(float, line.split()))
                for i in range(3):
                    data["MR"].append(metrics[1])
                    data["MRR"].append(metrics[2])
                    data["Hits1"].append(metrics[3])
                    data["Hits3"].append(metrics[4])
                    data["Hits10"].append(metrics[5])
                    data["MR_BEST"].append(metrics[6])
                    data["MRR_BEST"].append(metrics[7])
                    data["Hits1_BEST"].append(metrics[8])
                    data["Hits3_BEST"].append(metrics[9])
                    data["Hits10_BEST"].append(metrics[10])
    df = pd.DataFrame(data)
    return df
# 转换为DataFrame
df=predata(file_path)
df1=predata(file_path1)
# Sample data generation
epochs = np.linspace(0, 1500, 1500)

# Creating the plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Hits@1 plot
ax1.plot(epochs, df['Hits1'], label='HRAN', color='gold')
ax1.plot(epochs, df1['Hits1'], label='IAGNN', color='darkblue')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Hits@1')
# ax1.set_ylim([0.40, 0.60])
ax1.legend()
ax1.set_title('Hits@1 over Epochs')
ax1.grid(True)

# Hits@1 inset
inset1 = inset_axes(ax1, width="55%", height="30%", loc=5)
inset1.plot(epochs, df['Hits1'], label='HRAN', color='gold')
inset1.plot(epochs, df1['Hits1'], label='IAGNN', color='darkblue')
inset1.set_xlim(600, 1500)
inset1.set_ylim([0.7, 0.9])
# inset1.set_xticks(range(800, 1201, 100))
inset1.grid(True)

# MRR plot
ax2.plot(epochs, df['MRR'], label='HRAN', color='gold')
ax2.plot(epochs, df1['MRR'], label='IAGNN', color='darkblue')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('MRR')
# ax2.set_ylim([0.20, 0.40])
ax2.legend()
ax2.set_title('MRR over Epochs')
ax2.grid(True)

# MRR inset
inset2 = inset_axes(ax2, width="55%", height="30%", loc=5)
inset2.plot(epochs, df['MRR'], label='HRAN', color='gold')
inset2.plot(epochs, df1['MRR'], label='IAGNN', color='darkblue')
inset2.set_xlim(600, 1500)
inset2.set_ylim([0.8, 0.95])
# inset2.set_xticks(range(800, 1201, 100))
inset2.grid(True)

plt.tight_layout()
plt.show()
