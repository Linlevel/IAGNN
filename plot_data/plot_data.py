import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict as ddict
file_path='D:/PycharmProjects\HRAN-main\HRAN/results_weight1/59.txt'

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

# 转换为DataFrame
df = pd.DataFrame(data)

overlap_idxs = dict()
for feature, best in zip(["MR", "MRR", "Hits1", "Hits3", "Hits10"], ["MR_BEST", "MRR_BEST", "Hits1_BEST", "Hits3_BEST", "Hits10_BEST"]):
    overlap_idxs[best]=df[df[feature] == df[best]].index

# 归一化处理
scaler = MinMaxScaler()
features = ["Loss", "MR", "MRR", "Hits1", "Hits3", "Hits10","MR_BEST", "MRR_BEST", "Hits1_BEST", "Hits3_BEST", "Hits10_BEST"]
df[features] = scaler.fit_transform(df[features])

# 绘图
plt.figure(figsize=(12, 8))
for feature in ["MRR", "Hits1", "Hits3", "Hits10"]:
    plt.scatter(df['Iteration'][df[feature].idxmax()], df[feature].max(), color='black')
plt.scatter(df['Iteration'][df['MR'].idxmin()], df['MR'].min(), color='black')

# 寻找重叠点
for feature, best, color in zip(["MR", "MRR", "Hits1", "Hits3", "Hits10"], ["MR_BEST", "MRR_BEST", "Hits1_BEST", "Hits3_BEST", "Hits10_BEST"], ['green', 'red', 'orange', 'brown', 'purple']):
    plt.scatter(df.loc[overlap_idxs[best], 'Iteration'], df.loc[overlap_idxs[best], feature], color=color, s=5, label=best)

for feature in ["MRR", "Hits1", "Hits3", "Hits10"]:
    plt.plot([df['Iteration'][df[feature].idxmax()],df['Iteration'][df[feature].idxmax()]],[0, df[feature].max()], color='red', linewidth=2)
plt.plot([df['Iteration'][df['MR'].idxmin()],df['Iteration'][df['MR'].idxmin()]],[1, df['MR'].min()], color='red', linewidth=2)
for feature, color in zip(["Loss", "MR", "MRR", "Hits1", "Hits3", "Hits10"], ['blue', 'green', 'red', 'orange', 'brown', 'purple']):
    plt.plot('Iteration', feature, data=df, marker='', color=color, linewidth=0.5, label=feature)


plt.legend()
plt.title('Normalized Metrics Over Iterations')
plt.xlabel('Iteration')
plt.ylabel('Normalized Value')
plt.show()