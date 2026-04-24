# IAGNN
一种交互注意力图神经网络（Interactive Attention Graph Neural Network，IAGNN）的知识图谱嵌入模型。该模型受到图注意力网络（Graph Attention Network，GAT）的更新策略和 HRAN 的自注意值的启发，提出了一种基于实体与关系的依存关 系进行运算的交互注意力机制，用于更新实体嵌入和关系嵌入。此外，通过利用 这种注意力机制，可进一步形成关系注意力机制，得到关系路径权重，选择性地聚合关键邻居实体的特征，从而显著提升链接预测的效果和准确度。
