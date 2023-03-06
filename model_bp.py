import copy

import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

import xlrd2

# 打开表格文件
file_path = 'dataSet/data.xlsx'
data = xlrd2.open_workbook(file_path)
table = data.sheet_by_name('Sheet1')

# 获取总行数
nrows = table.nrows
# 获取总列数
ncols = table.ncols

# 获取一行的全部数值，例如第5行
row_value = table.row_values(5)

# 用于对非数值的特征向量化
x_value_sex = {"female": [1, 0], "male": [0, 1]}
x_value_smoker = {"yes": [1, 0], "no": [0, 1]}
x_value_region = {"southwest": [1, 0, 0, 0], "northwest": [0, 1, 0, 0], "southeast": [0, 0, 1, 0],
                  "northeast": [0, 0, 0, 1]}

# 记录每个数值特征最大值，用于归一化
age_max = 0.0
bim_max = 0.0
children_max = 0.0
charges_max = 0.0

# 网络输入维度：[年龄1， 性别2， 暴模指数1， 孩子数1，是否吸烟2， 房子朝向4}
vector_x = 1 + 2 + 1 + 1 + 2 + 4
x_train_raw = np.empty([nrows, vector_x], dtype=float)

# 输出为保险金额，一维
y_train_raw = np.empty([nrows, 1], dtype=float)

# 读入数据集
for i in range(1, nrows):
    row_value = table.row_values(i)

    age = row_value[0]
    sex = row_value[1]
    bim = row_value[2]
    children = row_value[3]
    smoker = row_value[4]
    region = row_value[5]
    charges = row_value[6]

    if age > age_max:
        age_max = age

    if bim > bim_max:
        bim_max = bim

    if children > children_max:
        children_max = children

    if charges > charges_max:
        charges_max = charges

    index = 0
    x_train_raw[i - 1][index] = age
    index = index + 1

    sex_f = x_value_sex[sex]
    for j in range(0, len(sex_f)):
        x_train_raw[i - 1][index + j] = sex_f[j]

    index = index + len(sex_f)

    x_train_raw[i - 1][index] = bim
    index = index + 1

    x_train_raw[i - 1][index] = children
    index = index + 1

    smoker_f = x_value_smoker[smoker]
    for j in range(0, len(smoker_f)):
        x_train_raw[i - 1][index + j] = smoker_f[j]

    index = index + len(smoker_f)

    region_f = x_value_region[region]
    for j in range(0, len(region_f)):
        x_train_raw[i - 1][index + j] = region_f[j]

    index = index + len(region_f)

    y_train_raw[i - 1][0] = charges

# 除以最大值，把数值映射的到0-1之间，简单归一化
for i in range(0, len(x_train_raw)):
    x_train_raw[i][0] /= age_max
    x_train_raw[i][3] /= bim_max
    x_train_raw[i][4] /= children_max
    y_train_raw[i][0] /= charges_max


# 定义神经网络模型
class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.layer1 = tf.keras.layers.Dense(vector_x, activation='tanh')
        self.layer2 = tf.keras.layers.Dense(8, activation='tanh')
        self.layer3 = tf.keras.layers.Dense(4, activation='tanh')
        self.layer4 = tf.keras.layers.Dense(2, activation='tanh')
        self.layer5 = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        out_layer = self.layer1(inputs)
        out_layer = self.layer2(out_layer)
        out_layer = self.layer3(out_layer)
        out_layer = self.layer4(out_layer)
        out_layer = self.layer5(out_layer)
        return out_layer


# 创建模型
#model = MyModel()
model = tf.keras.models.load_model("mode/m1")  ##载入训练好的模型
# 设置模型训练参数（使用Adam梯度优化）
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# model.fit(x_train_raw, y_train_raw, validation_split=0.1, batch_size=64, epochs=100)


# 绘图
def show(x_train, y_train, title):
    x = range(0, len(x_train))
    y = [1] * len(x_train)

    y_predict_raw = model(x_train)
    y_predict = [1] * len(x_train)

    for i in range(0, len(y)):
        y[i] = y_train[i][0] * charges_max
        y_predict[i] = float(y_predict_raw[i][0]) * charges_max

    # 设置x、y坐标轴的范围
    plt.title(title)
    plt.xlim(0, 100)
    plt.ylim(0, charges_max)
    # 绘制图形，红色是实际值，蓝色是预测值
    plt.plot(x, y, c='b')
    plt.plot(x, y_predict, c='r')
    plt.show()


show(x_train_raw, y_train_raw, "truly and predict")

# 去掉特征age进行预测测试
x_train_no_age = copy.deepcopy(x_train_raw)
for i in range(0, len(x_train_raw)):
    x_train_no_age[i][0] = 0.0

show(x_train_no_age, y_train_raw, "no age")

# 去掉特征sex进行预测测试
x_train_no_sex = copy.deepcopy(x_train_raw)
for i in range(0, len(x_train_raw)):
    x_train_no_sex[i][1] = 0.0
    x_train_no_sex[i][2] = 0.0

show(x_train_no_sex, y_train_raw, "no sex")

# 去掉特征bim进行预测测试
x_train_no_bim = copy.deepcopy(x_train_raw)
for i in range(0, len(x_train_raw)):
    x_train_no_bim[i][3] = 0.0

show(x_train_no_bim, y_train_raw, "no bim")

# 去掉特征children进行预测测试
x_train_no_children = copy.deepcopy(x_train_raw)
for i in range(0, len(x_train_raw)):
    x_train_no_children[i][4] = 0.0

show(x_train_no_children, y_train_raw, "no children")

# 去掉特征smoker进行预测测试
x_train_no_smoker = copy.deepcopy(x_train_raw)
for i in range(0, len(x_train_raw)):
    x_train_no_smoker[i][5] = 0.0
    x_train_no_smoker[i][6] = 0.0

show(x_train_no_smoker, y_train_raw, "no smoker")

# 去掉特征region进行预测测试
x_train_no_region = copy.deepcopy(x_train_raw)
for i in range(0, len(x_train_raw)):
    x_train_no_region[i][7] = 0.0
    x_train_no_region[i][8] = 0.0
    x_train_no_region[i][9] = 0.0
    x_train_no_region[i][10] = 0.0

show(x_train_no_region, y_train_raw, "no region")
