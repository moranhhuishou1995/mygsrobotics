import numpy as np

# 创建一个7x9x3的随机矩阵
matrix = np.random.rand(5, 5, 3)
print(matrix)

# 对每个元素的第二个元素进行逐元素求和
sum_second_element_elementwise = np.sum(matrix[:, :, 1], axis=0)

print("对每个元素的第二个元素进行逐元素求和的结果:")
print(sum_second_element_elementwise)
