# 吴恩达在2011年录制的视频里面使用的是octave这个语言，我并不打算用这个，那么就使用我想用的python + numpy就可以了，那么现在学一下numpy的基础
# 教程 https://www.yiibai.com/numpy/numpy_ndarray_object.html
# 通常矩阵的变量都是大写，我这个涂方便换了小写

import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt

'''
# 一维数据
a = np.array([1,2,3])

# 二维数据
b = np.array([[1,2], [3,4]])

# 最小维度
c = np.array([1, 2, 3, 4, 5], ndmin = 2)

# dtype 参数
d = np.array([1,  2,  3], dtype = complex)

'''


# # dtype
# dt = np.dtype([('age',np.int8)])
# a = np.array([(10,),(20,),(30,)], dtype = dt)

# # reshape
# b = np.array([[1,2,3],[4,5,6]])
# c = b.reshape(3,2)


# # 等间隔数字的数组
# d = np.arange(24)
# e = d.reshape(2,4,3)

# a = np.zeros(5, dtype = float)
# # [ 0.  0.  0.  0.  0.]

# b = np.ones(5) # dtype=float

# # 下面两个都是矩阵
# c = np.matlib.ones((2,2))
# d = np.matlib.ones((3,2))
# # print(d)


# i = np.matrix('1,2;3,4')
# e = np.matrix('3,4;5,7')
# # print(i * e)

# v = np.matrix('1;2;4')
# # print(np.log(v))

# h = [1,2,3]
# # print(np.mat(h))

# m = np.mat([[2,5,1],[4,6,2]])
# # print(m < 3)

t = np.arange(0,1,0.01)
y = np.sin(2*np.pi*4*t)
y2 = np.cos(2*np.pi*4*t)
plt.plot(t, y)
# plt.ylabel('some numbers')
# plt.ion()
plt.plot(t, y2, 'r--')
# plt.pause(1)
plt.show()