import numpy as np
from data_utils import load_cifar10
import matplotlib.pyplot as plt
import time
from  knn import KNearestNeighbor

#3.1 ) 将数据载入模型
x_train,y_train,x_test,y_test=load_cifar10('cifar-10-batches-py') #这里调用数据处理的函数data_utils.py

print('training data shape:',x_train.shape)
print('training labels shape:',y_train.shape)
print('test data shape:',x_test.shape)
print('test labels shape:',y_test.shape)


# 这50000张训练集每一类中随机挑选samples_per_class张图片
classes=['plane','car','bird','cat','deer','dog','frog','horse','ship','truck']
num_claesses=len(classes)
samples_per_class=10
for y ,cls in enumerate(classes):
    """
    flatnonzero(): 该函数输入一个矩阵，返回扁平化后矩阵中非零元素的位置（index）
    >>> x = np.arange(-2, 3)  得到 array([-2, -1, 0, 1, 2])
    >>> np.flatnonzero(x)    得到array([0, 1, 3, 4]) 这几个序列位置为非零
    """
    print("----------------我是分割线初始----------------")
    print("y的值: ",y)
    print("y_train的值: ",y_train)
    print("cls的值: ",cls)
    print("y_train == y 的值: ",y_train == y)
    idxs=np.flatnonzero(y_train==y)
    print("idxs初始值：",idxs)
    idxs=np.random.choice(idxs,samples_per_class,replace=False) #随机选取7个数值
    print("idxs随机挑选后的值：", idxs)
    print("----------------我是分割线结束----------------")
    '''
    numpy.random.choice(a, size=None, replace=True, p=None)
    Generates a random sample from a given 1-D array
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.choice.html
    '''
    for i ,idx in enumerate(idxs):
        plt_idx=i*num_claesses+y+1
        plt.subplot(samples_per_class,num_claesses,plt_idx) #subplot()是将整个figure均等分割
        plt.imshow(x_train[idx].astype('uint8'))
        plt.axis('off')
        if i ==0:
            plt.title(cls)
# plt.show()
# plt.close()


#选取5000张训练集， 500张测试集，
num_training = 5000
mask = range(num_claesses)
x_train = x_train[mask]
y_train = y_train[mask]
num_test = 500
mask = range(num_test)
x_test = x_test[mask]
y_test = y_test[mask]

x_train = np.reshape(x_train, (x_train.shape[0], -1)) #把图像数据拉长为行向量
x_test = np.reshape(x_test, (x_test.shape[0], -1))
print("x_train的shape：", x_train.shape)
print("x_test的shape：", x_test.shape)



# 3.2) 测试集预测
classifier = KNearestNeighbor()
classifier.train(x_train, y_train)
dists=classifier.compute_distances_no_loops(x_test)
print(dists)

