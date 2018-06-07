
import pickle
"""
pickle提供了一个简单的持久化功能。可以将对象以文件的形式存放在磁盘上。
pickle.dump(obj, file[, protocol]):序列化对象，并将结果数据流写入到文件对象中。参数protocol是序列化模式，默认值为0，表示以文本的形式序列化。protocol的值还可以是1或2，表示以二进制的形式序列化。
pickle.load(file):反序列化对象。将文件中的数据解析为一个Python对象。
二进制文件就用二进制方法读取'rb'
"""
import numpy as np
import os

#导入单个数据模块
def load_cifar_data(filename):
    with open(filename, 'rb') as f:
        datadict =  pickle.load(f, encoding='bytes')
        x = datadict[b'data']
        y = datadict[b'labels']
        x = x.reshape(10000,3,32,32).transpose(0,2,3,1).astype('float')
        y = np.array(y)
        print("********")
        print(x.shape)
        print(y.shape)
        print("********")
        return x, y

#循环导入多个数据模块，
def load_cifar10(root):
    xs = [] #定义数组用来储存
    ys = []
    for b in range(1, 6):
        f = os.path.join(root, 'data_batch_%d' % b )
        x, y = load_cifar_data(f)
        xs.append(x) #添加数组，append
        ys.append(y)
    Xtrain = np.concatenate(xs) #把所有的数组都组合起来一个大的数组
    Ytrain = np.concatenate(ys)

    print("##第一次看Xtrain, Ytrain的shape###")
    print(Xtrain.shape)
    print(Ytrain.shape)
    print("##第一次看Xtrain, Ytrain的shape###")

    print("##正在处理Xtest，Ytest，变为标准格式。。。###")

    del x, y
    Xtest, Ytest = load_cifar_data(os.path.join(root, 'test_batch'))
    # Xtest, Ytest = load_cifar_batch(os.path.join(root, 'test_batch'))
    print("##处理结束。。。。。。。。。###")

    print("##第二次看Xtrain, Ytrain， Xtest，Ytest的shape###")
    print(Xtrain.shape)
    print(Ytrain.shape)
    print(Xtest.shape)
    print(Ytest.shape)
    print("##第二次看Xtrain, Ytrain， Xtest，Ytest的shape###")
    return Xtrain, Ytrain, Xtest, Ytest


x_train,y_train,x_test,y_test = load_cifar10('cifar-10-batches-py')
print("--------------------")
print("y_train:",y_train)
print("y_test:",y_test)
# print(y_train == y)
print("--------------------")


classes=['plane','car','bird','cat','deer','dog','frog','horse','ship','truck']
num_claesses=len(classes)
samples_per_class=7
for y ,cls in enumerate(classes):
    idxs=np.flatnonzero(y_train==y) #第一次循环，所有为plane
    print(idxs)
    print("i am here")
