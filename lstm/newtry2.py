import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from movingaverage import *
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

#定义常量
rnn_unit=128       #hidden layer units
input_size=1
output_size=1
lr=0.0006         #学习率
#——————————————————导入数据——————————————————————
for i in range(1, 22):
    if i < 41:
        if i < 10:
            locals()['a' + str(i)] = np.loadtxt(r'C:\Users\29860\Documents\1806\data_aligned\C30' + str(i) + '.txt',
                                                skiprows=0)
        elif i < 17:
            locals()['a' + str(i)] = np.loadtxt(r'C:\Users\29860\Documents\1806\data_aligned\C3' + str(i) + '.txt',
                                                skiprows=0)
        elif i < 26:
            locals()['a' + str(i)] = np.loadtxt(
                r'C:\Users\29860\Documents\1806\data_aligned\C50' + str(i - 16) + '.txt', skiprows=0)
        else:
            locals()['a' + str(i)] = np.loadtxt(r'C:\Users\29860\Documents\1806\data_aligned\C5' + str(i - 16) + '.txt',
                                                skiprows=0)
# X_train = np.empty((42500*39,1),dtype="float32")
# y_train = np.empty((42500*39,1),dtype="float32")
# X_test = np.empty((42500,1),dtype="float32")
# y_test = np.empty((42500,1),dtype="float32")
# for i in range(1, 41):
#     for j in range(42500):
#         if i == 40:
#             X_test[j, :] = a40[0 + 4 * j:300 + 4 * j:4, 1]
#             y_test[j] = a40[300 + 4 * j, 2]
#         else:
#             X_train[j, :] = locals()['a' + str(i + 1)][0 + 4 * j:300 + 4 * j:4, 1]
#             y_train[j, :] = locals()['a' + str(i + 1)][300 + 4 * j, 2]
X_train = a1[:170000:8, 1, np.newaxis]
y_train = a1[:170000:8, 2, np.newaxis]
data = np.concatenate((X_train, y_train), axis=1)
for i in range(2, 22):
    data = np.concatenate((data, np.concatenate((locals()['a'+str(i)][:170000:8, 1, np.newaxis], locals()['a'+str(i)][:170000:8, 2, np.newaxis]), axis=1)), axis=0)
print('data', data.shape)

#获取训练集
def get_train_data(batch_size=64,time_step=150,train_begin=0,train_end=0):
    batch_index=[]
    data_train=data[train_begin:train_end]
    print('data_train', data_train.shape)
    normalized_train_data=(data_train-np.mean(data_train,axis=0))/np.std(data_train,axis=0)  #标准化
    train_x,train_y=[],[]   #训练集
    for i in range(len(normalized_train_data)-time_step):
       if i % batch_size==0:
           batch_index.append(i)
       x=normalized_train_data[i:i+time_step,0,np.newaxis]
       y=normalized_train_data[i:i+time_step,1,np.newaxis]
       train_x.append(x.tolist())
       train_y.append(y.tolist())
    batch_index.append((len(normalized_train_data)-time_step))
    print('X_train1', np.shape(train_x))
    print('y_train1', np.shape(train_y))
    return batch_index,train_x,train_y



#获取测试集
def get_test_data(time_step=150,test_begin=0):
    data_test=data[test_begin:]
    mean=np.mean(data_test,axis=0)
    std=np.std(data_test,axis=0)
    normalized_test_data=(data_test-mean)/std  #标准化
    size=(len(normalized_test_data)+time_step-1)//time_step  #有size个sample
    test_x,test_y=[],[]
    for i in range(size-1):
       x=normalized_test_data[i*time_step:(i+1)*time_step,0,np.newaxis]
       y=normalized_test_data[i*time_step:(i+1)*time_step,1,np.newaxis]
       test_x.append(x.tolist())
       test_y.extend(y)
    test_x.append((normalized_test_data[(i+1)*time_step:,0]).tolist())
    test_y.extend((normalized_test_data[(i+1)*time_step:,1]).tolist())
    print('X_test', np.shape(test_x))
    return mean,std,test_x,test_y



#——————————————————定义神经网络变量——————————————————
#输入层、输出层权重、偏置

weights={
         'in':tf.Variable(tf.random_normal([input_size,rnn_unit])),
         'out':tf.Variable(tf.random_normal([rnn_unit,1]))
        }
biases={
        'in':tf.Variable(tf.constant(0.1,shape=[rnn_unit,])),
        'out':tf.Variable(tf.constant(0.1,shape=[1,]))
       }

#——————————————————定义神经网络变量——————————————————
def lstm(X):
    batch_size=tf.shape(X)[0]
    time_step=tf.shape(X)[1]
    w_in=weights['in']
    b_in=biases['in']
    input=tf.reshape(X,[-1,input_size])  #需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
    input_rnn=tf.matmul(input,w_in)+b_in
    input_rnn=tf.reshape(input_rnn,[-1,time_step,rnn_unit])  #将tensor转成3维，作为lstm cell的输入
    cell=tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
    init_state=cell.zero_state(batch_size,dtype=tf.float32)
    output_rnn,final_states=tf.nn.dynamic_rnn(cell, input_rnn,initial_state=init_state, dtype=tf.float32)  #output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果
    output=tf.reshape(output_rnn,[-1,rnn_unit]) #作为输出层的输入
    w_out=weights['out']
    b_out=biases['out']
    pred=tf.matmul(output,w_out)+b_out
    return pred,final_states



#——————————————————训练模型——————————————————
def train_lstm(batch_size=80,time_step=150,train_begin=0,train_end=42500*20):
    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
    Y=tf.placeholder(tf.float32, shape=[None,time_step,output_size])
    batch_index,train_x,train_y=get_train_data(batch_size,time_step,train_begin,train_end)
    pred,_=lstm(X)
    #损失函数
    loss=tf.reduce_mean(tf.square(tf.reshape(pred,[-1])-tf.reshape(Y, [-1])))
    train_op=tf.train.AdamOptimizer(lr).minimize(loss)
    saver=tf.train.Saver(tf.global_variables(),max_to_keep=3)
    # module_file = tf.train.latest_checkpoint()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # saver.restore(sess, module_file)
        #重复训练10000次
        for i in range(500):
            for step in range(len(batch_index)-1):
                _,loss_=sess.run([train_op,loss],feed_dict={X:train_x[batch_index[step]:batch_index[step+1]],Y:train_y[batch_index[step]:batch_index[step+1]]})
            print(i,loss_)
            if i % 25==0:
                print("保存模型：",saver.save(sess,'ckpt/newtry2_sway.ckpt',global_step=i))

# train_lstm()

#————————————————预测模型————————————————————
def prediction(time_step=150, test_begin=42500*0):
    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
    #Y=tf.placeholder(tf.float32, shape=[None,time_step,output_size])
    mean,std,test_x,test_y=get_test_data(time_step, test_begin)
    print('test_x', np.shape(test_x))
    pred,_=lstm(X)
    # ckpt = tf.train.get_checkpoint_state('ckpt/newtry2_2000.ckpt-2000')
    # saver = tf.train.import_meta_graph('ckpt/newtry2_2000.ckpt-2000' + '.meta')
    # saver=tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        #参数恢复
        # module_file = tf.train.latest_checkpoint('ckpt/')
        # module_file = tf.train.latest_checkpoint('ckpt/newtry2_2000.ckpt-2000')
        # print(ckpt.model_checkpoint_path)
        # saver.restore(sess, 'ckpt/newtry2_2000.ckpt-2000')
        new_saver = tf.train.import_meta_graph('ckpt/newtry2_2000.ckpt-2000.meta')
        new_saver.restore(sess, 'ckpt/newtry2_2000.ckpt-2000')
        test_predict=[]
        for step in range(len(test_x)-1):
          prob=sess.run(pred,feed_dict={X:[test_x[step]]})
          predict=prob.reshape((-1))
          test_predict.extend(predict)
        test_y=np.array(test_y)*std[1]+mean[1]
        test_predict=np.array(test_predict)*std[1]+mean[1]
        acc=np.average(np.abs(test_predict-test_y[:len(test_predict)])/test_y[:len(test_predict)])  #偏差
        print('acc', acc)
        #以折线图表示结果
        ave_step = 1
        test_pred = movingaverage(test_predict, ave_step, avoid_fp_drift = True)
        plt.figure()
        plt.plot(list(range(int((ave_step+1)/2)-1, len(test_predict)-int((ave_step+1)/2)+1)), list(test_pred), color='b')
        plt.plot(list(range(len(test_y))), test_y,  color='r')
        plt.show()

prediction()