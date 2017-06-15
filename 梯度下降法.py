# 本节主要讲解使用层级构造函数
# 本节主要讲解使用层级构造函数
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#构造层级函数
def add_layer(inputs,in_size,out_size,activation_function=None):
    Weights=tf.Variable(tf.random_normal([in_size,out_size]))
    biases = tf.Variable(tf.zeros([1,out_size])+0.1)
    Wx_plus_b = tf.matmul(inputs,Weights)+biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b) #使用激励函数，（非线性函数）
    return outputs
# 输入值
x_data = np.linspace(-1,1,300,dtype='float32')[:,np.newaxis]
# 偏差值
noise = np.random.normal(0,0.05,x_data.shape)
# 校验值
y_data= np.square(x_data)-0.5 + noise

# placeholder使用传参传入输入值
xs = tf.placeholder(tf.float32,[None,1])
# placeholder使用传参传入输出值
ys = tf.placeholder(tf.float32,[None,1])


# 输入层 参数10是隐藏层
l1 = add_layer(xs,1,10,activation_function=tf.nn.relu)
# 输出层 参数10是隐藏层
predition = add_layer(l1,10,1,activation_function=None)

# 校验偏差值
loss = tf.reduce_min(tf.reduce_sum(tf.square(ys - predition),reduction_indices=[1]))
# 最重要的一步，梯度下降法学习（以loss的偏差值以0.1为标准学习）
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# 初始化变量
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

#数据可视化
#定义一张图表
fig = plt.figure()
ax = fig.add_subplot(1,1,1) #
ax.scatter(x_data,y_data)
plt.ion()
plt.show()

for i in range(1000):
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    if i % 50 ==0:
        #print (sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        predition_value = sess.run(predition,feed_dict={xs:x_data})
        lines = ax.plot(x_data,predition_value,'r-',lw=5)
        plt.pause(0.1)
