# 这节主要学习placeholderh函数
# plcaeholder是一个外部传值的变量

import tensorflow as tf

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

output = tf.mul(input1,input2)

with tf.Session() as sess:
    print sess.run(output,feed_dict={input1:[5],input2:[3]})
