import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

c = tf.constant('Hello, World!')
with tf.Session() as sess:
    print(sess.run(c))

hello = tf.constant("Hello, TensorFlow")
a = tf.constant(10)
b = tf.constant(20)

sess = tf.Session()
print(sess.run(hello))
print(sess.run(a + b))

# 初始化一个2*3*1的神经网络的，随机初始化权重
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))
x = tf.constant([[0.7, 0.9]])
# 执行计算，没有权重bias
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

with tf.Session() as sess:
    # 必须要先执行初始化
    sess.run(w1.initializer)
    sess.run(w2.initializer)
    # print(sess.run(a))
    print(sess.run(y))
