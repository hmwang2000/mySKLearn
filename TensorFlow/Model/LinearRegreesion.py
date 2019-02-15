import matplotlib.pyplot as plt;
import numpy as np;
import tensorflow as tf;
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class LinearRegression:
    def __init__(self, datalength=1000, w=2, b=0.5):
        self.length = datalength;
        self.w = w;
        self.b = b;

    def CreateDataSet(self):
        self.simx = np.linspace(0, 1.0, self.length).astype(np.float32);  # np.random.randn(self.length);
        self.simy = self.w * self.simx + self.b + np.random.randn(self.length) * 0.15;
        self.simx = np.reshape(self.simx, newshape=[self.length, 1]);
        self.simy = np.reshape(self.simy, newshape=[self.length, 1]);

    def Train(self):
        W = tf.Variable(tf.abs(tf.random_normal([1])));
        b = tf.Variable(tf.random_normal([1]));

        x = tf.placeholder(tf.float32, name='x')
        y = tf.placeholder(tf.float32, name='y')

        pred_y = tf.add(tf.multiply(x, W), b);
        sub_loss = tf.abs(pred_y - y);
        # why here use pow not sqrt. it is easy overflow when use sqrt when the value is very small.
        sqrt_loss = (tf.pow(sub_loss, 2));
        # why here use mean, not use sum
        loss = tf.reduce_mean(0.5 * sqrt_loss);

        optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss);

        init = tf.global_variables_initializer();
        with tf.Session() as sess:
            sess.run(init);
            feed = {x: self.simx, y: self.simy};

            print("w:", sess.run(W), "b:", sess.run(b), "loss:", sess.run(loss, feed));
            '''
            print (sess.run(pred_y,feed));
            print (sess.run(sub_loss,feed));
            print (sess.run(sqrt_loss,feed))
            '''
            for i in range(10000):
                feed = {x: self.simx, y: self.simy};
                sess.run(optimizer, feed_dict=feed);
                if (i % 1000 == 0):
                    print("w:", sess.run(W), "b:", sess.run(b), "loss:", sess.run(loss, feed));
            self.pred_w = sess.run(W);
            self.pred_b = sess.run(b);

    def Show(self):
        plt.figure();
        plt.scatter(self.simx, self.simy, c='g', marker='o');
        plt.plot(self.simx, self.simx * self.pred_w + self.pred_b);
        plt.show();


if __name__ == "__main__":
    lr = LinearRegression();
    lr.CreateDataSet();
    lr.Train();
    lr.Show();