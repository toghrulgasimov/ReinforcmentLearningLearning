import tensorflow as tf
from tensorflow.examples.tutorials.mnist import  input_data
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(threshold=np.nan)

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

n_h1 = 20
n_h2 = 20
n_h3 = 20

clas = 10
b = 1

data = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

h1 = {'w':tf.Variable(tf.random_normal([784, n_h1])),
          'b':tf.Variable(tf.random_normal([n_h1]))}
h2 = {'w':tf.Variable(tf.random_normal([n_h2, n_h3])),
          'b':tf.Variable(tf.random_normal([n_h2]))}
h3 = {'w':tf.Variable(tf.random_normal([n_h2, n_h3])),
          'b':tf.Variable(tf.random_normal([n_h3]))}
ol = {'w':tf.Variable(tf.random_normal([n_h3, clas])),
          'b':tf.Variable(tf.random_normal([clas]))}

l1 = tf.add(tf.matmul(data, h1['w']), h1['b'])#using namespace syd int main
l1 = tf.nn.relu(l1)

l2 = tf.add(tf.matmul(l1, h2['w']), h2['b'])
l2 = tf.nn.relu(l2)

l3 = tf.add(tf.matmul(l2, h3['w']), h3['b'])
l3 = tf.nn.relu(l3)

out = tf.add(tf.matmul(l3, ol['w']), ol['b'])

f = tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=y)
cost = tf.reduce_mean(f)
optimizer = tf.train.AdamOptimizer().minimize(cost)

ses = tf.Session()
tf.summary.scalar("losssss", cost)
merged_summary = tf.summary.merge_all()
writer = tf.summary.FileWriter("/home/gasimov/Desktop/Developer/tensor/ant",ses.graph)
writer.add_graph(ses.graph)

ses.run(tf.initialize_all_variables())
epochs = 10
c = 0
for i in range(epochs) :
    los = 0
    for j in range(int(mnist.train._num_examples/b)) :
        e_x, e_y = mnist.train.next_batch(b)
        t = ses.run([optimizer, cost, merged_summary], feed_dict={data:e_x,y:e_y})
        los = los + t[1]
        print(los)
        c = c + 1
        writer.add_summary(t[2],c)
        ans = ses.run(f,feed_dict={data:e_x,y:e_y});
        #print (ans)

    print (los)
    print ('Epoch', i, 'loss:', los)



ses.close()
writer.close()
