import tensorflow as tf
import numpy as np

a = tf.Variable(12)

b = tf.Variable(12)

c = tf.add(a,b)

model = tf.initialize_all_variables()
with tf.Session() as sess :
    #print(sess.run(a))
    sess.run(model)
    print(sess.run(c))
