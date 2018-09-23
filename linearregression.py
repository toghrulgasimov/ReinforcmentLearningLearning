import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
x_p = []
y_p = []
a = 0.22
b = 0.78

for i in range(500) :
    x = np.random.normal(0,0.5)
    y = a*x + b+np.random.normal(0,0.1)
    x_p.append(x)
    y_p.append(y)
plt.plot(x_p, y_p,'o')
plt.legend()
plt.show()
ses = tf.Session()
A = tf.Variable(tf.random_uniform([1],-1,1))
b = tf.Variable(tf.zeros([1]))

y = A * x_p + b
cost = tf.reduce_mean(tf.square(y - y_p))
optimizer = tf.train.GradientDescentOptimizer(0.1)
train = optimizer.minimize(cost)
model = tf.initialize_all_variables()
ses.run(model)


for i in range(21) :
    ses.run(train)
    print ses.run(A)
    plt.plot(x_p, y_p, 'o')
    plt.plot(x_p, x_p*ses.run(A)+ses.run(b))
    plt.legend()
    plt.show()
