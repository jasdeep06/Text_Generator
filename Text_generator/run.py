import tensorflow as tf
import numpy as np
a=tf.Variable(3.0)
b=tf.Variable(4.0)
c=tf.Variable(5.0)
x=tf.placeholder("float")


#assign=tf.assign(b,tf.subtract(b,3.0))
product=tf.multiply(a,b)

add=tf.add(product,x)

"""""
with tf.control_dependencies([assign]):
    add = tf.add(product, c)

"""



sess=tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
#print(sess.run(product))
for i in range(2):
    print(sess.run(add,feed_dict={x:4.0}))
    print(sess.run(add,feed_dict={x:4.0}))


