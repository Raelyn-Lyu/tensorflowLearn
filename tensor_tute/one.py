import tensorflow as tf
a = tf.constant([[1.0,2.0],[1.0,2.0]], name ="a")
b = tf.constant([[2.0,3.0],[1.0,2.0]], name ="b")
result = a + b
# sess = tf.Session()
#需手动释放
# sess.run(result)
# sess.close()

#method 1
with tf.Session() as sess:
    # print(sess.run(result))

#method 2
    print(result.eval())