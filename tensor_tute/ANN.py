import tensorflow as tf
from numpy.random import RandomState

batch_size = 8
#define the weight for ANN
w1 = tf.Variable(tf.random_normal([2,3], stddev=1,seed=1))
w2 = tf.Variable(tf.random_normal([3,1], stddev=1,seed=1))

x = tf.placeholder(tf.float32,shape = (None,2), name="x_input")
y_ = tf.placeholder(tf.float32,shape = (None,1), name="y_input")

#define the forward propagation
a = tf.matmul(x,w1)
y = tf.matmul(a,w2)

#define cost function and back propagation algorithm
y = tf.sigmoid(y)
cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y,1e-10,1.0))+(1-y_)*tf.log(tf.clip_by_value(1-y,1e-10,1.0)))
#define training scheme
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

#randomly generate training set
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size,2)

Y = [[int(x1+x2<1)] for (x1,x2) in X]

# create session
with tf.Session() as sess:
    #initialize all variables
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    print("w1=",sess.run(w1))
    print("w2=",sess.run(w2))

    #training
    STEP = 50000
    for i in range(STEP):
        #each time choose a batch_size of data to process training
        start = (i*batch_size)%dataset_size
        end = min(start+batch_size,dataset_size)

        #train and update weights
        sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})
        if i % 5000 == 0:
            total_cross_entropy = sess.run(cross_entropy,feed_dict={x:X,y_:Y})
            print ("After %d training steps,cross entropy on all data is %g" %(i,total_cross_entropy))
    print(sess.run(w1))
    print(sess.run(w2))
