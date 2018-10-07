from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import  os
import inference

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULAR_RATE = 0.0001
TRAINING_STEP = 30000
MOVING_AVERAGE_DECAY = 0.99

SAVE_PATH = "path/"
MODEL_NAME = "model.ckpt"

def train(mnist):
    x = tf.placeholder(tf.float32, [None,inference.IPT_NODES], name ='x_ipt')
    y_ = tf.placeholder(tf.float32, [None,inference.OPT_NODES], name = 'y_ipt')

    regularizer = tf.contrib.layers.l2_regularizer(REGULAR_RATE)

    y = inference.inference(x,regularizer)
    global_step = tf.Variable(0,trainable=False)

    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = y, labels= tf.argmax(y_,1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,global_step,mnist.train.num_examples /BATCH_SIZE, LEARNING_RATE_DECAY
    )

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
    with tf.control_dependencies([train_step,variable_averages_op]):
        train_op = tf.no_op(name = 'train')

    saver = tf.train.Saver
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for i in range(TRAINING_STEP):
            xs,ys = mnist.train.next_batch(BATCH_SIZE)
            print("shape",ys.shape)
            _,loss_value,step = sess.run([train_op,loss,global_step], feed_dict ={x:xs,y:ys})

            if i % 1000 == 0:
                print("After %d training step, loss on training" "batch is %g." %(step,loss_value))
            saver.save(sess,os.path.join(SAVE_PATH,MODEL_NAME),global_step=global_step)
def main(argv=None):
    mnist = input_data.read_data_sets("path/", one_hot=True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()