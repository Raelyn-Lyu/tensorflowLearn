import tensorflow as tf

# define ipt and opt nodes
IPT_NODES = 28 * 28
OPT_NODES = 10
LAYER1_NODES = 500

def get_weight_variable(shape,regularizer):
    weights = tf.get_variable(
        "weights",shape,initializer=tf.truncated_normal_initializer(stddev=0.1)
    )

    '''truncated normal initializer more suitable for the nn, the value that more than two
    standard deviation over mean is discard'''

    if regularizer != None:
        tf.add_to_collection('loss',regularizer(weights))
    return weights

# Forward propagate
def inference(input_tensor,regularizer):
    with tf.variable_scope('layer1'):
        weights = get_weight_variable([IPT_NODES,LAYER1_NODES],regularizer)
        bias = tf.get_variable("bias",[LAYER1_NODES],initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor,weights)+bias)
    with tf.vaiable_scope(layer2):
        weights = get_weight_variable([LAYER1_NODES,OPT_NODES],regularizer)
        bias = tf.get_variable("bias",[OPT_NODES],initializer=tf.constant_initializer(0.0))
        layers2 = tf.nn.relu(tf.matmul(layer1,weights)+bias)
    return layer2