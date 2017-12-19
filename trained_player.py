from player import Player
import tensorflow as tf
import functools


def lazy_property(fn):
    attribute = '_cache_' + fn.__name__

    @property
    @functools.wraps(fn)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, fn(self))
        return getattr(self, attribute)

    return decorator


def with_graph(fn):

    @functools.wraps(fn)
    def decorator(self, *args, **kwargs):
        with self.graph.as_default():
            return fn(self, *args, **kwargs)

    return decorator


class TrainedPlayer(Player):

    @with_graph
    def __init__(self):
        self.x = tf.placeholder(tf.float32, [None, 9], name="x")
        self.y_ = tf.placeholder(tf.float32, [None, 9], name="y_")

        self.prediction
        self.optimize
        self.accuracy

        self._loaded = False

    @lazy_property
    def graph(self):
        return tf.Graph()

    @lazy_property
    @with_graph
    def prediction(self):

        # Initialise weights with some noise, as suggested in Deep MNIST tutorial

        W = tf.Variable(tf.truncated_normal([9, 9], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[9]), name="b")

        y = tf.nn.softmax(tf.matmul(self.x, W) + b)

        return y

    @lazy_property
    @with_graph
    def cost(self):
        # We will use tf.nn.softmax_cross_entropy_with_logits, which is a more stable version of this calc:
        # cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(self.prediction), reduction_indices=[1]))

        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.prediction))
        return cross_entropy

    @lazy_property
    @with_graph
    def optimize(self):
        train_step = tf.train.GradientDescentOptimizer(0.5).minimize(self.cost)
        return train_step

    @lazy_property
    @with_graph
    def accuracy(self):

        correct_prediction = tf.equal(tf.argmax(self.prediction,1), tf.argmax(self.y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return accuracy

    def get_model_path(self):
        return "models/model_{0}.ckpt".format(self.name())

    @with_graph
    def save(self):
        sess = tf.get_default_session()
        saver = tf.train.Saver()
        saver.save(sess, self.get_model_path())

    @lazy_property
    def infer_session(self):
        return tf.Session(graph=self.graph)

    @with_graph
    def load(self):
        sess = self.infer_session
        saver = tf.train.Saver()
        saver.restore(sess, self.get_model_path())

    @with_graph
    def do_move(self, board):
        sess = self.infer_session

        if not self._loaded:
            self.load()
            self._loaded = True

        weights_deep = sess.run(self.prediction, feed_dict={self.x: [board]})

        weights = weights_deep[0]

        free_cells = self.get_free_cells(board)

        d = dict(zip(range(0,len(weights)), weights))

        d = [(a,d[a]) for a in free_cells]

        d = sorted(d, key=lambda x: x[1], reverse=True)

        return d[0][0]


class DeepTrainedPlayer(TrainedPlayer):

    hidden_nodes = 10

    @lazy_property
    @with_graph
    def prediction(self):

        def weight_variable(shape):
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial)

        def bias_variable(shape):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial)

        W_layer1 = weight_variable([9, self.hidden_nodes])
        b_layer1 = bias_variable([self.hidden_nodes])

        W_layer2 = weight_variable([self.hidden_nodes, 9])
        b_layer2 = bias_variable([9])

        hidden_layer1 = tf.matmul(self.x, W_layer1) + b_layer1
        hidden_layer2 = tf.nn.relu(hidden_layer1)

        output_layer = tf.matmul(hidden_layer2, W_layer2) + b_layer2

        y = tf.nn.softmax(output_layer)

        return y


class DeepTrainedPlayer2x100(DeepTrainedPlayer):

    hidden_nodes = 100


class DeepTrainedPlayer10x100(DeepTrainedPlayer):

    hidden_layers = 10
    hidden_nodes = 100

    @lazy_property
    @with_graph
    def prediction(self):

        output = self.x

        for layer in range(self.hidden_layers):
            output = tf.layers.dense(output, self.hidden_nodes, name="hidden{}".format(layer), activation=tf.nn.relu)

        y = tf.layers.dense(output, 9, name="logits", activation=tf.nn.softmax)

        return y


class DeepTrainedPlayerTF2x100(DeepTrainedPlayer10x100):

    hidden_layers = 2


class DeepTrainedPlayer5x100(DeepTrainedPlayer10x100):

    hidden_layers = 5


class DeepTrainedPlayer5x200(DeepTrainedPlayer10x100):

    hidden_layers = 5
    hidden_nodes = 200

class DeepTrainedPlayer5x300(DeepTrainedPlayer10x100):

    hidden_layers = 5
    hidden_nodes = 300


class DeepTrainedPlayer2x200(DeepTrainedPlayer10x100):

    hidden_layers = 2
    hidden_nodes = 200


class DeepTrainedPlayer2x200retrained(DeepTrainedPlayer2x200):
    pass


class DeepTrainedPlayer2x200rt_drop(DeepTrainedPlayer2x200retrained):

    hidden_layers = 2
    hidden_nodes = 200

    dropout_rate = 0.5

    @lazy_property
    @with_graph
    def prediction(self):

        self.training = tf.placeholder_with_default(False, shape=(), name="training")

        output = self.x

        output = tf.layers.dropout(output, self.dropout_rate, training=self.training)

        for layer in range(self.hidden_layers):
            output = tf.layers.dense(output, self.hidden_nodes, name="hidden{}".format(layer), activation=tf.nn.relu)
            output = tf.layers.dropout(output, self.dropout_rate, training=self.training)

        y = tf.layers.dense(output, 9, name="logits", activation=tf.nn.softmax)

        return y

    def validate_feed_dict(self, feed, training=False):
        feed[self.training] = training
        return feed


class DeepTrainedPlayer2x200drop(DeepTrainedPlayer2x200rt_drop):
    pass


