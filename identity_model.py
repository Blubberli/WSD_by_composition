import tensorflow as tf
from holst import AbstractModel
import ops

class Identity_Model(AbstractModel.AbstractModel):

    def __init__(self, no_senses, vocab_size):
        super(Identity_Model, self).__init__()
        self._no_senses = no_senses
        self._vocab_size = vocab_size

    def create_architecture(self, batch_size, lookup):
        self._u = tf.placeholder(tf.int64, shape=[batch_size])
        self._v = tf.placeholder(tf.int64, shape=[batch_size])

        print(self._u.shape)

        # lookup of the embeddings
        self._embeddings_u = tf.nn.embedding_lookup(lookup, self._u)
        self._embeddings_v = tf.nn.embedding_lookup(lookup, self._v)

        embedding_size = lookup.shape[1]

        # initialization variables
        mean = 0
        stddev = 0.0001

        # embedding weight matrix. shape: n x n
        self._W = tf.get_variable("W", shape=[embedding_size * 2, embedding_size])

        # biasvector. shape: n
        self._b = tf.get_variable("b", shape=[embedding_size])

        #desicion matrix
        # create a lookuptable that contains the word matrices for all heads / modifier
        with tf.device("/cpu:0"):
            gauss_matrix = tf.random_normal(shape=[self._vocab_size, embedding_size*2, self._no_senses],
                                            mean=mean, stddev=stddev)
            self._matrix_lookup = tf.get_variable("desicion_matrix_lookup", initializer=gauss_matrix)

            #lookuptable for sensematrix. each sense is initialized by original word embedding
            identity_matrix = ops.identity_initialization(lookup[0:self._vocab_size], self._vocab_size, self._no_senses,
                                                          embedding_size)
            self._senseMatrix = tf.get_variable("s", initializer=identity_matrix)
            print(self._senseMatrix.shape)

        self._D = tf.nn.embedding_lookup(self._matrix_lookup, self._u)
        s = tf.nn.embedding_lookup(self._senseMatrix, self._u)
        self._db = tf.get_variable("db", shape=[self._no_senses])

        self._selection_weights = self.get_selection_weights(self._embeddings_u, self._embeddings_v, self._D, self._db)
        self._specialized_modifier = self.select_senses(s, self._selection_weights)
        self._specialized_modifier = super(Identity_Model,
            self).l2_normalization_layer(
            self._specialized_modifier,
            1)

        self._architecture = self.compose(u=self._specialized_modifier, v=self._embeddings_v,
                                          W=self._W, b=self._b, axis=1)


        self._architecture_normalized = super(
            Identity_Model,
            self).l2_normalization_layer(
            self._architecture,
            1)

    def select_senses(self, sense_matrix, selection_vec):
        print("sense matrix")
        print(sense_matrix.shape)
        vecs = tf.multiply(tf.expand_dims(selection_vec, axis=2), sense_matrix)
        return tf.reduce_sum(vecs, axis=1)


    def compose(self, u, v, W, b, axis):
        """
        composition of the form:
        p = g(W[v; u] + b)
        takes as input two tensors of any shape (tensors need to have at least rank 2)
        concatenates vector v and u and multiplies the concatenation by weightmatrix.
        adds biasvector b
        """
        return ops.uv_affine_transform(u, v, W, b)

    def get_selection_weights(self, u, v, selection_matrix, selection_bias):
        """
        Return the selected sense for a given input
        """
        con = tf.concat(values=[u, v], axis=1)
        con = tf.expand_dims(con, axis=1)
        print("D")
        print(selection_matrix.shape)
        print("input")
        print(con.shape)
        selection_weights = tf.squeeze(tf.matmul(con, selection_matrix), axis=1)
        selection_weights = tf.add(selection_weights, selection_bias)
        selection_weights = tf.nn.softmax(selection_weights)
        print("selection weights")
        print(selection_weights.shape)
        return selection_weights


    @property
    def W(self):
        return self._W

    @property
    def b(self):
        return self._b

    @property
    def no_sense(self):
        return self._no_senses

    @property
    def vocab_size(self):
        return self._vocab_size

    @property
    def selection_weights(self):
        return self._selection_weights

    @property
    def senseMatrix(self):
        return self._senseMatrix

    @property
    def specialized_modifier(self):
        return self._specialized_modifier


