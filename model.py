import tensorflow as tf
from holst import AbstractModel

class model(AbstractModel.AbstractModel):

    def __init__(self, no_senses):
        super(model, self).__init__()
        self._no_senses = no_senses

    def create_architecture(self, batch_size, lookup):
        self._u = tf.placeholder(tf.int64, shape=[batch_size])
        self._v = tf.placeholder(tf.int64, shape=[batch_size])

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
        self._D = tf.nn.embedding_lookup(self._matrix_lookup, self._u)

        self._db = tf.get_variable("db", shape=[self._no_senses])

        self._architecture = self.compose(u=self._embeddings_u, v=self._embeddings_v,
                                          W=self._W, b=self._b, axis=1)

        # adds nonlinearity to the composition
        self._architecture = self.nonlinearity(self._architecture)

        self._architecture_normalized = super(
            model,
            self).l2_normalization_layer(
            self._architecture,
            1)

    def compose(self, u, v, W, b, axis):
        """
        composition of the form:
        p = g(W[v; u] + b)
        takes as input two tensors of any shape (tensors need to have at least rank 2)
        concatenates vector v and u and multiplies the concatenation by weightmatrix.
        adds biasvector b
        """
        return uv_affine_transform(u, v, W, b)



