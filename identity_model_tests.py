import unittest
import tensorflow as tf
import numpy as np
from identity_model import Identity_Model
from MaskModel import MaskModel
from holst import Data


class identity_model_tests(unittest.TestCase):

    def setUp(self):
        embedding_file = "./test_data/test_embeddings.txt"
        data_file = "./test_data/test_data.txt"
        word_embeddings = Data.read_word_embeddings(embedding_file)
        self._db = Data.generate_instances(batch_size=3, file_path=data_file, word_index=word_embeddings.wv.vocab,
                                           unknown_word_key="$unknown_word$")

        self._lookup = word_embeddings.wv.syn0
        self._batch_size = 3



    def tearDown(self):
        tf.reset_default_graph()


    def test_selectionfunction(self):
        u = np.array([[2.0, 2.0, 1.0], [1.0, 1.0, 2.0], [1.0, 1.0, 1.0]])
        v = np.array([[3.0, 2.0, 1.0], [2.0, 1.0, 2.0],[1.0, 1.0, 2.0]])
        s = np.array([[[2.0, 2.0, 1.0], [1.0, 1.0, 1.0]], [[1.0, 1.0, 2.0], [2.0, 2.0, 2.0]],[[1.0, 1.0, 1.0], [2.0, 1.0, 2.0]]])
        selection_matrix = np.full(shape=(3, 6, 2), fill_value=1.0)
        selection_matrix[0][0][0] = 2.0
        selection_matrix[0][2][0] = 3.0
        selection_matrix[0][3][0] = 3.0
        selection_matrix[0][4][0] = 2.0

        selection_matrix[0][0][1] = 3.0
        selection_matrix[0][4][1] = 2.0

        selection_matrix[1][3][0] = 2.0

        selection_matrix[1][0][1] = 2.0
        selection_matrix[1][1][1] = 2.0
        selection_matrix[1][2][1] = 3.0
        selection_matrix[1][4][1] = 2.0

        selection_matrix[2][2][0] = 2.0
        selection_matrix[2][4][1] = 2.0

        selection_bias = np.full(shape=(2,), fill_value=1.0)
        model = Identity_Model(no_senses=2, vocab_size=10)
        with tf.Session() as sess:
            vec = sess.run(model.get_selection_weights(u=u, v=v, selection_matrix=selection_matrix, selection_bias=selection_bias))
            new_u = sess.run(model.select_senses(sense_matrix=s, selection_vec=vec))
        np.testing.assert_equal(u.shape, new_u.shape)
        np.testing.assert_equal(u is new_u, False)

    def test_architecture(self):
        wsd_model = Identity_Model(no_senses=4, vocab_size=6)
        with tf.Session() as sess:
            wsd_model.create_architecture(batch_size=self._batch_size, lookup=self._lookup)
            sess.run(tf.global_variables_initializer())
            for batch in range(self._db.no_batches):
                p = sess.run([wsd_model.architecture_normalized],
                             feed_dict={
                                 wsd_model._u: self._db.modifier_batches[batch],
                                 wsd_model._v: self._db.head_batches[batch],
                             })
                np.testing.assert_equal(p[0].shape, (self._db.modifier_batches[batch].shape[0], self._lookup.shape[1]))

    def test_architecture(self):
        wsd_model = MaskModel(no_senses=4, vocab_size=6)
        with tf.Session() as sess:
            wsd_model.create_architecture(batch_size=self._batch_size, lookup=self._lookup)
            sess.run(tf.global_variables_initializer())
            for batch in range(self._db.no_batches):
                p = sess.run([wsd_model.architecture_normalized],
                             feed_dict={
                                 wsd_model._u: self._db.modifier_batches[batch],
                                 wsd_model._v: self._db.head_batches[batch],
                             })
                np.testing.assert_equal(p[0].shape, (self._db.modifier_batches[batch].shape[0], self._lookup.shape[1]))




if __name__ == "__main__":
    tf.enable_eager_execution()
    unittest.main()