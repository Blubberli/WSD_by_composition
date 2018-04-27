import unittest
import numpy as np
from TrainingGraph import TrainingGraph, RunMode
from identity_model import Identity_Model
from holst import Data
from MaskModel import MaskModel
import tensorflow as tf

class TrainingGraphTest(unittest.TestCase):
    """
    This class contains the tests for the TrainingGraph
    """

    def setUp(self):
        embedding_file = "./test_data/test_embeddings.txt"
        data_file = "./test_data/test_data.txt"
        word_embeddings = Data.read_word_embeddings(embeddings_file=embedding_file)
        self._db = Data.generate_instances(batch_size=3, file_path=data_file,
                                           word_index=word_embeddings.wv.vocab, unknown_word_key="$unknown_word$")
        self._lookup = word_embeddings.wv.syn0
        self._batch_size = self._db.modifier_batches[0].shape[0]
        tf.set_random_seed(1)

    def tearDown(self):
        tf.reset_default_graph()

    def get_train_model(self, model, alpha):
        model.create_architecture(batch_size=3, lookup=self._lookup)
        train_model = TrainingGraph(
            wsd_model=model,
            batch_size=3,
            lookup_table=self._lookup,
            run_mode=RunMode.training,
            learning_rate=0.01,
            alpha=alpha)
        return train_model

    def run_model(self, train_model, sess):
        losses = []
        sess.run(tf.global_variables_initializer())
        for epoch in range(10):
            train_loss = 0.0
            for tidx in range(self._db.no_batches):
                loss, _ = sess.run(
                    [train_model.loss, train_model.train_op],
                    feed_dict={train_model.original_vector: self._db.compound_batches[tidx],
                               train_model.model._u: self._db.modifier_batches[tidx],
                               train_model.model._v: self._db.head_batches[tidx]})
                train_loss += loss
            train_loss /= self._db.no_batches
            losses.append(train_loss)
        return losses

    # test if the loss is going down for different composition models
    def test_loss_identity(self):
        with tf.Session() as sess:
            wsd_model = Identity_Model(vocab_size=6, no_senses=4)
            train_model = self.get_train_model(wsd_model, 0.3)
            losses = self.run_model(train_model, sess)
        np.testing.assert_equal(losses[0] > losses[9], True)

    def test_loss_MaskModel(self):
        with tf.Session() as sess:
            wsd_model = MaskModel(vocab_size=6, no_senses=4)
            train_model = self.get_train_model(wsd_model, 0.3)
            losses = self.run_model(train_model, sess)
        np.testing.assert_equal(losses[0] > losses[9], True)



if __name__ == "__main__":
    unittest.main()