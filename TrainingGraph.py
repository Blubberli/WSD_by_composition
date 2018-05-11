import tensorflow as tf
from enum import Enum
from ops import entropy

print(tf.__version__)

class RunMode(Enum):
    training = 1
    validation = 2
    prediction = 3

class TrainingGraph():
    def __init__(self, wsd_model, batch_size, lookup_table, learning_rate, run_mode, alpha, both_positions):
        self._model = wsd_model
        self._is_training = self.model._is_training

        self._original_vector = tf.placeholder(dtype=tf.int64, shape=[batch_size])
        self._original_embeddings = tf.nn.embedding_lookup(params=lookup_table, ids=self._original_vector)

        self._predictions = self._model._architecture_normalized

        if both_positions:
            self._selection_weights_mod = self._model._selection_weights_mod
            self._selection_weights_head = self._model._selection_weights_head
        else:
            self._selection_weights = self._model._selection_weights

        self._loss = self.get_loss(alpha, both_positions)

        self._train_op = tf.no_op()

        if run_mode is RunMode.training:
           self._train_op = tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(self._loss)



    def get_loss(self, alpha, both_positions):
        l1 = tf.losses.cosine_distance(labels=self._original_embeddings,
                                               predictions=self._predictions, axis=1, reduction=tf.losses.Reduction.SUM)
        if both_positions:
            l2_mod = entropy(self._selection_weights_mod, True)
            l2_head = entropy(self._selection_weights_head, True)
            l2 = (l2_mod+l2_head)/2
        else:
            l2 = entropy(self._selection_weights, True)
        return l1+alpha*l2

    @property
    def is_training(self):
        return self._is_training

    @property
    def model(self):
        return self._model

    @property
    def original_vector(self):
        return self._original_vector

    @property
    def architecture(self):
        return self._architecture

    @property
    def loss(self):
        return self._loss

    @property
    def reg_loss(self):
        return self._reg_loss

    @property
    def predictions(self):
        return self._predictions

    @property
    def train_op(self):
        return self._train_op
