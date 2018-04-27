import tensorflow as tf
from enum import Enum
from ops import entropy

print(tf.__version__)

class RunMode(Enum):
    training = 1
    validation = 2
    prediction = 3

class TrainingGraph():
    def __init__(self, wsd_model, batch_size, lookup_table, learning_rate, run_mode, alpha):
        self._model = wsd_model
        self._is_training = self.model._is_training

        self._original_vector = tf.placeholder(dtype=tf.int64, shape=[batch_size])
        self._original_embeddings = tf.nn.embedding_lookup(params=lookup_table, ids=self._original_vector)

        self._predictions = self._model._architecture_normalized
        self._selection_weights = self._model._selection_weights
        self._loss = self.get_loss(alpha)

        self._train_op = tf.no_op()

        if run_mode is RunMode.training:
           self._train_op = tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(self._loss)



    def get_loss(self, alpha):
        l1 = tf.losses.cosine_distance(labels=self._original_embeddings,
                                               predictions=self._predictions, axis=1, reduction=tf.losses.Reduction.SUM)
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
