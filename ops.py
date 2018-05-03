import tensorflow as tf


def uv_affine_transform(u, v, W, b):
    """
    Concatenate u and v, and apply an affine transformation.

    W[u;v] + b

    u and v should have shape [batch_size, embed_size].
    """
    concatenation = tf.concat(values=[u, v], axis=1)
    return tf.nn.xw_plus_b(concatenation, W, b)


def entropy(probs, normalize):
    entropy = -tf.reduce_sum(tf.multiply(probs, tf.log(probs)), axis=1)

    if normalize:
        n_outcomes = tf.cast(probs.shape[1], tf.float32)
        entropy = tf.div(entropy, tf.log(n_outcomes))

    return tf.reduce_sum(entropy)


def identity_initialization(input, vocab_size, no_senses, embeddingdim):
    return tf.reshape(tf.tile(tf.identity(input), multiples=[1, no_senses]), shape=[vocab_size, no_senses, embeddingdim])