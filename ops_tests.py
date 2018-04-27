import numpy as np
from ops import uv_affine_transform, identity_initialization
import tensorflow as tf
import unittest


class UVAffineTransformationTest(unittest.TestCase):
    def test_transformation(self):
        """Test if the affine transformation function returns the correct result"""
        u = tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3])
        v = tf.constant([2, 2, 2, 1, 1, 1], shape=[2, 3])
        W = tf.constant(2, shape=[6, 3])
        b = tf.constant(1, shape=[3])
        t = uv_affine_transform(u, v, W, b)
        result = [[25, 25, 25], [37, 37, 37]]
        np.testing.assert_allclose(t, result)

    def test_identity_initialisation(self):
        u = tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3])
        batch_size = u.shape[0]
        embeddingsize = u.shape[1]
        no_senses=3
        with tf.Session() as sess:
            m = sess.run(identity_initialization(input=u, batch_size=batch_size, embeddingdim=embeddingsize, no_senses=no_senses))

        #np.testing.assert_equal(m.shape[0], batch_size)
        #np.testing.assert_equal(m.shape[1], no_senses)
        #np.testing.assert_equal(m.shape[2], embeddingsize)
        print(m)
        np.testing.assert_equal(m[0][0], m[0][1])

if __name__ == "__main__":
    tf.enable_eager_execution()
    unittest.main()
