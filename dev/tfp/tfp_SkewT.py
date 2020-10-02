# See: https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Distribution
# https://github.com/tensorflow/probability/blob/v0.11.0/tensorflow_probability/python/distributions/student_t.py#L149-L450

import tensorflow_probability as tfp
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow_probability.python.internal import samplers

def skew_t_lpdf(x, nu, loc, scale, skew):
    z = (x - loc) / scale
    u = skew * z * tf.sqrt((nu + 1) / (nu + z * z));
    kernel = (tfd.StudentT(nu, 0, 1).log_prob(z) + 
              tfd.StudentT(nu + 1, 0, 1).log_cdf(u))
    return kernel + tf.math.log(2/scale)


# FIXME!
# See: https://github.com/tensorflow/probability/blob/7cf006d6390d1d0a6fe541e5c49196a58a22b875/tensorflow_probability/python/distributions/student_t.py#L43
def random_skew_t(shape, nu, loc, scale, phi, dtype, seed=None):
    gamma_seed, tn_seed, normal_seed = samplers.split_seed(seed, n=3, salt='skew_t')
    w = tf.random.gamma(shape, nu/2, nu/2, dtype=dtype, seed=gamma_seed[0])
    z = tfd.TruncatedNormal(loc, scale, 0.0, np.inf).sample(shape, seed=tn_seed[0])
    delta = phi / tf.sqrt(1 + phi * phi)
    return tf.random.normal(shape,
                            loc + scale * z * delta,
                            scale * tf.sqrt(1 - delta * delta), dtype=dtype, seed=normal_seed[0])


n = [100000]
x = tf.random.normal(n)
nu = tf.random.uniform(n)
loc = tf.random.normal(n)
scale = tf.random.uniform(n)
skew = tf.random.normal(n)
skew_t_lpdf(x, nu, loc, scale, skew)

random_skew_t([], 1., 1., 1., 1., tf.float32)
