# See: https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Distribution
# https://github.com/tensorflow/probability/blob/v0.11.0/tensorflow_probability/python/distributions/student_t.py#L149-L450

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

class SkewT(tfd.Distributions):
    def __init__(self, loc, scale, df, skew, validate_args=False,
                 allow_nan_stats=True, name='SkewT'):
        parameters = dict(locals())
        with tf.name_scope(name) as name:
          dtype = dtype_util.common_dtype([loc, scale, df, skew], tf.float32)
          self._loc = tensor_util.convert_nonref_to_tensor(
              loc, name='loc', dtype=dtype)
          self._scale = tensor_util.convert_nonref_to_tensor(
              scale, name='scale', dtype=dtype)
          self._df = tensor_util.convert_nonref_to_tensor(
              df, name='df', dtype=dtype)
          self._skew = tensor_util.convert_nonref_to_tensor(
              skew, name='skew', dtype=dtype)
          dtype_util.assert_same_float_dtype((self._loc, self._scale, self._df, self.skew))
          super(SkewT, self).__init__(
              dtype=dtype,
              reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
              validate_args=validate_args,
              allow_nan_stats=allow_nan_stats,
              parameters=parameters,
              name=name)

    @staticmethod
    def _param_shapes(sample_shape):
    return dict(
        zip(('loc', 'scale', 'df', 'skew'),
            ([tf.convert_to_tensor(sample_shape, dtype=tf.int32)] * 4)))

