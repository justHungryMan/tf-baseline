# code from https://github.com/google-research/big_transfer/blob/master/bit_tf2/models.py
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ResNet architecture as used in BiT."""

import tensorflow.compat.v2 as tf


def add_name_prefix(name, prefix=None):
  return prefix + "/" + name if prefix else name


class ReLU(tf.keras.layers.ReLU):

  def compute_output_shape(self, input_shape):
    return tf.TensorShape(input_shape)


class PaddingFromKernelSize(tf.keras.layers.Layer):
  """Layer that adds padding to an image taking into a given kernel size."""

  def __init__(self, kernel_size, **kwargs):
    super(PaddingFromKernelSize, self).__init__(**kwargs)
    pad_total = kernel_size - 1
    self._pad_beg = pad_total // 2
    self._pad_end = pad_total - self._pad_beg

  def compute_output_shape(self, input_shape):
    batch_size, height, width, channels = tf.TensorShape(input_shape).as_list()
    if height is not None:
      height = height + self._pad_beg + self._pad_end
    if width is not None:
      width = width + self._pad_beg + self._pad_end
    return tf.TensorShape((batch_size, height, width, channels))

  def call(self, x):
    padding = [
        [0, 0],
        [self._pad_beg, self._pad_end],
        [self._pad_beg, self._pad_end],
        [0, 0]]
    return tf.pad(x, padding)


class StandardizedConv2D(tf.keras.layers.Conv2D):
  """Implements the abs/1903.10520 technique (see go/dune-gn).
  You can simply replace any Conv2D with this one to use re-parametrized
  convolution operation in which the kernels are standardized before conv.
  Note that it does not come with extra learnable scale/bias parameters,
  as those used in "Weight normalization" (abs/1602.07868). This does not
  matter if combined with BN/GN/..., but it would matter if the convolution
  was used standalone.
  Author: Lucas Beyer
  """

  def build(self, input_shape):
    super(StandardizedConv2D, self).build(input_shape)
    # Wrap a standardization around the conv OP.
    default_conv_op = self._convolution_op

    def standardized_conv_op(inputs, kernel):
      # Kernel has shape HWIO, normalize over HWI
      mean, var = tf.nn.moments(kernel, axes=[0, 1, 2], keepdims=True)
      # Author code uses std + 1e-5
      return default_conv_op(inputs, (kernel - mean) / tf.sqrt(var + 1e-10))

    self._convolution_op = standardized_conv_op
    self.built = True


def group_normalize(x, gamma, beta, num_groups=None, group_size=None, eps=1e-5):
  """Applies group-normalization to NHWC `x` (see abs/1803.08494, go/dune-gn).
  This function just does the math, if you want a "layer" that creates the
  necessary variables etc., see `group_norm` below.
  You must either specify a fixed number of groups `num_groups`, which will
  automatically select a corresponding group size depending on the input's
  number of channels, or you must specify a `group_size`, which leads to an
  automatic number of groups depending on the input's number of channels.
  Args:
    x: N..C-tensor, the input to group-normalize. For images, this would be a
      NHWC-tensor, for time-series a NTC, for videos a NHWTC or NTHWC, all of
      them work, as normalization includes everything between N and C. Even just
      NC shape works, as C is grouped and normalized.
    gamma: tensor with C entries, learnable scale after normalization.
    beta: tensor with C entries, learnable bias after normalization.
    num_groups: int, number of groups to normalize over (divides C).
    group_size: int, size of the groups to normalize over (divides C).
    eps: float, a small additive constant to avoid /sqrt(0).
  Returns:
    Group-normalized `x`, of the same shape and type as `x`.
  Author: Lucas Beyer
  """
  assert x.shape.ndims >= 2, (
      "Less than 2-dim Tensor passed to GroupNorm. Something's fishy.")

  num_channels = x.shape[-1]
  assert num_channels is not None, "Cannot apply GroupNorm on dynamic channels."
  assert (num_groups is None) != (group_size is None), (
      "You must specify exactly one of `num_groups`, `group_size`")

  if group_size is not None:
    num_groups = num_channels // group_size

  assert num_channels % num_groups == 0, (
      "GroupNorm: {} not divisible by {}".format(num_channels, num_groups))

  orig_shape = tf.shape(x)

  # This shape is NHWGS where G is #groups and S is group-size.
  extra_shape = [num_groups, num_channels // num_groups]
  group_shape = tf.concat([orig_shape[:-1], extra_shape], axis=-1)
  x = tf.reshape(x, group_shape)

  # The dimensions to normalize over: HWS for images, but more generally all
  # dimensions except N (batch, first) and G (cross-groups, next-to-last).
  # So more visually, normdims are the dots in N......G. (note the last one is
  # also a dot, not a full-stop, argh!)
  normdims = list(range(1, x.shape.ndims - 2)) + [x.shape.ndims - 1]
  mean, var = tf.nn.moments(x, normdims, keepdims=True)

  # Interestingly, we don't have a beta/gamma per group, but still one per
  # channel, at least according to the original paper. Reshape such that they
  # broadcast correctly.
  beta = tf.reshape(beta, extra_shape)
  gamma = tf.reshape(gamma, extra_shape)
  x = tf.nn.batch_normalization(x, mean, var, beta, gamma, eps)
  return tf.reshape(x, orig_shape)


class GroupNormalization(tf.keras.layers.Layer):
  """A group-norm "layer" (see abs/1803.08494 go/dune-gn).
  This function creates beta/gamma variables in a name_scope, and uses them to
  apply `group_normalize` on the input `x`.
  You can either specify a fixed number of groups `num_groups`, which will
  automatically select a corresponding group size depending on the input's
  number of channels, or you must specify a `group_size`, which leads to an
  automatic number of groups depending on the input's number of channels.
  If you specify neither, the paper's recommended `num_groups=32` is used.
  Authors: Lucas Beyer, Joan Puigcerver.
  """

  def __init__(self,
               num_groups=None,
               group_size=None,
               eps=1e-5,
               beta_init=tf.zeros_initializer(),
               gamma_init=tf.ones_initializer(),
               **kwargs):
    """Initializer.
    Args:
      num_groups: int, the number of channel-groups to normalize over.
      group_size: int, size of the groups to normalize over.
      eps: float, a small additive constant to avoid /sqrt(0).
      beta_init: initializer for bias, defaults to zeros.
      gamma_init: initializer for scale, defaults to ones.
      **kwargs: other tf.keras.layers.Layer arguments.
    """
    super(GroupNormalization, self).__init__(**kwargs)
    if num_groups is None and group_size is None:
      num_groups = 32

    self._num_groups = num_groups
    self._group_size = group_size
    self._eps = eps
    self._beta_init = beta_init
    self._gamma_init = gamma_init

  def build(self, input_size):
    channels = input_size[-1]
    assert channels is not None, "Cannot apply GN on dynamic channels."
    self._gamma = self.add_weight(
        name="gamma", shape=(channels,), initializer=self._gamma_init,
        dtype=self.dtype)
    self._beta = self.add_weight(
        name="beta", shape=(channels,), initializer=self._beta_init,
        dtype=self.dtype)
    super(GroupNormalization, self).build(input_size)

  def call(self, x):
    return group_normalize(x, self._gamma, self._beta, self._num_groups,
                           self._group_size, self._eps)


class BottleneckV2Unit(tf.keras.layers.Layer):
  """Implements a standard ResNet's unit (version 2).
  """

  def __init__(self, num_filters, stride=1, weight_decay=0, **kwargs):
    """Initializer.
    Args:
      num_filters: number of filters in the bottleneck.
      stride: specifies block's stride.
      **kwargs: other tf.keras.layers.Layer keyword arguments.
    """
    super(BottleneckV2Unit, self).__init__(**kwargs)
    self._num_filters = num_filters
    self._stride = stride
    self.weight_decay = weight_decay

    self._proj = None
    self._unit_a = tf.keras.Sequential([
        GroupNormalization(name="group_norm"),
        ReLU(),
    ], name="a")
    self._unit_a_conv = StandardizedConv2D(
        filters=num_filters,
        kernel_size=1,
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=tf.keras.regularizers.l2(weight_decay / 2),
        padding="VALID",
        trainable=self.trainable,
        name="a/standardized_conv2d")

    self._unit_b = tf.keras.Sequential([
        GroupNormalization(name="group_norm"),
        ReLU(),
        PaddingFromKernelSize(kernel_size=3),
        StandardizedConv2D(
            filters=num_filters,
            kernel_size=3,
            kernel_initializer='he_normal',
            kernel_regularizer=tf.keras.regularizers.l2(weight_decay / 2),
            strides=stride,
            use_bias=False,
            padding="VALID",
            trainable=self.trainable,
            name="standardized_conv2d")
    ], name="b")

    self._unit_c = tf.keras.Sequential([
        GroupNormalization(name="group_norm"),
        ReLU(),
        StandardizedConv2D(
            filters=4 * num_filters,
            kernel_size=1,
            kernel_initializer='he_normal',
            kernel_regularizer=tf.keras.regularizers.l2(weight_decay / 2),
            use_bias=False,
            padding="VALID",
            trainable=self.trainable,
            name="standardized_conv2d")
    ], name="c")

  def build(self, input_shape):
    input_shape = tf.TensorShape(input_shape).as_list()

    # Add projection layer if necessary.
    if (self._stride > 1) or (4 * self._num_filters != input_shape[-1]):
      self._proj = StandardizedConv2D(
          filters=4 * self._num_filters,
          kernel_size=1,
          strides=self._stride,
          kernel_initializer='he_normal',
          kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay / 2),
          use_bias=False,
          padding="VALID",
          trainable=self.trainable,
          name="a/proj/standardized_conv2d")
    self.built = True

  def compute_output_shape(self, input_shape):
    current_shape = self._unit_a.compute_output_shape(input_shape)
    current_shape = self._unit_a_conv.compute_output_shape(current_shape)
    current_shape = self._unit_b.compute_output_shape(current_shape)
    current_shape = self._unit_c.compute_output_shape(current_shape)
    return current_shape

  def call(self, x):
    x_shortcut = x
    # Unit "a".
    x = self._unit_a(x)
    if self._proj is not None:
      x_shortcut = self._proj(x)
    x = self._unit_a_conv(x)
    # Unit "b".
    x = self._unit_b(x)
    # Unit "c".
    x = self._unit_c(x)

    return x + x_shortcut


class ResnetV2(tf.keras.Model):
  """Generic ResnetV2 architecture, as used in the BiT paper."""

  def __init__(self,
               num_units=(3, 4, 6, 3),
               num_outputs=1000,
               filters_factor=4,
               strides=(1, 2, 2, 2),
               weight_decay=0,
               **kwargs):
    super(ResnetV2, self).__init__(**kwargs)

    num_blocks = len(num_units)
    num_filters = tuple(16 * filters_factor * 2**b for b in range(num_blocks))
    
    self.weight_decay = weight_decay
    self._root = self._create_root_block(num_filters=num_filters[0])
    self._blocks = []
    for b, (f, u, s) in enumerate(zip(num_filters, num_units, strides), 1):
      n = "block{}".format(b)
      self._blocks.append(
          self._create_block(num_units=u, num_filters=f, stride=s, name=n))
    self._pre_head = [
        GroupNormalization(name="group_norm"),
        ReLU(),
        tf.keras.layers.GlobalAveragePooling2D()
    ]
    self._head = None
    if num_outputs:
      self._head = tf.keras.layers.Dense(
          units=num_outputs,
          use_bias=True,
          kernel_initializer="zeros",
          kernel_regularizer=tf.keras.regularizers.l2(weight_decay / 2),
          bias_regularizer=tf.keras.regularizers.l2(weight_decay / 2),
          trainable=self.trainable,
          name="head/dense")

  def _create_root_block(self,
                         num_filters,
                         conv_size=7,
                         conv_stride=2,
                         pool_size=3,
                         pool_stride=2):
    layers = [
        PaddingFromKernelSize(conv_size),
        StandardizedConv2D(
            filters=num_filters,
            kernel_size=conv_size,
            kernel_initializer='he_normal',
            kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay / 2),
            strides=conv_stride,
            trainable=self.trainable,
            use_bias=False,
            name="standardized_conv2d"),
        PaddingFromKernelSize(pool_size),
        tf.keras.layers.MaxPool2D(
            pool_size=pool_size, strides=pool_stride, padding="valid")
    ]
    return tf.keras.Sequential(layers, name="root_block")

  def _create_block(self, num_units, num_filters, stride, name):
    layers = []
    for i in range(1, num_units + 1):
      layers.append(
          BottleneckV2Unit(
              num_filters=num_filters,
              stride=(stride if i == 1 else 1),
              weight_decay=self.weight_decay,
              name="unit%02d" % i))
    return tf.keras.Sequential(layers, name=name)

  def compute_output_shape(self, input_shape):
    current_shape = self._root.compute_output_shape(input_shape)
    for block in self._blocks:
      current_shape = block.compute_output_shape(current_shape)
    for layer in self._pre_head:
      current_shape = layer.compute_output_shape(current_shape)
    if self._head is not None:
      batch_size, features = current_shape.as_list()
      current_shape = (batch_size, 1, 1, features)
      current_shape = self._head.compute_output_shape(current_shape).as_list()
      current_shape = (current_shape[0], current_shape[3])
    return tf.TensorShape(current_shape)

  def call(self, x):
    x = self._root(x)
    for block in self._blocks:
      x = block(x)
    for layer in self._pre_head:
      x = layer(x)
    if self._head is not None:
      x = self._head(x)
    return x


KNOWN_MODELS = {
    f'R{l}x{w}': f'gs://bit_models/R{l}x{w}.h5'
    for l, w in [(50, 1), (50, 3), (101, 1), (101, 3), (152, 4)]
}

NUM_UNITS = {
    k: (3, 4, 6, 3) if 'R50' in k else
       (3, 4, 23, 3) if 'R101' in k else
       (3, 8, 36, 3)
    for k in KNOWN_MODELS
}


def create_name(arechitecture_name, num_outputs, weight_decay=0, **kwargs):
    filter_factor = int(arechitecture_name.split('x')[-1]) * 4
    return ResnetV2(num_units=NUM_UNITS[arechitecture_name],
                    num_outputs=num_outputs,
                    filters_factor=filter_factor,
                    strides=(1, 2, 2, 2),
                    trainable=True,
                    weight_decay=weight_decay,
                    dtype=tf.float32,
                    **kwargs)