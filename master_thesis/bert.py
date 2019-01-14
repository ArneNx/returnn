import tensorflow as tf

from TFNetworkLayer import LayerBase
from TFUtil import Data


class GatherPositionsLayer(LayerBase):
  """
  Gathers the vectors at the specific positions.
  """
  layer_class = "gather_positions"

  def __init__(self, **kwargs):
    """
    :param LayerBase|None seq_len_source: if not given, uses source
    :param str|int axis:
    :param float mask_value:
    """

    super(GatherPositionsLayer, self).__init__(**kwargs)
    data = self.sources[0].output.get_placeholder_as_batch_major()
    positions = self.sources[1].output.get_placeholder_as_batch_major()
    data_shape = tf.shape(data)
    batch_size = data_shape[0]
    seq_length = data_shape[1]
    # data = tf.print(data, [data, tf.shape(data)], "Print my data before flattening", summarize=100)
    data_flat = tf.reshape(data, [batch_size * seq_length, -1])  # (B * T, D)

    offsets_flat = tf.reshape(tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
    # positions = tf.print(positions, [positions, tf.shape(positions)], "Print my positons", summarize=100)
    positions_flat = tf.reshape(positions + offsets_flat, [-1])
    data = tf.gather(data_flat, positions_flat)
    data = tf.reshape(data, [batch_size, tf.shape(positions)[1], -1])
    # data = tf.print(data, [data, tf.shape(data)], "Print my data", summarize=100)
    # data = positions.get_sequence_mask()
    self.output.placeholder = data
    self.output.size_placeholder = {
      0: self.sources[1].output.size_placeholder[0]
    }

  @classmethod
  def get_out_data_from_opts(cls, sources, name, *args, **kwargs):
    data = Data(
      name=name,
      shape= sources[0].output.shape,
      dim=sources[0].output.dim,
      sparse=False,
      batch_dim_axis=sources[0].output.batch_dim_axis,
      time_dim_axis=sources[0].output.time_dim_axis,
      feature_dim_axis=sources[0].output.feature_dim_axis_or_unspecified,
      dtype=sources[0].output.dtype,
      beam_size=sources[0].output.beam_size)
    return data


