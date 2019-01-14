from threading import Thread

import h5py
import numpy
import sys
import os

from Dataset import DatasetSeq
from LmDataset import TranslationDataset

class MaskedLmDataset(TranslationDataset):
  """
  This class is used to handle the data for a BERT
  It gets a directory and expects these files:

      source.dev(.gz)?
      source.train(.gz)?
      source.vocab.pkl
      target.dev(.gz)?
      target.train(.gz)?
      target.vocab.pkl
      clean_source.dev(.gz)?
      clean_source.train(.gz)?
      positions.dev.npy
      positions.train.npy
  """

  MapToDataKeys = {"source": "data",
                   "target": "classes",
                   }  # just by our convention
  _main_data_key = None
  _main_classes_key = None

  def __init__(self, delayed_seq_data_start_symbol, add_clean_data=False, *args, **kwargs):
    self.add_clean_data = add_clean_data
    if add_clean_data:
      self.delayed_seq_data_start_symbol = delayed_seq_data_start_symbol
      self._keys_to_read_extended = ["clean_data_delayed","data","classes"]
      self.MapToDataKeys = {"source": "data",
                   "target": "classes",
                   "clean_source": "clean_data_delayed"
                   }  # just by our convention
    else:
      self._keys_to_read_extended = ["data","classes"]
      self.MapToDataKeys = {"source": "data",
                   "target": "classes",
                   }  # just by our convention
    super(MaskedLmDataset, self).__init__(*args, **kwargs)
    filename = "%s/%s.%s.h5" % (self.path, 'positions', self.file_postfix)
    with h5py.File(self._transform_filename(filename), 'r', libver='latest') as positions_in:
      positions_arr = positions_in['array'][:]
    positons_list = []
    for line in positions_arr:
      positons_list.append(line[:line[-1]])
    self._data['positions'] = positons_list
    self.num_outputs['positions'] = [1, 1]

  def _thread_main(self):
    from Util import interrupt_main
    try:
      import better_exchook
      better_exchook.install()
      from Util import AsyncThreadRun
      # First iterate once over the data to get the data len as fast as possible.
      data_len = 0
      while True:
        ls = self._data_files[self._main_data_key].readlines(10 ** 4)
        data_len += len(ls)
        if not ls:
          break
      with self._lock:
        self._data_len = data_len
      self._data_files[self._main_data_key].seek(0, os.SEEK_SET)  # we will read it again below

      # Now, read and use the vocab for a compact representation in memory.
      keys_to_read = list(self._keys_to_read_extended)
      while True:
        for k in keys_to_read:
          data_strs = self._data_files[k].readlines(10 ** 6)
          if not data_strs:
            assert len(self._data[k]) == self._data_len
            keys_to_read.remove(k)
            continue
          assert len(self._data[k]) + len(data_strs) <= self._data_len
          self._extend_data(k, data_strs)
        if not keys_to_read:
          break
      for k, f in list(self._data_files.items()):
        f.close()
        self._data_files[k] = None

    except Exception:
      sys.excepthook(*sys.exc_info())
      interrupt_main()

  def _collect_single_seq(self, seq_idx):
    if seq_idx >= self._num_seqs:
      return None
    line_nr = self._seq_order[seq_idx]
    features = {}
    features['data'] = self._get_data(key=self._main_data_key, line_nr=line_nr)
    features['positions'] = self._get_data(key="positions", line_nr=line_nr)
    features['classes'] = self._get_data(key=self._main_classes_key, line_nr=line_nr)
    if self.add_clean_data:
      clean_data = self._get_data(key="clean_data_delayed", line_nr=line_nr)
      features['clean_data_delayed'] = numpy.concatenate(
        ([self._vocabs['data'][self.delayed_seq_data_start_symbol]], clean_data[:-1]))
      assert features['clean_data_delayed'].shape == clean_data.shape

    assert features is not None
    return DatasetSeq(
      seq_idx=seq_idx,
      seq_tag=self._tag_prefix + str(line_nr),
      features=features,
      targets=None)
