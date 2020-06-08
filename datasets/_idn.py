import numpy as np
from hashlib import sha256

import torch

import common.modes
import datasets._isr


def update_argparser(parser):
  datasets._isr.update_argparser(parser)
  parser.add_argument(
      '--noise_sigma',
      help='Noise sigma level for image denoising',
      default=25.0,
      type=float)
  parser.set_defaults(scale=1,)


class ImageDenoisingDataset(datasets._isr.ImageSuperResolutionDataset):

  def __init__(self, mode, params, image_files):
    super(ImageDenoisingDataset, self).__init__(mode, params, image_files,
                                                image_files)

  def __getitem__(self, index):
    x, y = super(ImageDenoisingDataset, self).__getitem__(index)
    if self.mode == common.modes.TRAIN:
      noise = torch.randn_like(x) * (self.params.noise_sigma / 255.0)
    else:
      image_file = self.lr_files[index][0]
      seed = np.frombuffer(
          sha256(image_file.encode('utf-8')).digest(), dtype='uint32')
      rstate = np.random.RandomState(seed)
      noise = rstate.normal(0, self.params.noise_sigma / 255.0, x.shape)
      noise = torch.from_numpy(noise).float()
    x += noise
    return x, y


class ImageDenoisingHdf5Dataset(datasets._isr.ImageSuperResolutionHdf5Dataset):

  def __init__(
      self,
      mode,
      params,
      image_files,
      image_cache_file,
      lib_hdf5='h5py',
  ):
    super(ImageDenoisingHdf5Dataset, self).__init__(
        mode,
        params,
        image_files,
        image_files,
        image_cache_file,
        image_cache_file,
        lib_hdf5=lib_hdf5,
    )

  def __getitem__(self, index):
    x, y = super(ImageDenoisingHdf5Dataset, self).__getitem__(index)
    if self.mode == common.modes.TRAIN:
      noise = torch.randn_like(x) * (self.params.noise_sigma / 255.0)
    else:
      image_file = self.lr_files[index][0]
      seed = np.frombuffer(
          sha256(image_file.encode('utf-8')).digest(), dtype='uint32')
      rstate = np.random.RandomState(seed)
      noise = rstate.normal(0, self.params.noise_sigma / 255.0, x.shape)
      noise = torch.from_numpy(noise).float()
    return x + noise, y
