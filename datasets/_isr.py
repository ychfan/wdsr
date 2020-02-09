"""Image Super-resolution dataset."""

import os
import random
import numpy as np
from PIL import Image
from skimage.io import imread
from skimage import img_as_float, img_as_ubyte

import torch.utils.data as data
import torchvision.transforms as transforms

import common.modes
import common.io
from common.images import imresize
import datasets


def update_argparser(parser):
  datasets.update_argparser(parser)
  parser.add_argument(
      '--scale',
      help='Scale factor for image super-resolution.',
      default=2,
      type=int)
  parser.add_argument(
      '--lr_patch_size',
      help='Number of pixels in height or width of LR patches.',
      default=48,
      type=int)
  parser.add_argument(
      '--ignored_boundary_size',
      help='Number of ignored boundary pixels of LR patches.',
      default=2,
      type=int)
  parser.add_argument(
      '--num_patches',
      help='Number of sampling patches per image for training.',
      default=100,
      type=int)
  parser.set_defaults(
      train_batch_size=16,
      eval_batch_size=1,
      image_mean=0.5,
  )


class ImageSuperResolutionDataset(data.Dataset):

  def __init__(self, mode, params, lr_files, hr_files):
    super(ImageSuperResolutionDataset, self).__init__()
    self.mode = mode
    self.params = params
    self.lr_files = lr_files
    self.hr_files = hr_files

  def __getitem__(self, index):
    if self.mode == common.modes.PREDICT:
      lr_image = np.asarray(Image.open(self.lr_files[index][1]))
      lr_image = transforms.functional.to_tensor(lr_image)
      return lr_image, self.hr_files[index][0]

    if self.mode == common.modes.TRAIN:
      index = index // self.params.num_patches

    lr_image, hr_image = self._load_item(index)

    # sample patch while training
    if self.mode == common.modes.TRAIN:
      # sample patch while training
      x = random.randrange(
          self.params.ignored_boundary_size, lr_image.shape[0] -
          self.params.lr_patch_size + 1 - self.params.ignored_boundary_size)
      y = random.randrange(
          self.params.ignored_boundary_size, lr_image.shape[1] -
          self.params.lr_patch_size + 1 - self.params.ignored_boundary_size)
      lr_image = lr_image[x:x + self.params.lr_patch_size, y:y +
                          self.params.lr_patch_size]
      hr_image = hr_image[x *
                          self.params.scale:(x + self.params.lr_patch_size) *
                          self.params.scale, y *
                          self.params.scale:(y + self.params.lr_patch_size) *
                          self.params.scale]
    else:
      hr_image = hr_image[:lr_image.shape[0] *
                          self.params.scale, :lr_image.shape[1] *
                          self.params.scale]

    # augmentation while training
    if self.mode == common.modes.TRAIN:
      if random.random() < 0.5:
        lr_image = lr_image[::-1]
        hr_image = hr_image[::-1]
      if random.random() < 0.5:
        lr_image = lr_image[:, ::-1]
        hr_image = hr_image[:, ::-1]
      if random.random() < 0.5:
        lr_image = np.swapaxes(lr_image, 0, 1)
        hr_image = np.swapaxes(hr_image, 0, 1)

    lr_image = np.ascontiguousarray(lr_image)
    hr_image = np.ascontiguousarray(hr_image)

    lr_image = transforms.functional.to_tensor(lr_image)
    hr_image = transforms.functional.to_tensor(hr_image)
    return lr_image, hr_image

  def _load_item(self, index):
    lr_image = np.asarray(Image.open(self.lr_files[index][1]))
    hr_image = np.asarray(Image.open(self.hr_files[index][1]))
    return lr_image, hr_image

  def __len__(self):
    if self.mode == common.modes.TRAIN:
      return len(self.lr_files) * self.params.num_patches
    else:
      return len(self.lr_files)


class ImageSuperResolutionHdf5Dataset(ImageSuperResolutionDataset):

  def __init__(
      self,
      mode,
      params,
      lr_files,
      hr_files,
      lr_cache_file,
      hr_cache_file,
      lib_hdf5='h5py',
  ):
    super(ImageSuperResolutionHdf5Dataset, self).__init__(
        mode,
        params,
        lr_files,
        hr_files,
    )
    self.lr_cache_file = common.io.Hdf5(lr_cache_file, lib_hdf5)
    self.hr_cache_file = common.io.Hdf5(hr_cache_file, lib_hdf5)

    cache_dir = os.path.dirname(lr_cache_file)
    if not os.path.exists(cache_dir):
      os.makedirs(cache_dir)

    if not os.path.exists(lr_cache_file):
      for lr_file in self.lr_files:
        self.lr_cache_file.add(lr_file[0], np.asarray(Image.open(lr_file[1])))
    if self.mode != common.modes.PREDICT:
      if not os.path.exists(hr_cache_file):
        for hr_file in self.hr_files:
          self.hr_cache_file.add(hr_file[0], np.asarray(Image.open(hr_file[1])))

  def _load_item(self, index):
    lr_image = self.lr_cache_file.get(self.lr_files[index][0])
    hr_image = self.hr_cache_file.get(self.hr_files[index][0])
    return lr_image, hr_image


class ImageSuperResolutionBicubicDataset(ImageSuperResolutionDataset):

  def __init__(self, mode, params, hr_files):
    super(ImageSuperResolutionBicubicDataset, self).__init__(
        mode,
        params,
        hr_files,
        hr_files,
    )

  def __getitem__(self, index):
    if self.mode == common.modes.PREDICT:
      hr_image = imread(self.lr_files[index][1])
      if hr_image.shape[0] % self.params.scale:
        hr_image = hr_image[:-(hr_image.shape[0] % self.params.scale), :]
      if hr_image.shape[1] % self.params.scale:
        hr_image = hr_image[:, :-(hr_image.shape[1] % self.params.scale)]
      lr_image = imresize(hr_image, scalar_scale=1 / self.params.scale)
      lr_image = np.asarray(lr_image)
      lr_image = transforms.functional.to_tensor(lr_image)
      return lr_image, self.hr_files[index][0]
    else:
      return super(ImageSuperResolutionBicubicDataset, self).__getitem__(index)

  def _load_item(self, index):
    hr_image = imread(self.hr_files[index][1])
    if hr_image.shape[0] % self.params.scale:
      hr_image = hr_image[:-(hr_image.shape[0] % self.params.scale), :]
    if hr_image.shape[1] % self.params.scale:
      hr_image = hr_image[:, :-(hr_image.shape[1] % self.params.scale)]
    lr_image = imresize(hr_image, scalar_scale=1 / self.params.scale)
    lr_image = np.asarray(lr_image)
    hr_image = np.asarray(hr_image)
    return lr_image, hr_image
