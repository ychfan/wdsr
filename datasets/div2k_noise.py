import os

import common.modes
import datasets._idn

LOCAL_DIR = 'data/DIV2K/'
TRAIN_HR_DIR = LOCAL_DIR + 'DIV2K_train_HR/'
EVAL_HR_DIR = LOCAL_DIR + 'DIV2K_valid_HR/'


def update_argparser(parser):
  datasets._idn.update_argparser(parser)
  parser.add_argument(
      '--input_dir', help='Directory of input files in predict mode.')
  parser.set_defaults(
      num_channels=3,
      num_patches=1000,
      train_batch_size=16,
      eval_batch_size=1,
  )


def get_dataset(mode, params):
  if mode == common.modes.PREDICT:
    return DIV2K_(mode, params)
  else:
    return DIV2K(mode, params)


class DIV2K(datasets._idn.ImageDenoisingHdf5Dataset):

  def __init__(self, mode, params):
    hr_cache_file = 'cache/div2k_{}_hr.h5'.format(mode)

    hr_dir = {
        common.modes.TRAIN: TRAIN_HR_DIR,
        common.modes.EVAL: EVAL_HR_DIR,
    }[mode]

    hr_files = list_image_files(hr_dir)

    super(DIV2K, self).__init__(
        mode,
        params,
        hr_files,
        hr_cache_file,
    )


class DIV2K_(datasets._idn.ImageDenoisingDataset):

  def __init__(self, mode, params):
    hr_dir = {
        common.modes.TRAIN: TRAIN_HR_DIR,
        common.modes.EVAL: EVAL_HR_DIR,
        common.modes.PREDICT: params.input_dir,
    }[mode]

    hr_files = list_image_files(hr_dir)

    super(DIV2K_, self).__init__(
        mode,
        params,
        hr_files,
    )


def list_image_files(d):
  files = sorted(os.listdir(d))
  files = [(f, os.path.join(d, f)) for f in files if f.endswith('.png')]
  return files
