import os

import common.modes
import datasets._isr

LOCAL_DIR = 'data/Manga109/'


def update_argparser(parser):
  datasets._isr.update_argparser(parser)
  parser.add_argument(
      '--input_dir', help='Directory of input files in predict mode.')
  parser.set_defaults(
      num_channels=3,
      eval_batch_size=1,
  )


def get_dataset(mode, params):
  if mode == common.modes.EVAL:
    return Set_(mode, params)
  else:
    raise NotImplementedError


class Set_(datasets._isr.ImageSuperResolutionBicubicDataset):

  def __init__(self, mode, params):
    hr_dir = {
        common.modes.TRAIN: LOCAL_DIR,
        common.modes.EVAL: LOCAL_DIR,
        common.modes.PREDICT: params.input_dir,
    }[mode]

    hr_files = list_image_files(hr_dir)

    super(Set_, self).__init__(
        mode,
        params,
        hr_files,
    )


def list_image_files(d):
  files = sorted(os.listdir(d))
  files = [(f, os.path.join(d, f)) for f in files if f.endswith('.png')]
  return files
