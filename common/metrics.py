"""Metrics."""

import torch
import torch.nn.functional as F


def psnr(sr, hr, shave=4):
  sr = sr.to(hr.dtype)
  sr = (sr * 255).round().clamp(0, 255) / 255
  diff = sr - hr
  valid = diff[..., shave:-shave, shave:-shave]
  mse = valid.pow(2).mean([-3, -2, -1])
  psnr = -10 * mse.log10()
  return psnr.mean()


def psnr_y(sr, hr, shave=4):
  sr = sr.to(hr.dtype)
  sr = (sr * 255).round().clamp(0, 255) / 255
  diff = sr - hr
  filters = torch.tensor([0.257, 0.504, 0.098],
                         dtype=diff.dtype,
                         device=diff.device)
  diff = F.conv2d(diff, filters.view([1, -1, 1, 1]))
  valid = diff[..., shave:-shave, shave:-shave]
  mse = valid.pow(2).mean([-3, -2, -1])
  psnr = -10 * mse.log10()
  return psnr.mean()
