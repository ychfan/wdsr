"""Metrics."""


def psnr(sr, hr, shave=4):
  sr = sr.to(hr.dtype)
  sr = (sr * 255).round().clamp(0, 255) / 255
  diff = sr - hr
  valid = diff[..., shave:-shave, shave:-shave]
  mse = valid.pow(2).mean([-3, -2, -1])
  psnr = -10 * mse.log10()
  return psnr.mean()