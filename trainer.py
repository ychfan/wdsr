"""Trainer
"""
import argparse
import importlib
import logging
import os

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from apex.parallel import DistributedDataParallel, convert_syncbn_model
from apex import amp

import common.meters
import common.modes

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--dataset',
      help='Dataset name.',
      default=None,
      type=str,
      required=True,
  )
  parser.add_argument(
      '--model',
      help='Model name.',
      default=None,
      type=str,
      required=True,
  )
  parser.add_argument(
      '--job_dir',
      help='Directory to write checkpoints and export models.',
      default=None,
      type=str,
      required=True,
  )
  parser.add_argument(
      '--ckpt',
      help='File path to load checkpoint.',
      default=None,
      type=str,
  )
  parser.add_argument(
      '--override_epoch',
      help='Override epoch number when loading from checkpoint.',
      default=None,
      type=int,
  )
  parser.add_argument(
      '--eval_only',
      default=False,
      action='store_true',
      help='Running evaluation only.',
  )
  parser.add_argument(
      '--eval_datasets',
      help='Dataset names for evaluation.',
      default=None,
      type=str,
      nargs='+',
  )
  # Experiment arguments
  parser.add_argument(
      '--save_checkpoints_epochs',
      help='Number of epochs to save checkpoint.',
      default=1,
      type=int)
  parser.add_argument(
      '--keep_checkpoints',
      help='Keepining intermediate checkpoints.',
      default=False,
      action='store_true')
  parser.add_argument(
      '--train_epochs',
      help='Number of epochs to run training totally.',
      default=10,
      type=int)
  parser.add_argument(
      '--log_steps',
      help='Number of steps for training logging.',
      default=100,
      type=int)
  parser.add_argument(
      '--random_seed',
      help='Random seed for TensorFlow.',
      default=None,
      type=int)
  # Performance tuning parameters
  parser.add_argument(
      '--opt_level',
      help='Number of GPUs for experiments.',
      default='O0',
      type=str)
  parser.add_argument(
      '--sync_bn',
      default=False,
      action='store_true',
      help='Enabling apex sync BN.')
  # Verbose
  parser.add_argument(
      '-v',
      '--verbose',
      action='count',
      default=0,
      help='Increasing output verbosity.',
  )
  parser.add_argument('--local_rank', default=0, type=int)
  parser.add_argument('--node_rank', default=0, type=int)

  # Parse arguments
  args, _ = parser.parse_known_args()
  logging.basicConfig(
      level=[logging.WARNING, logging.INFO, logging.DEBUG][args.verbose],
      format='%(asctime)s:%(levelname)s:%(message)s')
  dataset_module = importlib.import_module(
      'datasets.' + args.dataset if args.dataset else 'datasets')
  dataset_module.update_argparser(parser)
  model_module = importlib.import_module('models.' +
                                         args.model if args.model else 'models')
  model_module.update_argparser(parser)
  params = parser.parse_args()
  logging.critical(params)

  torch.backends.cudnn.benchmark = True

  params.distributed = False
  params.master_proc = True
  if 'WORLD_SIZE' in os.environ:
    params.distributed = int(os.environ['WORLD_SIZE']) > 1
  if params.distributed:
    torch.cuda.set_device(params.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    if params.node_rank or params.local_rank:
      params.master_proc = False

  train_dataset = dataset_module.get_dataset(common.modes.TRAIN, params)
  if params.eval_datasets:
    eval_datasets = []
    for eval_dataset in params.eval_datasets:
      eval_dataset_module = importlib.import_module('datasets.' + eval_dataset)
      eval_datasets.append(
          (eval_dataset,
           eval_dataset_module.get_dataset(common.modes.EVAL, params)))
  else:
    eval_datasets = [(params.dataset,
                      dataset_module.get_dataset(common.modes.EVAL, params))]
  if params.distributed:
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset)
    eval_sampler = None
    params.train_batch_size //= torch.cuda.device_count()
  else:
    train_sampler = None
    eval_sampler = None
  train_data_loader = DataLoader(
      dataset=train_dataset,
      num_workers=params.num_data_threads,
      batch_size=params.train_batch_size,
      shuffle=(train_sampler is None),
      drop_last=True,
      pin_memory=True,
      sampler=train_sampler,
  )
  eval_data_loaders = [(data_name,
                        DataLoader(
                            dataset=dataset,
                            num_workers=params.num_data_threads,
                            batch_size=params.eval_batch_size,
                            shuffle=False,
                            drop_last=False,
                            pin_memory=True,
                            sampler=eval_sampler,
                        )) for data_name, dataset in eval_datasets]
  model, criterion, optimizer, lr_scheduler, metrics = model_module.get_model_spec(
      params)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = model.to(device)
  criterion = criterion.to(device)

  model, optimizer = amp.initialize(
      model, optimizer, opt_level=params.opt_level, verbosity=params.verbose)

  if params.ckpt or os.path.exists(os.path.join(params.job_dir, 'latest.pth')):
    checkpoint = torch.load(
        params.ckpt or os.path.join(params.job_dir, 'latest.pth'),
        map_location=lambda storage, loc: storage.cuda())
    try:
      model.load_state_dict(checkpoint['model_state_dict'])
      optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
      lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
    except RuntimeError as e:
      logging.critical(e)
    latest_epoch = checkpoint['epoch']
    logging.critical('Loaded checkpoint from epoch {}.'.format(latest_epoch))
    if params.override_epoch is not None:
      latest_epoch = params.override_epoch
      logging.critical('Overrode epoch number to {}.'.format(latest_epoch))
  else:
    latest_epoch = 0

  if params.distributed:
    if params.sync_bn:
      model = convert_syncbn_model(model)
    model = DistributedDataParallel(model)

  def train(epoch):
    if params.distributed:
      train_sampler.set_epoch(epoch)
    loss_meter = common.meters.AverageMeter()
    time_meter = common.meters.TimeMeter()
    model.train()
    for batch_idx, (data, target) in enumerate(train_data_loader):
      data = data.to(device, non_blocking=True)
      target = target.to(device, non_blocking=True)
      optimizer.zero_grad()
      output = model(data)
      loss = criterion(output, target)
      with amp.scale_loss(loss, optimizer) as scaled_loss:
        scaled_loss.backward()
      optimizer.step()
      if batch_idx % params.log_steps == 0:
        loss_meter.update(loss.item(), data.size(0))
        time_meter.update_count(batch_idx + 1)
        logging.info(
            'Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tSpeed: {:.6f} seconds/batch'
            .format(epoch, batch_idx * len(data),
                    len(train_data_loader) * len(data),
                    100. * batch_idx / len(train_data_loader), loss.item(),
                    time_meter.avg))
    logging.critical('Train epoch {} finished.\tLoss: {:.6f}'.format(
        epoch, loss_meter.avg))
    if params.master_proc:
      writer.add_scalar('training_loss', loss_meter.avg, epoch)

  def evaluate(epoch):
    with torch.no_grad():
      model.eval()
      for eval_data_name, eval_data_loader in eval_data_loaders:
        metric_meters = {}
        for metric_name in metrics.keys():
          metric_meters[metric_name] = common.meters.AverageMeter()
        time_meter = common.meters.TimeMeter()
        for data, target in eval_data_loader:
          data = data.to(device, non_blocking=True)
          target = target.to(device, non_blocking=True)
          output = model(data)
          for metric_name in metrics.keys():
            metric_meters[metric_name].update(
                metrics[metric_name](output, target).item(), data.size(0))
          time_meter.update(data.size(0))
        for metric_name in metrics.keys():
          logging.critical('Eval {}: Average {}: {:.4f}'.format(
              eval_data_name, metric_name, metric_meters[metric_name].avg))
          if epoch is not None:
            writer.add_scalar(eval_data_name + '/' + metric_name,
                              metric_meters[metric_name].avg, epoch)
        logging.critical('Eval {}: Average Speed: {:.6f} seconds/sample'.format(
            eval_data_name, time_meter.avg))

  if params.eval_only:
    evaluate(None)
    exit()

  if params.master_proc:
    writer = SummaryWriter(params.job_dir)

  for epoch in range(latest_epoch + 1, params.train_epochs + 1):
    train(epoch)
    lr_scheduler.step()
    if epoch % params.save_checkpoints_epochs == 0:
      if params.master_proc:
        if not os.path.exists(params.job_dir):
          os.makedirs(params.job_dir)
        torch.save(
            {
                'model_state_dict':
                    model.module.state_dict()
                    if params.distributed else model.state_dict(),
                'optimizer_state_dict':
                    optimizer.state_dict(),
                'lr_scheduler_state_dict':
                    lr_scheduler.state_dict(),
                'epoch':
                    epoch,
            }, os.path.join(params.job_dir, 'epoch_{}.pth'.format(epoch)))
        latest_path = os.path.join(params.job_dir, 'latest.pth')
        if os.path.exists(latest_path):
          if not params.keep_checkpoints:
            os.remove(os.path.join(params.job_dir, os.readlink(latest_path)))
          os.remove(latest_path)
        os.symlink('epoch_{}.pth'.format(epoch), latest_path)
        evaluate(epoch)

  if params.master_proc:
    writer.close()
