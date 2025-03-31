import os
import sys
import json
from time import time
from datetime import datetime, timezone
from matplotlib import pyplot as plt
import numpy as np
import torch
import data.augmentor
import data.dataset
import data.utils
import data.augmentor
import engine.optimizer
import engine.trainer
import interpretation.visualization
import model.backbone
import model.head
import model.tmodel
import utils.storage
import utils.logger


def train(basis_chkp_id: str = None, basis_cfg_id: str = None):
  '''
  Parameters:
    `basis_chkp_id`. Checkpoint identifier to be parsed.
    `basis_cfg_id`. Template identifier to be parsed.
  '''

  assert bool(basis_chkp_id)^bool(basis_cfg_id), 'E: Exactly one of chkp_id and tplt_id must be specified.'

  if basis_chkp_id is not(None): # and tplt_id is None
    basis_id = basis_chkp_id
  else: # chkp_id is None and tplt_id is not None
    basis_id = basis_cfg_id

  ## Training Initialization

  acceptable_iter_optim_algs = {'adam': torch.optim.Adam}
  session_init_datetime = datetime.now().astimezone(timezone.utc)
  device = torch.device(device='cuda:0' if torch.cuda.is_available() else 'cpu')
  current_session_chkp_id = session_init_datetime.strftime('D%Y%m%d%H%M%SUTC0')

  # Parse checkpoint information or some-config.json and then initialize the current session's checkpoint
  session_checkpoint_manager = utils.storage.CheckpointStorageManager(chkp_id=current_session_chkp_id, basis_id=basis_id, session_datetime=session_init_datetime)

  # Prepare configuration parameters
  rng_seed = session_checkpoint_manager.config.content['training_cfg']['rng_seed']
  model_id = session_checkpoint_manager.config.content['model_architecture_cfg']['type']
  data_id = session_checkpoint_manager.config.content['training_cfg']['data']
  instance_prsd_shape = session_checkpoint_manager.config.content['model_architecture_cfg']['instance_prsd_shape']
  max_epochs = session_checkpoint_manager.config.content['training_cfg']['max_epochs']
  M_minibatch = session_checkpoint_manager.config.content['training_cfg']['M_minibatch']
  iter_optim_alg_name = session_checkpoint_manager.config.content['training_cfg']['optimization']['type']
  lr = session_checkpoint_manager.config.content['training_cfg']['optimization']['lr']
  train_fraction = session_checkpoint_manager.config.content['training_cfg']['train_fraction']
  subset_size = session_checkpoint_manager.config.content['training_cfg']['subset_size']
  tparams = session_checkpoint_manager.tparams.content

  # Prepare training history
  epoch_offset = len(session_checkpoint_manager.training_history.content['metrics_measurements']['train']['loss'])
  metrics_ids = list(session_checkpoint_manager.training_history.content['metrics_measurements']['train'].keys())

  if 'undoCheckpoint' in os.environ['DEBUG_CONFIG'].split(';'):
    session_checkpoint_manager.rm_chkp()
  if 'limit2SmallSubsetofData' in os.environ['DEBUG_CONFIG'].split(';'):
    subset_size = 100

  assert iter_optim_alg_name in acceptable_iter_optim_algs.keys(), 'E: Invalid iterative optimization algorithm.'
  iter_optim_alg = acceptable_iter_optim_algs[iter_optim_alg_name]

  assert epoch_offset < max_epochs, 'E: Epoch offset cannot be bigger or equal than the maximum number of epochs.'

  np.random.seed(seed=rng_seed)

  # Assemble complete trainable architecture
  trainable_model = model.tmodel.trainable_model_assembler(model_architecture_cfg=session_checkpoint_manager.config.content['model_architecture_cfg'], device=device, state_dict=tparams)

  # Training algorithm configuration
  contrastive_loss = engine.optimizer.contrastive_loss
  optimizer = iter_optim_alg(params=trainable_model.parameters(), lr=lr)

  # Data parsing
  dataset = data.dataset.Cifar(instance_prsd_shape=instance_prsd_shape, M_minibatch=M_minibatch, train_fraction=train_fraction, subset_size=subset_size, parse_labels=True, device=device)

  opt_measurements = dict()
  for measurement_mode in {'train', 'val'}:
    opt_measurements[measurement_mode] = dict()
    for metric_id in metrics_ids:
      opt_measurements[measurement_mode][metric_id] = -10**10

  # stdout: print shapes, sizes, notify training is about to start
  print('\n[TRAINING SUMMARY]')
  print('Dataset: %s'%(data_id))
  print('Train data: %d'%(len(dataset.train_set)))
  print('Val data: %d'%(len(dataset.val_set)))
  print('Test data: %d'%(len(dataset.test_set)))
  print('Total data: %d'%(len(dataset)))
  print('Initial epoch: %d'%(epoch_offset))
  print('\n[COMMENCING THE TRAINING PROCESS]')

  t_0_training = time()

  ## The training process

  for epoch in range(epoch_offset, max_epochs):

    # Resets measurement lists
    metrics_measurement_lists = dict()
    for measurement_mode in {'est_train', 'train', 'val'}:
      metrics_measurement_lists[measurement_mode] = dict()
      for metric_id in metrics_ids:
        metrics_measurement_lists[measurement_mode][metric_id] = []

    t_0_epoch = time()
    t_0_datetime_str = datetime.now().astimezone(timezone.utc).strftime('D%Y%m%d%H%M%SUTC0')

    print('\n[EPOCH %d/%d @ %s]'%(epoch, max_epochs-1, t_0_datetime_str))

    print('Model training state:')

    while next(dataset.train_set):

      t_0_iteration = time()

      # Get augmented pairs from current minibatch
      train_set_ist_minibatch_pairs = dataset.train_set.x_current_grp_agm_pairs

      # Forward pass
      train_set_ist_minibatch_pairs_descriptor = trainable_model(train_set_ist_minibatch_pairs)
      train_loss_minibatch = contrastive_loss(descriptor_pairs=train_set_ist_minibatch_pairs_descriptor, temperature=.1, device=device)

      # Reset Jacobian tensors (useful only for loops)
      optimizer.zero_grad()

      # Jacobian tensors computation
      train_loss_minibatch.backward()

      # Update trainable model
      optimizer.step()

      delta_t_iteration = time() - t_0_iteration

      # stdout: Iteration state
      progress_bar_ist = utils.logger.get_progress_bar_ist(i=dataset.train_set.minibatch_idx, n=dataset.train_set.n_steps-1, delta_t=delta_t_iteration, bar_length = 30, bar_background_char = ' ', bar_fill_char = '=')
      utils.logger.dyn_stdout_write(s=progress_bar_ist)

      for metric_id in metrics_ids:
        metrics_measurement_lists['est_train'][metric_id].append(train_loss_minibatch.item())

    print('Train set evaluation state:')

    t_0_performance_measurement_period = time()

    # Train performance measurement
    while next(dataset.train_set):

      t_0_iteration = time()

      # Get augmented pairs from current minibatch
      train_set_ist_minibatch_pairs = dataset.train_set.x_current_grp_agm_pairs

      # Forward pass
      train_set_ist_minibatch_pairs_descriptor = trainable_model(train_set_ist_minibatch_pairs)
      train_loss_minibatch = contrastive_loss(descriptor_pairs=train_set_ist_minibatch_pairs_descriptor, temperature=.1, device=device)

      delta_t_iteration = time() - t_0_iteration

      # stdout: Iteration state
      progress_bar_ist = utils.logger.get_progress_bar_ist(i=dataset.train_set.minibatch_idx, n=dataset.train_set.n_steps-1, delta_t=delta_t_iteration, bar_length = 30, bar_background_char = ' ', bar_fill_char = '=')
      utils.logger.dyn_stdout_write(s=progress_bar_ist)

      # Performance metrics
      for metric_id in metrics_ids:
        metrics_measurement_lists['train'][metric_id].append(train_loss_minibatch.item())

    print('Validation set evaluation state:')

    # Val performance measurement
    while next(dataset.val_set):

      # Get augmented pairs from current minibatch
      val_set_ist_minibatch_pairs = dataset.val_set.x_current_grp_agm_pairs

      # Forward pass
      val_set_ist_minibatch_pairs_descriptor = trainable_model(val_set_ist_minibatch_pairs)
      val_loss_minibatch = contrastive_loss(descriptor_pairs=val_set_ist_minibatch_pairs_descriptor, temperature=.1, device=device)

      delta_t_iteration = time() - t_0_iteration

      # stdout: Iteration state
      progress_bar_ist = utils.logger.get_progress_bar_ist(i=dataset.val_set.minibatch_idx, n=dataset.val_set.n_steps-1, delta_t=delta_t_iteration, bar_length = 30, bar_background_char = ' ', bar_fill_char = '=')
      utils.logger.dyn_stdout_write(s=progress_bar_ist)

      # Performance metrics
      for metric_id in metrics_ids:
        metrics_measurement_lists['val'][metric_id].append(val_loss_minibatch.item())

    delta_t_performance_measurement_period = time() - t_0_performance_measurement_period
    t_0_performance_measurement_period_h = utils.logger.get_delta_t_h(t=delta_t_performance_measurement_period)
    delta_t_epoch = time() - t_0_epoch
    delta_t_epoch_h = utils.logger.get_delta_t_h(t=delta_t_epoch)

    # Measurement compounding and file recording
    for metric_id in metrics_ids:
      session_checkpoint_manager.training_history.content['metrics_measurements']['train'][metric_id].append\
      (
        [
          min(metrics_measurement_lists['est_train'][metric_id]),
          max(metrics_measurement_lists['est_train'][metric_id]),
          sum(metrics_measurement_lists['est_train'][metric_id])/len(metrics_measurement_lists['est_train'][metric_id]),
          sum(metrics_measurement_lists['train'][metric_id])/len(metrics_measurement_lists['train'][metric_id])
        ]
      )
      session_checkpoint_manager.training_history.content['metrics_measurements']['val'][metric_id].append\
      (
        [
          sum(metrics_measurement_lists['val'][metric_id])/len(metrics_measurement_lists['val'][metric_id])
        ]
      )
    session_checkpoint_manager.training_history.content['datetime_ended'] = datetime.now().astimezone(timezone.utc).strftime('D%Y%m%d%H%M%SUTC0')
    session_checkpoint_manager.training_history.content['total_time'] += delta_t_epoch

    # Write training_history
    session_checkpoint_manager.training_history.write()

    # If the current loss is optimal
    for measurement_mode in {'train', 'val'}:
      if opt_measurements[measurement_mode]['loss'] < session_checkpoint_manager.training_history.content['metrics_measurements'][measurement_mode]['loss'][-1][-1]:
        # Record all metrics for the same epoch
        for metric_id in metrics_ids:
          opt_measurements[measurement_mode][metric_id] = session_checkpoint_manager.training_history.content['metrics_measurements'][measurement_mode]['loss'][-1][-1]
        # Record and write tparams only in case this is optimal with respect to the validation set
        if measurement_mode == 'val':
          session_checkpoint_manager.opt_tparams.content = trainable_model.state_dict()
          session_checkpoint_manager.opt_tparams.write()

    # Write last model
    session_checkpoint_manager.tparams.content = trainable_model.state_dict()
    session_checkpoint_manager.tparams.write()

    # stdout: Epoch state
    print('Epoch Report')
    print('Epoch time: %s'%(delta_t_epoch_h))
    print('Performance measurement time: %s'%(t_0_performance_measurement_period_h))



    # Update session checkpoint
    # |-- Update history file
    # |   |-- Update datetime_ended
    # |   |-- Update total_time
    # |   |-- Update metrics_measurements -> train -> * -> (est_min, est_max, est_avg, avg)
    # |   |-- Update metrics_measurements -> val -> * -> avg
    # |
    # |-- Update pt file
    # |-- Update performance images
    # |-- DONT Update config.json
    # |-- Update log

  # stdout: Training completion
  # |-- display training time
  # |-- display test set performance

  delta_t = time() - t_0_training

  print('Training completed.')
  print('\n[TRAINING CONCLUSIVE SUMMARY]')
  print('Exported Checkpoint:\n%s'%(current_session_chkp_id))