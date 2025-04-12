import os
from time import time
from datetime import datetime, timezone
import numpy as np
import torch
import data.dataset
import engine.optimizer
import interpretation.visualizer
import model.tmodel
import utils.storage
import utils.logger
import pandas as pd
import copy
from tabulate import tabulate


def train(*, basis_chkp_id: str = None, basis_cfg_fp: str = None, device_id: str = None):
  '''
  Description:
    Training API.

  Parameters:
    `basis_chkp_id`. Checkpoint identifier to be parsed.
    `basis_cfg_fp`. Template configuration file path to be parsed.
    `device_id`. Device identifier to be selected for DL framework.
  '''

  assert bool(basis_chkp_id)^bool(basis_cfg_fp), 'E: Exactly one of chkp_id and tplt_id must be specified.'

  if basis_chkp_id is not(None):
    basis_path = basis_chkp_id
  else:
    basis_path = basis_cfg_fp

  ## Training Initialization

  dataset_selector = {'cifar': data.dataset.Cifar}
  acceptable_iter_optim_algs = {'adam': torch.optim.Adam}
  session_init_datetime = datetime.now().astimezone(timezone.utc)

  if device_id == 'gpu':
    device = torch.device(device='cuda:0' if torch.cuda.is_available() else 'cpu')
  else:
    device = torch.device(device='cpu')

  current_session_chkp_id = session_init_datetime.strftime('D%Y%m%d%H%M%SUTC0')

  # Parse checkpoint information or some-config.json and then initialize the current session's checkpoint
  session_checkpoint_manager = utils.storage.CheckpointStorageManager(chkp_id=current_session_chkp_id, basis_path=basis_path, session_datetime=session_init_datetime)

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

  # Prepare training history
  epoch_offset = len(session_checkpoint_manager.training_history.content['metrics_measurements']['train']['loss'])
  metrics_ids = session_checkpoint_manager.metrics_ids

  if 'limit2SmallSubsetofData' in os.environ['DEBUG_CONFIG'].split(';'):
    subset_size = 100

  assert iter_optim_alg_name in acceptable_iter_optim_algs.keys(), 'E: Invalid iterative optimization algorithm.'
  iter_optim_alg = acceptable_iter_optim_algs[iter_optim_alg_name]

  assert epoch_offset < max_epochs, 'E: Epoch offset cannot be bigger or equal than the maximum number of epochs.'

  np.random.seed(seed=rng_seed)

  # Assemble complete trainable architecture
  trainable_model = model.tmodel.trainable_model_assembler(model_architecture_cfg=session_checkpoint_manager.config.content['model_architecture_cfg'], device=device, state_dict=session_checkpoint_manager.tparams.content)

  # Training algorithm configuration
  contrastive_loss = engine.optimizer.contrastive_loss
  optimizer = iter_optim_alg(params=trainable_model.parameters(), lr=lr)

  # Data parsing
  dataset = dataset_selector[data_id](instance_prsd_shape=instance_prsd_shape, M_minibatch=M_minibatch, train_fraction=train_fraction, subset_size=subset_size, parse_labels=True, device=device)

  init_measurements = dict()
  opt_measurements = dict()
  worst_measurements = dict()
  for measurement_mode in {'train', 'val'}:
    init_measurements[measurement_mode] = dict()
    opt_measurements[measurement_mode] = dict()
    worst_measurements[measurement_mode] = dict()
    for metric_id in metrics_ids:
      if metric_id == 'loss':
        init_measurements[measurement_mode][metric_id] = None
        opt_measurements[measurement_mode][metric_id] = float('inf')
        worst_measurements[measurement_mode][metric_id] = -float('inf')
      else:
        init_measurements[measurement_mode][metric_id] = None
        opt_measurements[measurement_mode][metric_id] = None
        worst_measurements[measurement_mode][metric_id] = None

  plt2d = dict()
  inv_metrics_measurements = session_checkpoint_manager.get_inverted_dataset_metric()
  for metric_id in metrics_ids:
    plt2d[metric_id] = interpretation.visualizer.Plot2D(metric_id=metric_id, y=inv_metrics_measurements[metric_id])
    session_checkpoint_manager.loss_history_plot[metric_id].content = plt2d[metric_id].plt2image()
    session_checkpoint_manager.loss_history_plot[metric_id].write()
  del inv_metrics_measurements

  # stdout: print shapes, sizes, notify training is about to start
  print('\n[TRAINING SUMMARY]\n')
  print('Model: %s'%(model_id))
  print('Dataset: %s'%(data_id))
  print('Train data: %d'%(len(dataset.train_set)))
  print('Val data: %d'%(len(dataset.val_set)))
  print('Test data: %d'%(len(dataset.test_set)))
  print('Total data: %d'%(len(dataset)))
  print('Initial epoch: %d'%(epoch_offset))

  print('\n[PRETRAINING EVALUATION]\n')

  t_0 = time()
  compounding_delta_t_stdout = 0

  initial_val_loss = []

  # Val performance measurement
  while next(dataset.val_set):

    # Get augmented pairs from current minibatch
    val_set_ist_minibatch_pairs = dataset.val_set.x_current_grp_agm_pairs

    # Forward pass
    val_set_ist_minibatch_pairs_descriptor = trainable_model(val_set_ist_minibatch_pairs)
    val_loss_minibatch = contrastive_loss(descriptor_pairs=val_set_ist_minibatch_pairs_descriptor, temperature=.1, device=device)

    delta_t_iteration = (time()-t_0) / (dataset.val_set.minibatch_idx+1)

    initial_val_loss.append(val_loss_minibatch.item())

    # stdout: Iteration state
    compounding_delta_t_stdout += delta_t_iteration
    if dataset.val_set.minibatch_idx == 0 or dataset.val_set.minibatch_idx == dataset.val_set.n_steps - 1 or compounding_delta_t_stdout > 0.1:
      compounding_delta_t_stdout = 0
      progress_bar_ist = utils.logger.get_progress_bar_ist(i=dataset.val_set.minibatch_idx, n=dataset.val_set.n_steps-1, delta_t=delta_t_iteration, loss=None, bar_length = 30, bar_background_char = ' ', bar_fill_char = '=')
      utils.logger.dnmc_stdout_write(s=progress_bar_ist)

  # Record loss
  measurement = sum(initial_val_loss) / len(initial_val_loss)
  init_measurements['val']['loss'] = measurement
  opt_measurements['val']['loss'] = measurement
  worst_measurements['val']['loss'] = measurement

  if len(session_checkpoint_manager.training_history.content['metrics_measurements']['val']['loss']) != 0:

    val_losses = copy.deepcopy(session_checkpoint_manager.training_history.content['metrics_measurements']['val']['loss'])

    historical_min_val_loss = min([val_losses[epoch][0] for epoch in range(len(val_losses))])
    opt_measurements['val']['loss'] = min([historical_min_val_loss, init_measurements['val']['loss']])

    # If historical_min_val_loss is bigger than init_measurements['val']['loss'], then the reason the optimal model is not stored from this point is because *the model is almost certainly not worth saving*.

    del val_losses

  print('\n[COMMENCING THE TRAINING PROCESS]')

  ## The training process

  for epoch in range(epoch_offset, max_epochs):

    # Resets measurement lists
    metrics_measurement_lists = dict()
    for measurement_mode in {'est_train', 'train', 'val'}:
      metrics_measurement_lists[measurement_mode] = dict()
      for metric_id in metrics_ids:
        metrics_measurement_lists[measurement_mode][metric_id] = []

    t_0_epoch = time()
    t_0_epoch_datetime_str = datetime.now().astimezone(timezone.utc).strftime('D%Y%m%d%H%M%SUTC0')
    t_1 = time()
    compounding_delta_t_stdout = 0

    print('\n[EPOCH %d/%d @ %s]\n'%(epoch, max_epochs-1, t_0_epoch_datetime_str))

    print('Iterative optimization state:')

    # Optimization loop
    while next(dataset.train_set):

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

      metrics_measurement_lists['est_train']['loss'].append(train_loss_minibatch.item())

      delta_t_iteration = (time()-t_1) / (dataset.train_set.minibatch_idx+1)

      # stdout: Iteration state
      compounding_delta_t_stdout += delta_t_iteration
      if dataset.train_set.minibatch_idx == 0 or dataset.train_set.minibatch_idx == dataset.train_set.n_steps - 1 or compounding_delta_t_stdout > 0.1:
        compounding_delta_t_stdout = 0
        progress_bar_ist = utils.logger.get_progress_bar_ist(i=dataset.train_set.minibatch_idx, n=dataset.train_set.n_steps-1, delta_t=delta_t_iteration, loss=metrics_measurement_lists['est_train']['loss'][-1], bar_length = 30, bar_background_char = ' ', bar_fill_char = '=')
        utils.logger.dnmc_stdout_write(s=progress_bar_ist)

    print('Epoch optimization completed; proceeding to model evaluation stage.')
    print('Train set evaluation state:')

    t_0_performance_measurement_period = time()
    t_2 = time()
    compounding_delta_t_stdout = 0

    # Train performance measurement
    while next(dataset.train_set):

      # Get augmented pairs from current minibatch
      train_set_ist_minibatch_pairs = dataset.train_set.x_current_grp_agm_pairs

      # Forward pass
      train_set_ist_minibatch_pairs_descriptor = trainable_model(train_set_ist_minibatch_pairs)
      train_loss_minibatch = contrastive_loss(descriptor_pairs=train_set_ist_minibatch_pairs_descriptor, temperature=.1, device=device)

      delta_t_iteration = (time()-t_2) / (dataset.train_set.minibatch_idx+1)

      # Performance metrics
      metrics_measurement_lists['train']['loss'].append(train_loss_minibatch.item())

      # stdout: Iteration state
      compounding_delta_t_stdout += delta_t_iteration
      if dataset.train_set.minibatch_idx == 0 or dataset.train_set.minibatch_idx == dataset.train_set.n_steps - 1 or compounding_delta_t_stdout > 0.1:
        compounding_delta_t_stdout = 0
        progress_bar_ist = utils.logger.get_progress_bar_ist(i=dataset.train_set.minibatch_idx, n=dataset.train_set.n_steps-1, delta_t=delta_t_iteration, loss=metrics_measurement_lists['train']['loss'][-1], bar_length = 30, bar_background_char = ' ', bar_fill_char = '=')
        utils.logger.dnmc_stdout_write(s=progress_bar_ist)

    print('Validation set evaluation state:')

    t_3 = time()
    compounding_delta_t_stdout = 0

    # Val performance measurement
    while next(dataset.val_set):

      # Get augmented pairs from current minibatch
      val_set_ist_minibatch_pairs = dataset.val_set.x_current_grp_agm_pairs

      # Forward pass
      val_set_ist_minibatch_pairs_descriptor = trainable_model(val_set_ist_minibatch_pairs)
      val_loss_minibatch = contrastive_loss(descriptor_pairs=val_set_ist_minibatch_pairs_descriptor, temperature=.1, device=device)

      delta_t_iteration = (time()-t_3) / (dataset.val_set.minibatch_idx+1)

      # Performance metrics
      metrics_measurement_lists['val']['loss'].append(val_loss_minibatch.item())

      # stdout: Iteration state
      compounding_delta_t_stdout += delta_t_iteration
      if dataset.val_set.minibatch_idx == 0 or dataset.val_set.minibatch_idx == dataset.val_set.n_steps - 1 or compounding_delta_t_stdout > 0.1:
        compounding_delta_t_stdout = 0
        progress_bar_ist = utils.logger.get_progress_bar_ist(i=dataset.val_set.minibatch_idx, n=dataset.val_set.n_steps-1, delta_t=delta_t_iteration, loss=metrics_measurement_lists['val']['loss'][-1], bar_length = 30, bar_background_char = ' ', bar_fill_char = '=')
        utils.logger.dnmc_stdout_write(s=progress_bar_ist)

    delta_t_performance_measurement_period = time() - t_0_performance_measurement_period
    t_0_performance_measurement_period_h = utils.logger.get_delta_t_h(t=delta_t_performance_measurement_period)

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

    # If the current loss is optimal
    if session_checkpoint_manager.training_history.content['metrics_measurements']['val']['loss'][-1][-1] < opt_measurements['val']['loss']:

      # Record all metrics for the same epoch
      for measurement_mode in {'train', 'val'}:
        for metric_id in metrics_ids:
          opt_measurements[measurement_mode][metric_id] = session_checkpoint_manager.training_history.content['metrics_measurements'][measurement_mode][metric_id][-1][-1]

      # Record and store tparams
      session_checkpoint_manager.opt_tparams.content = trainable_model.state_dict()
      session_checkpoint_manager.opt_tparams.write()

    # If the current loss is the worst
    if session_checkpoint_manager.training_history.content['metrics_measurements']['val']['loss'][-1][-1] > worst_measurements['val']['loss']:

      # Record all metrics for the same epoch
      for measurement_mode in {'train', 'val'}:
        for metric_id in metrics_ids:
          worst_measurements[measurement_mode][metric_id] = session_checkpoint_manager.training_history.content['metrics_measurements'][measurement_mode][metric_id][-1][-1]

    improvements = pd.DataFrame\
    (
      data=np.array\
      (
        [
          [
            '%.2f %%' % ( 100 * (session_checkpoint_manager.training_history.content['metrics_measurements']['val'][metric_id][-1][-1] - opt_measurements['val'][metric_id]) / (opt_measurements['val'][metric_id] + 10**(-8)) ),
            '%.2f %%' % ( 100 * (session_checkpoint_manager.training_history.content['metrics_measurements']['val'][metric_id][-1][-1] - worst_measurements['val'][metric_id]) / (worst_measurements['val'][metric_id] + 10**(-8)) ),
            '%.2f %%' % ( 100 * (session_checkpoint_manager.training_history.content['metrics_measurements']['val'][metric_id][-1][-1] - init_measurements['val'][metric_id]) / (init_measurements['val'][metric_id] + 10**(-8)) )
          ]
        ]
      ),
      index=['loss'],
      columns=['val_v_opt', 'val_v_worst', 'val_v_init']
    )

    last_model_measurements = pd.DataFrame\
    (
      data=np.array\
      (
        [
          [
            session_checkpoint_manager.training_history.content['metrics_measurements']['train'][metric_id][-1][0],
            session_checkpoint_manager.training_history.content['metrics_measurements']['train'][metric_id][-1][1],
            session_checkpoint_manager.training_history.content['metrics_measurements']['train'][metric_id][-1][2],
            session_checkpoint_manager.training_history.content['metrics_measurements']['train'][metric_id][-1][3],
            session_checkpoint_manager.training_history.content['metrics_measurements']['val'][metric_id][-1][-1]
          ] for metric_id in metrics_ids
        ]
      ),
      index=metrics_ids,
      columns=['est_train_min', 'est_train_max', 'est_train_avg', 'train', 'val']
    )

    session_checkpoint_manager.tparams.content = trainable_model.state_dict()

    # Epoch persistent storage
    session_checkpoint_manager.training_history.write()
    session_checkpoint_manager.tparams.write()

    # Update and store plot
    inv_metrics_measurements = session_checkpoint_manager.get_inverted_dataset_metric()
    for metric_id in metrics_ids:
      plt2d[metric_id].update_measurements(y_new=inv_metrics_measurements[metric_id], clear=True)
      session_checkpoint_manager.loss_history_plot[metric_id].content = plt2d[metric_id].plt2image()
      session_checkpoint_manager.loss_history_plot[metric_id].write()
    del inv_metrics_measurements

    # Time updates
    delta_t_epoch = time() - t_0_epoch
    delta_t_epoch_h = utils.logger.get_delta_t_h(t=delta_t_epoch)
    session_checkpoint_manager.training_history.content['total_time'] += delta_t_epoch
    session_checkpoint_manager.training_history.content['datetime_ended'] = datetime.now().astimezone(timezone.utc).strftime('D%Y%m%d%H%M%SUTC0')

    # stdout: Epoch state
    print('\n[EPOCH CONCLUSIVE REPORT]\n')
    print('Epoch time: %s'%(delta_t_epoch_h))
    print('Performance measurement time: %s'%(t_0_performance_measurement_period_h))
    print()
    print(tabulate(improvements, headers='keys', tablefmt='psql'))
    print(tabulate(last_model_measurements, headers='keys', tablefmt='psql'))

  delta_t_training = utils.logger.get_delta_t_h(t=session_checkpoint_manager.training_history.content['total_time'])

  print('\nTraining completed.')
  print('\n[TRAINING CONCLUSIVE REPORT]\n')
  print('Total training time: %s'%(delta_t_training))
  print('Exported Checkpoint:\n%s'%(current_session_chkp_id))