import os
import torch
from interpretation.eval_metrics import similarity
from utils.storage import TpFile, JsonFile, ImgFile
import interpretation.visualizer
import numpy as np
import data.dataset
import data.utils
import utils.logger
import random
import model.emodel
from time import time


def estimate(cfg_fp: str, tparams_fp: str, device_id: str):
  '''
  Description:
    Ranks a set of test images with respect to one base image using a contrastive learning model.

  Parameters:
    `cfg_fp`. Configuration file path to be parsed.
    `tparams_fp`. Filepath of the pretrained contrastive model's trainable parameters.
    `device_id`. Device identifier to be selected for DL framework.
  '''

  config = JsonFile(abs_fp=os.path.abspath(cfg_fp))
  tparams = TpFile(abs_fp=os.path.join(os.environ['ROOT_ABS_DP'], tparams_fp))

  if device_id == 'gpu':
    device = torch.device(device='cuda:0' if torch.cuda.is_available() else 'cpu')
  else:
    device = torch.device(device='cpu')

  # Prepare configuration parameters
  rng_seed = config.content['training_cfg']['rng_seed']
  model_id = config.content['model_architecture_cfg']['type']
  data_id = config.content['training_cfg']['data']
  instance_prsd_shape = config.content['model_architecture_cfg']['instance_prsd_shape']
  M_minibatch = config.content['training_cfg']['M_minibatch']
  train_fraction = config.content['training_cfg']['train_fraction']
  subset_size = config.content['training_cfg']['subset_size']

  np.random.seed(seed=rng_seed)

  if 'limit2SmallSubsetofData' in os.environ['DEBUG_CONFIG'].split(';'):
    subset_size = 100

  # Get test set with seed
  dataset = data.dataset.Cifar(instance_prsd_shape=instance_prsd_shape, M_minibatch=M_minibatch, train_fraction=train_fraction, subset_size=subset_size, parse_labels=True, augm=False, device=device)

  estimator_model = model.emodel.estimator_model_assembler(model_architecture_cfg=config.content['model_architecture_cfg'], device=device, state_dict=tparams.content)

  basis_idx = random.randint(0, len(dataset.test_set)-1)
  test_set_ist = dataset.test_set.x
  basis_img = test_set_ist[basis_idx] # Shape (3, H, W)

  dataset.train_set.shuffle, dataset.val_set.shuffle, dataset.test_set.shuffle = 3*[False]

  # Basis image estimation
  basis_img_repr = estimator_model(basis_img[torch.newaxis, ...])[0, :].detach()

  ## Train set estimation

  train_set_ist = []
  train_set_repr = []
  t_0 = time()
  compounding_delta_t_stdout = 0
  print('Train set estimation state:')

  while next(dataset.train_set):

    # Get augmented pairs from current minibatch
    train_set_ist_minibatch = dataset.train_set.x_current_grp

    # Forward pass; shape (G[M_minibatch], N_repr)
    train_set_minibatch_repr = estimator_model(train_set_ist_minibatch).detach()

    train_set_ist.append(train_set_ist_minibatch)
    train_set_repr.append(train_set_minibatch_repr)

    delta_t_iteration = (time()-t_0) / (dataset.train_set.minibatch_idx+1)

    # stdout: Iteration state
    compounding_delta_t_stdout += delta_t_iteration
    if dataset.train_set.minibatch_idx == 0 or dataset.train_set.minibatch_idx == dataset.train_set.n_steps - 1 or compounding_delta_t_stdout > 0.1:
      compounding_delta_t_stdout = 0
      progress_bar_ist = utils.logger.get_progress_bar_ist(i=dataset.train_set.minibatch_idx, n=dataset.train_set.n_steps-1, delta_t=delta_t_iteration, loss=None, bar_length = 30, bar_background_char = ' ', bar_fill_char = '=')
      utils.logger.dnmc_stdout_write(s=progress_bar_ist)

  train_set_ist = torch.cat(tensors=train_set_ist, dim=0)
  train_set_repr = torch.cat(tensors=train_set_repr, dim=0)

  ## Val set estimation

  val_set_ist = []
  val_set_repr = []
  t_0 = time()
  compounding_delta_t_stdout = 0
  print('Val set estimation state:')

  while next(dataset.val_set):

    # Get augmented pairs from current minibatch
    val_set_ist_minibatch = dataset.val_set.x_current_grp

    # Forward pass; shape (G[M_minibatch], N_repr)
    val_set_minibatch_repr = estimator_model(val_set_ist_minibatch).detach()

    val_set_ist.append(val_set_ist_minibatch)
    val_set_repr.append(val_set_minibatch_repr)

    delta_t_iteration = (time()-t_0) / (dataset.val_set.minibatch_idx+1)

    # stdout: Iteration state
    compounding_delta_t_stdout += delta_t_iteration
    if dataset.val_set.minibatch_idx == 0 or dataset.val_set.minibatch_idx == dataset.val_set.n_steps - 1 or compounding_delta_t_stdout > 0.1:
      compounding_delta_t_stdout = 0
      progress_bar_ist = utils.logger.get_progress_bar_ist(i=dataset.val_set.minibatch_idx, n=dataset.val_set.n_steps-1, delta_t=delta_t_iteration, loss=None, bar_length = 30, bar_background_char = ' ', bar_fill_char = '=')
      utils.logger.dnmc_stdout_write(s=progress_bar_ist)

  val_set_ist = torch.cat(tensors=val_set_ist, dim=0)
  val_set_repr = torch.cat(tensors=val_set_repr, dim=0)

  ## Test set estimation

  test_set_ist = []
  test_set_repr = []
  t_0 = time()
  compounding_delta_t_stdout = 0
  print('Test set estimation state:')

  while next(dataset.test_set):

    # Get augmented pairs from current minibatch
    test_set_ist_minibatch = dataset.test_set.x_current_grp

    # Forward pass; shape (G[M_minibatch], N_repr)
    test_set_minibatch_repr = estimator_model(test_set_ist_minibatch).detach()

    test_set_ist.append(test_set_ist_minibatch)
    test_set_repr.append(test_set_minibatch_repr)

    delta_t_iteration = (time()-t_0) / (dataset.test_set.minibatch_idx+1)

    # stdout: Iteration state
    compounding_delta_t_stdout += delta_t_iteration
    if dataset.test_set.minibatch_idx == 0 or dataset.test_set.minibatch_idx == dataset.test_set.n_steps - 1 or compounding_delta_t_stdout > 0.1:
      compounding_delta_t_stdout = 0
      progress_bar_ist = utils.logger.get_progress_bar_ist(i=dataset.test_set.minibatch_idx, n=dataset.test_set.n_steps-1, delta_t=delta_t_iteration, loss=None, bar_length = 30, bar_background_char = ' ', bar_fill_char = '=')
      utils.logger.dnmc_stdout_write(s=progress_bar_ist)

  test_set_ist = torch.cat(tensors=test_set_ist, dim=0)
  test_set_repr = torch.cat(tensors=test_set_repr, dim=0)

  ## Report generation

  # Evaluation
  train_set_sim = similarity(vec=basis_img_repr, mat=train_set_repr)
  val_set_sim = similarity(vec=basis_img_repr, mat=val_set_repr)
  test_set_sim = similarity(vec=basis_img_repr, mat=test_set_repr)

  # Similarity (descenting) sorting of instances and scores with respect to scores
  train_set_sim, sorted_train_set_idcs = torch.sort(input=train_set_sim, dim=0, descending=True)
  train_set_sim = train_set_sim.cpu().numpy()
  train_set_ist = np.transpose(a=train_set_ist[sorted_train_set_idcs].cpu().numpy(), axes=(0, 2, 3, 1))
  val_set_sim, sorted_val_set_idcs = torch.sort(input=val_set_sim, dim=0, descending=True)
  val_set_sim = val_set_sim.cpu().numpy()
  val_set_ist = np.transpose(a=val_set_ist[sorted_val_set_idcs].cpu().numpy(), axes=(0, 2, 3, 1))
  test_set_sim, sorted_test_set_idcs = torch.sort(input=test_set_sim, dim=0, descending=True)
  test_set_sim = test_set_sim.cpu().numpy()
  test_set_ist = np.transpose(a=test_set_ist[sorted_test_set_idcs].cpu().numpy(), axes=(0, 2, 3, 1))

  # Pyplot grid object (encoded in .png) containing top-5 and bottom-5
  plt_img_train_file = ImgFile(abs_fp='tmp/plt_img_train.png')
  plt_img_train_file.content = interpretation.visualizer.topnbottom5images(top5=train_set_ist[:5], bottom5=train_set_ist[-5:], top5_scores=train_set_sim[:5], bottom5_scores=train_set_sim[-5:])
  plt_img_val_file = ImgFile(abs_fp='tmp/plt_img_val.png')
  plt_img_val_file.content = interpretation.visualizer.topnbottom5images(top5=val_set_ist[:5], bottom5=val_set_ist[-5:], top5_scores=val_set_sim[:5], bottom5_scores=val_set_sim[-5:])
  plt_img_test_file = ImgFile(abs_fp='tmp/plt_img_test.png')
  plt_img_test_file.content = interpretation.visualizer.topnbottom5images(top5=test_set_ist[:5], bottom5=test_set_ist[-5:], top5_scores=test_set_sim[:5], bottom5_scores=test_set_sim[-5:])

  # Clear tmp directory; export pyplot images; store base image (for context)
  utils.storage.rm_tmp()
  plt_img_train_file.write()
  plt_img_val_file.write()
  plt_img_test_file.write()
  # 

  # Export images


  # stdout

