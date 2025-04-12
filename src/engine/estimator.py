import os
import torch
from interpretation.eval_metrics import similarity
from utils.storage import TpFile, JsonFile
import interpretation.visualizer
import numpy as np
import data.dataset
import data.utils
import random
import model.emodel


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

  basis_idx = random.randint(0, len(dataset.test_set))
  test_set_ist = dataset.test_set.x
  basis_img = test_set_ist[basis_idx]

  minibatch_instance_indcs_grp = data.utils.generate_minibatch_idcs(M=len(test_set_ist), M_grp=M_minibatch, shuffle=False)
  breakpoint()

  # for :
    # test_set_ist_minibatch = 
    # test_set_ist_minibatch_pairs = 
    # estimator_model()
    # interpretation.eval_metrics.similarity(vec=basis_repr_vec, mat=test_set_ist_repr)

  # Construct representation model (use breakpoint to filter out the head)

  # Rank image set indices

  # Display top-5 images using visualizer
