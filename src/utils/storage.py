import os
import json
import shutil
import glob
import torch
from datetime import datetime


class File_:

  def __init__(self, abs_fp: str):
    self.abs_fp = abs_fp
    self.extension = self.abs_fp.split('.')[-1]
    self.content = None

class JsonFile(File_):

  def __init__(self, abs_fp: str):
    super().__init__(abs_fp=abs_fp)
    if os.path.exists(self.abs_fp):
      self.read()

  def read(self):
    if os.path.exists(self.abs_fp):
      with open(file=self.abs_fp, mode='r') as f:
        self.content = json.load(fp=f)
    else:
      self.content = None

  def write(self):
    if not('disableExports' in os.environ['DEBUG_CONFIG'].split(';')):
      with open(self.abs_fp, 'w') as f:
        json.dump(self.content, f, indent=2)

class TpFile(File_):

  def __init__(self, abs_fp: str):
    super().__init__(abs_fp=abs_fp)
    if os.path.exists(self.abs_fp):
      self.read()

  def read(self):
    if os.path.exists(self.abs_fp):
      self.content = torch.load(f=self.abs_fp)
    else:
      self.content = None

  def write(self):
    if not('disableExports' in os.environ['DEBUG_CONFIG'].split(';')):
      torch.save(obj=self.content, f=self.abs_fp)

class ImgFile(File_):

  def __init__(self, abs_fp: str):
    super().__init__(abs_fp=abs_fp)

  def write(self):
    self.content.save(self.abs_fp)

class CheckpointStorageManager:
  '''
  Description:
    Storage handling of directory, filepaths of training checkpoints. An instance of this class ensures the handling of a new directory tree under model/checkpoint.
  '''

  ROOT_ABS_DP = os.environ['ROOT_ABS_DP']

  def __init__(self, chkp_id: str, basis_path: str, session_datetime: datetime):
    '''
    Parameters:
      `chkp_id`. Used as the parent directory name for current checkpoint.
      `basis_path`. Used as a basis to contruct the various data structures and directory of the session's checkpoint mechanism. Cases:
      |-- A existing configuration json file's base name from within the template directory.
      |-- A existing checkpoint directories base name that used as an initial point.

      `session_datetime`. Datetime of current session.
    '''

    self.chkp_root_abs_dp = os.path.join(self.ROOT_ABS_DP, 'checkpoint', chkp_id)
    self.session_datetime = session_datetime

    # Directory initialization
    if not os.path.exists(self.chkp_root_abs_dp):
      os.mkdir(self.chkp_root_abs_dp)
      print('Created a new checkpoint directory with path:\n%s'%(self.chkp_root_abs_dp))
    print('The checkpoint directory with path:\n%s\nis marked as the current checkpoint during this session.'%(self.chkp_root_abs_dp))

    self.config = JsonFile(abs_fp=os.path.join(self.chkp_root_abs_dp, 'config.json'))
    self.tparams = TpFile(abs_fp=os.path.join(self.chkp_root_abs_dp, 'tparams.pt'))
    self.opt_tparams = TpFile(abs_fp=os.path.join(self.chkp_root_abs_dp, 'opt_tparams.pt'))
    self.training_history = JsonFile(abs_fp=os.path.join(self.chkp_root_abs_dp, 'training-history.json'))

    self.initialize_content(basis_path=basis_path)

    self.metrics_ids = list(self.training_history.content['metrics_measurements']['train'].keys())

    self.loss_history_plot = dict()
    for metric_id in self.metrics_ids:
      self.loss_history_plot[metric_id] = ImgFile(abs_fp=os.path.join(self.chkp_root_abs_dp, f'{metric_id}.png'))

  def check_training_history_file_structural_integrity(self):
    '''
    Description:
      Performs data structure integrity check on the training history file. Must be executed after every update of the file.
    '''

    metrics_measurements = self.training_history.content['metrics_measurements']

    # Train and validation metric measurements must share the same number of epochs; ensures all train metrics and validation metrics have a pairwise equal length
    integrity_status = metrics_measurements['train'].keys() == metrics_measurements['val'].keys()
    for eval_key in metrics_measurements['train'].keys():
      integrity_status = integrity_status and (len(metrics_measurements['train'][eval_key]) == len(metrics_measurements['val'][eval_key]))

    m = len(metrics_measurements['train']['loss'])

    # All metrics must share the same number of epochs; ensures all train metrics have an equal length
    for eval_key0 in metrics_measurements['train'].keys():
      for eval_key1 in metrics_measurements['train'].keys():
        integrity_status = integrity_status and (m == len(metrics_measurements['train'][eval_key0]) == len(metrics_measurements['train'][eval_key1]))

    assert integrity_status, 'E: Invalid training history file.'

  def initialize_content(self, basis_path: str) -> dict:
    '''
    Description:
      Initializes training history and configuration files and data either from scratch using the template and config files, or simply by based on some existing checkpoint data (e.g., past training-history.json measurements are preserved).

    Parameters:
      `basis_path`. Path for basis/source checkpoint for training restart or the path of some configuration template if this is a clean start.
    '''

    if basis_path.split('.')[-1] != 'json':
      # A session directory is specified.

      basis_chkp_dp = os.path.abspath(os.path.join(self.ROOT_ABS_DP, 'checkpoint', basis_path))
      assert os.path.exists(basis_chkp_dp), 'E: Invalid source checkpoint directory name.'
      for basis_chkp_fp in glob.glob(os.path.join(basis_chkp_dp, "*")):
        shutil.copy(basis_chkp_fp, self.chkp_root_abs_dp)

      self.config.read()
      self.tparams.read()
      self.opt_tparams.read()
      self.training_history.read()

    else: # A config file is specified.

      # Copy everything to new session directory
      basis_config_fp = os.path.abspath(basis_path)
      assert os.path.exists(basis_config_fp), 'E: Invalid configuration filename.'
      shutil.copy(basis_config_fp, self.config.abs_fp)
      shutil.copy(os.path.join(self.ROOT_ABS_DP, 'template', 'training-history.json'), self.training_history.abs_fp)

      self.config.read()
      self.training_history.read()

      # Initialize some json file values
      self.training_history.content['torch_version'] = str(torch.__version__)
      self.training_history.content['datetime_started'] = self.session_datetime.strftime('D%Y%m%d%H%M%SUTC0')
      self.training_history.content['datetime_ended'] = self.session_datetime.strftime('D%Y%m%d%H%M%SUTC0')

    self.check_training_history_file_structural_integrity()

  def get_inverted_dataset_metric(self) -> dict[dict[list]]:
    '''
    Description:
      From the metrics_measurements (alias for self.training_history.content['metrics_measurements']), this function is used to invert the dataset and metric index positions.
      `metrics_measurements`.
      |-- `metrics_measurements['train']`. Performance measurements on train set.
      |   |-- `metrics_measurements['train'][metric_id]`. For some valid metric_id (e.g., 'loss') and epoch.
      |       |-- `metrics_measurements['train'][metric_id][epoch][0]`. Estimated minimum.
      |       |-- `metrics_measurements['train'][metric_id][epoch][1]`. Estimated maximum.
      |       |-- `metrics_measurements['train'][metric_id][epoch][2]`. Estimated average.
      |       |-- `metrics_measurements['train'][metric_id][epoch][3]`. Precise average (after optimization within epoch).
      |
      |-- `metrics_measurements['val']`. Length 1. Performance measurement on validation set.
          |-- `metrics_measurements['val'][metric_id]`. For some valid metric_id (e.g., 'loss') and epoch.
              |-- `metrics_measurements['val'][metric_id][epoch][0]`. Average.

    Returns:
      `inv_metrics_measurements`. Contains all information from metrics_measurements, but with inverted index positions. Hence the order of indices is metric_id -> dataset. E.g.,
      inv_metrics_measurements['loss']['train'][2][2] is the same as metrics_measurements['train']['loss'][2][2]
    '''

    metrics_measurements = self.training_history.content['metrics_measurements']

    metrics_ids = metrics_measurements['train'].keys()
    inv_metrics_measurements = {metric_id: None for metric_id in metrics_ids}
    for metric_id in metrics_ids:
      inv_metrics_measurements[metric_id] = \
      {
        'train': [metrics_measurements['train'][metric_id][epoch] for epoch in range(len(metrics_measurements['train'][metric_id]))],
        'val': [metrics_measurements['val'][metric_id][epoch] for epoch in range(len(metrics_measurements['val'][metric_id]))]
      }

    return inv_metrics_measurements

def rm_chkp():
  paths = glob.glob(os.path.join(os.environ['ROOT_ABS_DP'], 'checkpoint', '*'))
  for path in paths:
    if 'checkpoint/D' in path:
      shutil.rmtree(path)

def rm_tmp():
  paths = glob.glob(os.path.join(os.environ['ROOT_ABS_DP'], 'tmp', '*'))
  for path in paths:
    if 'tmp/' in path:
      if os.path.isdir(path):
        shutil.rmtree(path)
      elif os.path.isfile(path):
        os.remove(path)