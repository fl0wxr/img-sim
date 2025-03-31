import os
import json
import shutil
import glob
import torch
from datetime import datetime, timezone
from collections import OrderedDict
from torch.nn import Parameter


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
    torch.save(obj=self.content, f=self.abs_fp)

class CheckpointStorageManager:
  '''
  Description:
    Storage handling of directory, filepaths of training checkpoints. An instance of this class ensures the handling of a new directory tree under model/checkpoint.
  '''

  ROOT_ABS_DP = os.environ['ROOT_ABS_DP']

  def __init__(self, chkp_id: str, basis_id: str, session_datetime: datetime):
    '''
    Parameters:
      `chkp_id`. Used as the parent directory name for current checkpoint.
      `basis_id`. Used as a basis to contruct the various data structures and directory of the session's checkpoint mechanism. Cases:
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
    self.loss_history_plot = File_(abs_fp=os.path.join(self.chkp_root_abs_dp, 'loss.png'))
    self.training_stdout = File_(abs_fp=os.path.join(self.chkp_root_abs_dp, 'training.log'))

    self.initialize_content(basis_id=basis_id)

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

  def initialize_content(self, basis_id: str) -> dict:
    '''
    Description:
      Initializes training history and configuration files and data either from scratch using the template and config files, or simply by based on some existing checkpoint data (e.g., past training-history.json measurements are preserved).

    Parameters:
      `basis_id`. Path for basis/source checkpoint for training restart or the path of some configuration template if this is a clean start.
    '''

    if basis_id.split('.')[-1] != 'json': # A session directory is specified

      basis_chkp_dp = os.path.join(self.ROOT_ABS_DP, basis_id)
      for basis_chkp_fp in glob.glob(os.path.join(basis_chkp_dp, "*")):
        shutil.copy(basis_chkp_fp, self.chkp_root_abs_dp)

    else: # A config file is specified.

      # Copy everything to new session directory
      shutil.copy(os.path.join(self.ROOT_ABS_DP, 'template', basis_id), self.config.abs_fp)
      shutil.copy(os.path.join(self.ROOT_ABS_DP, 'template', 'training-history.json'), self.training_history.abs_fp)

      self.config.read()
      self.training_history.read()

      # Initialize some json file values
      self.training_history.content['torch_version'] = str(torch.__version__)
      self.training_history.content['datetime_started'] = self.session_datetime.strftime('D%Y%m%d%H%M%SUTC0')
      self.training_history.content['datetime_ended'] = self.session_datetime.strftime('D%Y%m%d%H%M%SUTC0')

    self.check_training_history_file_structural_integrity()

  def rm_chkp(self):
    if 'checkpoint/D20' in self.chkp_root_abs_dp:
      shutil.rmtree(self.chkp_root_abs_dp)