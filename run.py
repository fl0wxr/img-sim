import os
import sys
import argparse

os.environ['ROOT_ABS_DP'] = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(os.environ['ROOT_ABS_DP'], 'src'))

from engine.trainer import train as train
from engine.estimator import estimate as estimate
from utils.storage import rm_chkp


acceptable_modes = ['train', 'pred']
acceptable_debug_parameters = ['limit2SmallSubsetofData', 'clearExports']
acceptable_device = ['cpu', 'gpu']

parser = argparse.ArgumentParser(description='Entry point.')
parser.add_argument('--device', required=True, choices=acceptable_device, type=str, help='Device for deep learning backend.')
parser.add_argument('--mode', required=True, choices=acceptable_modes, type=str, help='Mandatory. Set the mode to specify the purpose of this session. Acceptable values:\n\t"train". For training.\n\t"pred". For estimation/prediction based on some model. The config.json and tparams.pt files have to be compatible.')
parser.add_argument('--checkpoint', required=False, type=str, help='Optional. Directory name containing the files of some past training session. Refer to the documentation to understand the underlying directory structure and file format of stored checkpoints.')
parser.add_argument('--config', required=False, type=str, help='Optional for "train" mode and mandatory for "pred" mode. File name of .json file that contains the configuration of some training algorithm and a model. The file name must include the file extension. The referred file should be located within "template".')
parser.add_argument('--tparams', required=False, type=str, help='Mandatory for "pred" mode. Specifies the .pt file path holding the model\'s state_dict.')
parser.add_argument('--debug', required=False, choices=acceptable_debug_parameters, nargs='+', help='Optional. Debug mode. This parameter can be utilized by this software\'s developer.')

args = parser.parse_args()

# Check compatibility of --mode with other prompt parameters.
if args.mode not in acceptable_modes:
  parser.error('Invalid mode argument.')
if args.mode == 'train':
  if not(bool(args.checkpoint)^bool(args.config)):
    parser.error('In train mode, exactly one of --checkpoint and --config must be specified. User has specified either both or neither.')
  if bool(args.tparams):
    parser.error('--tparams is an invalid parameter in "train" mode. Did you mean to specify the --checkpoint directory instead?')
elif args.mode == 'pred':
  if not(bool(args.config)):
    parser.error('--config is a mandatory parameter in "pred" mode.')
  if bool(args.checkpoint):
    parser.error('--checkpoint is an invalid parameter in "pred" mode.')
  if not(bool(args.tparams)):
    parser.error('--tparams is a mandatory parameter in "pred" mode.')

# Check paths
if bool(args.config) and args.config.split('.')[-1] != 'json':
  parser.error('The specified --config file name must have a json extension.')
if bool(args.tparams) and args.tparams.split('.')[-1] != 'pt':
  parser.error('The specified --tparams file name must have a pt extension.')

# Check --debug parameter
os.environ['DEBUG_CONFIG'] = ''
if bool(args.debug):
  if not(set(args.debug).issubset(acceptable_debug_parameters)):
    parser.error('Invalid --debug parameter.')
  os.environ['DEBUG_CONFIG'] = ';'.join(args.debug)
  print('W: Debug config is active.')


if __name__ == '__main__':
  if args.mode == 'train':
    train(basis_chkp_id=args.checkpoint, basis_cfg_fp=args.config, device_id=args.device)
  elif args.mode == 'pred':
    estimate(cfg_fp=args.config, tparams_fp=args.tparams, device_id=args.device)

  if 'clearExports' in os.environ['DEBUG_CONFIG'].split(';'):
    rm_chkp()