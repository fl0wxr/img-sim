import os
import sys

os.environ['ROOT_ABS_DP'] = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(os.environ['ROOT_ABS_DP'], 'src'))

import argparse
from engine.trainer import train as train


modes = ['train',]
acceptable_debug_parameters = ['limit2SmallSubsetofData', 'undoCheckpoint', 'disableTraining']

parser = argparse.ArgumentParser(description='Entry point.')
parser.add_argument('--mode', required=True, choices=modes, type=str, help='Mandatory. Set the mode to specify the purpose of this session. Acceptable values:\n\t"train". For training.\n\t"eval". For evaluation.')
parser.add_argument('--checkpoint', required=False, type=str, help='Optional. Directory name containing the files of some past training session. Refer to the documentation to understand the underlying directory structure and file format of stored checkpoints.')
parser.add_argument('--config', required=False, type=str, help='Optional. File name of .json file that contains the configuration of some training algorithm and a model. The file name must include the file extension. The referred file should be located within "template".')
parser.add_argument('--debug', required=False, choices=acceptable_debug_parameters, nargs='+', help='Optional. Debug mode. This parameter can be utilized by this software\'s developer.')

args = parser.parse_args()

if args.mode not in modes:
  parser.error('Invalid mode argument.')
if args.mode == 'train':
  if not(bool(args.checkpoint)^bool(args.config)):
    parser.error('In train mode, exactly one of --checkpoint and --config must be specified. User has specified either both or neither.')
else:
  if bool(args.config):
    parser.error('--config is an invalid parameter in "eval" mode.')
  elif not(bool(args.checkpoint)):
    parser.error('--checkpoint has to be specified in "eval" mode.')

if bool(args.config) and args.config.split('.')[-1] != 'json' and args.config != 'json':
  parser.error('The specified --config file name does not have a json extension.')

os.environ['DEBUG_CONFIG'] = ''
if bool(args.debug):
  if not(set(args.debug).issubset(acceptable_debug_parameters)):
    parser.error('Invalid --debug parameter.')
  os.environ['DEBUG_CONFIG'] = ';'.join(args.debug)
  print('W: Debug config is active.')


if __name__ == '__main__':
  if args.mode == 'train':
    train(basis_chkp_id=args.checkpoint, basis_cfg_id=args.config)