from torch.nn import Parameter
import torch
import model.backbone


def estimator_model_assembler(model_architecture_cfg: dict, device: torch.device, state_dict: Parameter) -> torch.nn.Module:
  '''
  Description:
    Assembles the estimator model's configuration based on provided parameters.

  Parameters:
    `model_architecture_cfg`. Aligns with the structure of config.json.
    `device`.
    `state_dict`. Contains the base model's trainable parameters.

  Returns:
    `estimator_model`. The estimator torch model.
  '''

  feasible_model_types = {'SimpleCNN'}

  assert model_architecture_cfg['type'] in feasible_model_types, 'E: Invalid model type.'

  if model_architecture_cfg['type'] == 'SimpleCNN':
    basis = model.backbone.SimpleCNN(N_repr=model_architecture_cfg['N_repr'], device=device)

  state_dict_base = dict()
  for tparams_key in state_dict.keys():
    block_id = tparams_key.split('.')[0]
    layer_id = '.'.join(tparams_key.split('.')[1:])
    if block_id == 'base':
      state_dict_base[layer_id] = state_dict[tparams_key]

  basis.load_state_dict(state_dict=state_dict_base)

  return basis