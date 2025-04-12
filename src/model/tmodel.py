from collections import OrderedDict
from torch.nn import Parameter
import torch
import model.backbone
import model.head


class TrainableModel(torch.nn.Module):

  def __init__(self, base, proj, device: torch.device):

    super(TrainableModel, self).__init__()
    self.base = base.to(device)
    self.proj = proj.to(device)

  def forward(self, x):

    h = self.base(x)
    g = self.proj(h)

    return g

def trainable_model_assembler(model_architecture_cfg: dict, device: torch.device, state_dict: OrderedDict[str, Parameter] = None) -> torch.nn.Module:
  '''
  Description:
    Assembles the trainable model's configuration based on provided parameters. The trainable parameters are also optionally substituted based on the argument (otherwise the parameters are initialized).

  Parameters:
    `model_architecture_cfg`. Aligns with the structure of config.json.
    `device`.
    `state_dict`. Contains the model's trainable parameters.

  Returns:
    `trainable_model`. The trainable torch model.
  '''

  feasible_model_types = {'SimpleCNN'}

  assert model_architecture_cfg['type'] in feasible_model_types, 'E: Invalid model type.'

  if model_architecture_cfg['type'] == 'SimpleCNN':
    base = model.backbone.SimpleCNN(N_repr=model_architecture_cfg['N_repr'], device=device)
    proj = model.head.MLP(N_repr=model_architecture_cfg['N_repr'], device=device)

  trainable_model = TrainableModel(base=base, proj=proj, device=device)

  if state_dict is not None:
    trainable_model.load_state_dict(state_dict=state_dict)

  return trainable_model