import os
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from PIL import Image
from io import BytesIO


class Plot2D:
  '''
  Description:
    A 2D plot corresponding to exactly one performance metric.
  '''

  def __init__(self, metric_id: str, y: dict[list[list]] = None):
    '''
    Description:
      Initializes plot information such as plot measurements included in argument y.

    Parameters:
      `metric_id`. Metric identifier (name).
      `y`. Performance measurements.
      If `y` is set to None:
        self.x and self.y are initialized appropriately.
      Else:
        `y`.
        |-- `y['train']`. Length 4. Performance measurement on train set. For some epoch.
        |   |-- `y['train'][epoch][0]`. Estimated minimum.
        |   |-- `y['train'][epoch][1]`. Estimated maximum.
        |   |-- `y['train'][epoch][2]`. Estimated average.
        |   |-- `y['train'][epoch][3]`. Precise average (after optimization within epoch).
        |
        |-- `y['val']`. Length 1. Performance measurement on validation set. For some epoch.
            |-- `y['val'][epoch][0]`. Average.
    '''

    self.plot_train_color = 'blue'
    self.plot_val_color = 'orange'
    self.fill_color = 'skyblue'

    self.metric_id = metric_id

    self.fig, self.ax = plt.subplots(figsize=(15, 7))

    if y is not(None):
      self.update_measurements(y_new=y, clear=False)
    else:
      self.y = {'train': [], 'val': []}
      self.x = []

  def check_measurements_integrity(self, y_like: dict[list[list]]):
    '''
    Description:
      Checks the integrity of y-axis measurements format. It has to align with the format of self.y.

    Parameters:
      `y_like`. Has the same format as y from __init__.
    '''

    integrity_status = True

    integrity_status = integrity_status and {'train', 'val'} == set(y_like.keys())
    integrity_status = integrity_status and len(y_like['train']) == len(y_like['val'])

    assert integrity_status, 'E: Invalid y format.'

    for value in y_like['train']:
      if not(type(value) == list and len(value) == 4):
        integrity_status = integrity_status and False
    for value in y_like['val']:
      if not(type(value) == list and len(value) == 1):
        integrity_status = integrity_status and False

    assert integrity_status, 'E: Invalid y format.'

  def update_measurements(self, y_new: dict[list[list]], clear: bool):
    '''
    Description:
      Updates y-axis measurements. Updates plot object accordingly.

    Parameters:
      `y_new`. New updated measurements. Has the same format as self.y.
      `clear`. Reset toggler of pyplot object.
    '''

    self.check_measurements_integrity(y_like=y_new)

    self.y = y_new
    self.x = list(range(len(self.y['train'])))

    if clear:
      self.ax.clear()

    self.ax.set_xlabel('epoch')
    self.ax.set_ylabel(self.metric_id)
    self.ax.grid(True)

    # Display integer valued x-axis ticks only
    self.ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    self.ax.plot(self.x, [self.y['train'][epoch][3] for epoch in self.x], label='train', color=self.plot_train_color)
    self.ax.plot(self.x, [self.y['val'][epoch][0] for epoch in self.x], label='val', color=self.plot_val_color)
    self.ax.fill_between(self.x, [self.y['train'][epoch][0] for epoch in self.x], [self.y['train'][epoch][1] for epoch in range(len(self.x))], color=self.fill_color, alpha=0.4, label='train dvt')

    self.ax.legend()

  def plt2image(self) -> Image.Image:
    '''
    Description:
      Generates and returns an image object of the plot.

    Returns:
      `img`. The plot image.
    '''

    buf = BytesIO()
    self.fig.savefig(fname=buf, format='png', bbox_inches='tight', dpi=300)
    buf.seek(0)
    img = Image.open(buf)

    return img