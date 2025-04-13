import sys
import shutil
from functools import wraps


def get_delta_t_h(t: float) -> str:
  '''
  Description:
    Converts time from seconds (and a fraction of seconds), into human readable format with appropriate unit.

  Parameters:
    `t`. Time.

  Returns:
    `t_h`. Human readable time.
  '''

  h = int(t//60**2)
  m = int((t-h*60**2)//60)
  s = int(t-m*60-h*60**2)
  fr = round(1000*(t-s-m*60-h*60**2))

  if 24*60**2 < t: # [24 h, \infty)
    t_h = f'{h}h'
  elif 60**2 <= t and t < 24*60**2: # [1 h, 24 h)
    t_h = f'{h:02d}h:{m:02d}m'
  elif 60 <= t and t < 60**2: # [1 m, 1 h)
    t_h = f'{m:02d}m:{s:02d}s'
  elif 1 <= t and t < 60: # [1 s, 1 m)
    t_h = f'{s:02d}s'
  elif 0 <= t and t < 1: # [0 s, 1 s)
    t_h = f'{fr:03d}ms'

  return t_h

def progress_panel_ist_dec(f):

  @wraps(f)
  def progress_panel_ist(**kwargs) -> str:
    '''
    Description:
      A panel that displays additional progress bar information.

    Returns:
      `progress_panel`. Progression panel in string.
    '''

    prog_bar = f(**kwargs)
    i = kwargs['i']
    n = kwargs['n']
    delta_t = kwargs['delta_t']
    loss = kwargs['loss']

    # Assuming time intervals do not depend on i
    est_elps_delta_t = delta_t*i
    est_rmng_delta_t = delta_t*(n-i)

    est_elps_delta_t_h = get_delta_t_h(t=est_elps_delta_t)
    est_rmng_delta_t_h = get_delta_t_h(t=est_rmng_delta_t)
    delta_t_h = get_delta_t_h(t=delta_t)
    if n != 0:
      completion_fraction = int((i/n)*100)
    else:
      completion_fraction = int(100)

    step_len = len(str(n))

    progress_panel = f'\r{completion_fraction:3d}% {prog_bar} S{i:{step_len}d}/{n} [{est_elps_delta_t_h}<{est_rmng_delta_t_h}; {delta_t_h}/it'

    if loss is not(None):
      progress_panel += f'; L{loss:.4f}]'
    else:
      progress_panel += f']'

    if i == n:
      progress_panel += '\n'

    return progress_panel

  progress_panel_ist.__doc__ = f'Wrapper around "{f.__name__}": {f.__doc__}\n{progress_panel_ist.__doc__}'

  return progress_panel_ist

@progress_panel_ist_dec
def get_progress_bar_ist(*, i: int, n: int, delta_t: float, loss: float = None, bar_length: int = 30, bar_background_char: str = ' ', bar_fill_char: str = '=') -> str:
  '''
  Description:
    Converts the fraction i/n to progression bar in text. Make sure you apply this right after the iteration's primary computations.

  Parameters:
    `i`. Current iteration. Starting from 0.
    `n`. The last iteration, or (number of iterations)-1.
    `delta_t`. Time per iteration.
    `loss`.
    `bar_length`. Total length of the bar.
    `bar_background_char`. Background character of the bar.
    `bar_fill_char`. Fill character of the bar.

  Returns:
    `prog_bar`. Printable progression bar.
  '''

  if n != 0:
    bar_fill_length = int(bar_length * i // n)
  else:
    bar_fill_length = int(bar_length)
  prog_bar = '|' + bar_fill_length*bar_fill_char + (bar_length-bar_fill_length)*bar_background_char + '|'

  return prog_bar

def dnmc_stdout_write(s: str, clear: bool = True):

  if clear:
    c = shutil.get_terminal_size().columns

    sys.stdout.write('\r'+c*' ')
    sys.stdout.flush()

  sys.stdout.write(s)
  sys.stdout.flush()

class Smoother:
  '''
  Description:
    Models a sample of values into a moving average with a predetermined window size.
  '''

  def __init__(self, window_size = 10):
    '''
    Parameters:
      `window_size`. The maximum sample size.
    '''

    self.window_size = window_size
    self.initialize_sample()

  def initialize_sample(self):
    self.sample = []

  def update_sample(self, value: float):
    '''
    Parameters:
      `value`. A new value appended to the sample.

    Returns:
      `smoothed_value`.
    '''

    self.sample.append(value)
    if len(self.sample) > self.window_size:
      self.sample.pop(0)
    smoothed_value = sum(self.sample) / len(self.sample)

    return smoothed_value