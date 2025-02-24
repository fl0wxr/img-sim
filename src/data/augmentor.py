from copy import deepcopy
import numpy as np
import cv2

from numpy.typing import NDArray


class augmentor:
  '''
  Description:
    A set of augmentation utilities applicable to a set of images.
  '''

  def __init__(self, enable_crop: bool, enable_jitter: bool, enable_color_drop: bool) -> None:

    self.data_set_ist_nrm_bgr = None
    self.enable_crop = enable_crop
    self.enable_jitter = enable_jitter
    self.enable_color_drop = enable_color_drop

  def update_input_instance_set(self, data_set_ist_nrm_bgr: NDArray) -> None:
    '''
    Description:
      Used to update the input dataset.

    Parameters:
      `data_set_ist_nrm_bgr`. Shape (M, H, W, 3). A sequence of image arrays. Pixels adhere to the BGR color model.
    '''

    self.data_set_ist_nrm_bgr = data_set_ist_nrm_bgr
    self.M, self.H, self.W = self.data_set_ist_nrm_bgr.shape[:3]

  def sample_augmented_pairs(self) -> NDArray:
    '''
    Description:
      Generates a pair of augmented images per input image from the input instance set.

    Returns:
      `sampled_augmentation_img_pairs_nrm`. Shape (M, 2, H, W, 3). A sequence of image arrays, where the second axis corresponds to the augmented image and the first axis specifies the original input instance which the augmented pair is associated with. Pixels adhere to the BGR color model.
    '''

    sampled_augmentation_img_pairs_nrm = []

    for ist_idx in range(self.M):
      base_ist_nrm_bgr = self.data_set_ist_nrm_bgr[ist_idx, ...]
      augmented_ist_pair = []
      for augmented_ist_idx in range(2):
        augmented_ist_nrm_bgr = deepcopy(base_ist_nrm_bgr)
        if self.enable_crop:
          augmented_ist_nrm_bgr = apply_random_crop(img_arr=augmented_ist_nrm_bgr)
        if self.enable_jitter:
          augmented_ist_nrm_bgr = apply_random_jitter(img_arr_nrm_bgr=augmented_ist_nrm_bgr)
        if self.enable_color_drop:
          augmented_ist_nrm_bgr = apply_random_color_drop(img_arr_nrm_bgr=augmented_ist_nrm_bgr)
        augmented_ist_pair.append(augmented_ist_nrm_bgr)
      sampled_augmentation_img_pairs_nrm.append(augmented_ist_pair)

    sampled_augmentation_img_pairs_nrm = np.array(sampled_augmentation_img_pairs_nrm)

    return sampled_augmentation_img_pairs_nrm


def bgr2gs(img_arr_nrm_bgr: NDArray) -> NDArray:
  '''
  Description:
    Converts a BGR image to its grayscale equivalent.

  Parameters:
    `img_arr_nrm_bgr`. Shape (H, W, 3). Normalized input image array. Pixels adhere to the BGR color model.

  Returns:
    `img_arr_nrm_gs_out`. Shape (H, W, 3).
  '''

  img_arr_nrm_gs_out = np.stack([0.114*img_arr_nrm_bgr[..., 0], 0.587*img_arr_nrm_bgr[..., 1], 0.299*img_arr_nrm_bgr[..., 2]], axis=-1)

  return img_arr_nrm_gs_out

def apply_random_crop(img_arr: NDArray) -> NDArray:
  '''
  Description:
    Crops an image randomly while preserving the initial image resolution.

  Parameters:
    `img_arr`. Shape (H, W, ...). Input image. If there is a channel axis, then the color system is BGR.

  Returns:
    `img_arr_out`. Shape (H, W, ...).
  '''

  H, W = img_arr.shape[:2]

  H_subimage_nrm = np.random.uniform(low=0.8, high=1)

  # Aspect ratio is limited close to 1:1; that is why only a small 10% deviation is added to subimage height.
  W_subimage_nrm = float(np.clip(a=H_subimage_nrm + np.random.uniform(low=-0.1, high=0.1), a_min=0, a_max=1))

  y_u_nrm = np.random.uniform(low=0, high=1-H_subimage_nrm)
  x_l_nrm = np.random.uniform(low=0, high=1-W_subimage_nrm)
  y_l_nrm = min(y_u_nrm + H_subimage_nrm, 1)
  x_r_nrm = min(x_l_nrm + W_subimage_nrm, 1)

  y_u = int(y_u_nrm*H) # Upper (u) height coordinate
  x_l = int(x_l_nrm*W) # Left (l) width coordinate
  y_l = int(y_l_nrm*H) # Lower (l) height coordinate
  x_r = int(x_r_nrm*W) # Right (r) width coordinate

  img_arr_out = img_arr[y_u:y_l, x_l:x_r]
  img_arr_out = cv2.resize(src=img_arr_out, dsize=(W, H))

  return img_arr_out

def apply_random_jitter(img_arr_nrm_bgr: NDArray) -> NDArray:
  '''
  Description:
    Performs random jitter distortion to a normalized input image where the pixel intensity value belongs to [0, 1].

  Parameters:
    `img_arr_nrm_bgr`. Shape (H, W, ...). Pixels adhere to the BGR color model.

  Returns:
    `img_arr_nrm_bgr_out`. Shape (H, W, ...).
  '''

  def apply_intensity_distortion(img_arr_nrm: NDArray, beta: float) -> NDArray:
    '''
    Description:
      Performs intensity distortion by adding an intensity offset value for either darkening (beta<0) or brightening (beta>0) of pixel intensity values.

    Parameters:
      `img_arr_nrm_bgr`. Shape (H, W, ...). Pixels adhere to the BGR color model.
      `beta`. Belongs to the interval [-0.2, 0.2]. Intensity offset.

    Returns:
      `img_arr_nrm_bgr_out`. Shape (H, W, ...).
    '''

    assert -.2 <= beta <= .2, "E: Disallowed beta argument detected."

    img_arr_nrm_out = img_arr_nrm + beta

    return np.clip(a=img_arr_nrm_out, a_min=0, a_max=1)

  def apply_contrast_distortion(img_arr_nrm: NDArray, alpha: float) -> NDArray:
    '''
    Description:
      Performs contrast distortion based on a  pixel intensity distortion parameter. In other words it either shrinks (alpha<1) or expands (alpha>1) the image's intensity histogram depending on the alpha parameter. The shrinking/expansion is applied with respect to the mean intensity value.

    Parameters:
      `img_arr_nrm`. Shape (H, W, ...).
      `alpha`. Belongs to the interval [0.8, 1.2]. Contrast distortion scaling factor.

    Returns:
      `img_arr_nrm_out`. Shape (H, W, ...).
    '''

    assert .8 <= alpha <= 1.2, "E: Disallowed alpha argument detected."

    mean = np.mean(img_arr_nrm)
    img_arr_nrm_out = alpha * (img_arr_nrm - mean) + mean

    return np.clip(a=img_arr_nrm_out, a_min=0, a_max=1)

  def apply_saturation_distortion(img_arr_nrm_bgr: NDArray, alpha: float) -> NDArray:
    '''
    Description:
      Modifies image saturation based on a pixel color distortion parameter. Either increases (alpha<1) or decreases (alpha>1) the pixelwise diversification of RGB color intensities either corresponding to more vivid or duller colors respectively.

    Parameters:
      `img_arr_nrm_bgr`. Shape (H, W, 3). Pixels adhere to the BGR color model.
      `alpha`. Belongs to the interval [0.8, 1.2]. Color saturation distortion factor.

    Returns:
      `img_arr_nrm_bgr_out`. Shape (H, W, 3). Pixels adhere to the BGR color model.
    '''

    assert .8 <= alpha <= 1.2, "E: Disallowed alpha argument detected."

    img_arr_nrm_gs = bgr2gs(img_arr_nrm_bgr)
    img_arr_nrm_bgr_out = alpha * img_arr_nrm_bgr + (1-alpha) * img_arr_nrm_gs
    img_arr_nrm_bgr_out = np.clip(a=img_arr_nrm_bgr_out, a_min=0, a_max=1)

    return img_arr_nrm_bgr_out

  def apply_hue_distortion(img_arr_nrm_bgr: NDArray, delta: int) -> NDArray:
    '''
    Description:
      Modifies image hue based on a hue distortion offset parameter delta. By modifying the hue value, the perceived color is modified. OpenCV's hue value belongs to the interval [0, 180), with direction Red -> Green -> Blue (delta > 0). For example, for delta>0, green colored pixels are pushed towards more blue colors.

    Parameters:
      `img_arr_nrm_bgr`. Shape (H, W, 3). Pixels adhere to the BGR color model.
      `delta`. Belongs to the interval [-0.1*180, 0.1*180]. Hue distortion offset.

    Returns:
      `img_arr_nrm_bgr_out`. Shape (H, W, 3). Pixels adhere to the BGR color model.
    '''

    assert -.1*180 <= delta <= .1*180, "E: Disallowed delta argument detected."

    # Pixel: (blue, green, red) -> (hue, saturation, value)
    img_arr_nrm_hsv = cv2.cvtColor(src=img_arr_nrm_bgr, code=cv2.COLOR_BGR2HSV)
    img_arr_nrm_hsv[..., 0] = (img_arr_nrm_hsv[..., 0] + delta) % 180
    img_arr_nrm_bgr_out = cv2.cvtColor(src=img_arr_nrm_hsv, code=cv2.COLOR_HSV2BGR)

    return img_arr_nrm_bgr_out

  img_arr_nrm_bgr_out = deepcopy(img_arr_nrm_bgr)

  intensity_beta = np.random.uniform(low=-0.2, high=0.2)
  contrast_alpha = np.random.uniform(low=0.8, high=1.2)
  saturation_alpha = np.random.uniform(low=0.8, high=1.2)
  hue_delta = np.random.uniform(low=-0.1*180, high=0.1*180)

  img_arr_nrm_bgr_out = apply_intensity_distortion(img_arr_nrm=img_arr_nrm_bgr_out, beta=intensity_beta)
  img_arr_nrm_bgr_out = apply_contrast_distortion(img_arr_nrm=img_arr_nrm_bgr_out, alpha=contrast_alpha)
  img_arr_nrm_bgr_out = apply_saturation_distortion(img_arr_nrm_bgr=img_arr_nrm_bgr_out, alpha=saturation_alpha)
  img_arr_nrm_bgr_out = apply_hue_distortion(img_arr_nrm_bgr=img_arr_nrm_bgr_out, delta=hue_delta)

  return img_arr_nrm_bgr_out

def apply_random_color_drop(img_arr_nrm_bgr: NDArray) -> NDArray:
  '''
  Description:
    Substitutes a random combination of the BGR channels with a single value based on the BGR to grayscale transformation.

  Parameters:
    `img_arr_nrm_bgr`. Shape (H, W, 3). Pixels adhere to the BGR color model.

  Returns:
    `img_arr_nrm_out`. Shape (H, W, 3).
  '''

  b_weight = 0.114
  g_weight = 0.587
  r_weight = 0.299

  # [0, 0.8) no color dropping; [0.8, 0.95) 2-channel merging; [0.95, 1] complete color drop.
  color_drop_configuration = np.random.random()

  if color_drop_configuration < 0.8:
    img_arr_nrm_out = img_arr_nrm_bgr
  elif color_drop_configuration < 0.95:
    if color_drop_configuration < 0.85: # Merge blue and green
      bg_merged_slice = b_weight/(b_weight+g_weight)*img_arr_nrm_bgr[..., 0] + g_weight/(b_weight+g_weight)*img_arr_nrm_bgr[..., 1]
      img_arr_nrm_out = np.stack([bg_merged_slice, bg_merged_slice, img_arr_nrm_bgr[..., 2]], axis=-1)
    elif color_drop_configuration < 0.9: # Merge blue and red
      br_merged_slice = b_weight/(b_weight+r_weight)*img_arr_nrm_bgr[..., 0] + r_weight/(b_weight+r_weight)*img_arr_nrm_bgr[..., 2]
      img_arr_nrm_out = np.stack([br_merged_slice, img_arr_nrm_bgr[..., 1], br_merged_slice], axis=-1)
    else: # Merge green and red
      gr_merged_slice = g_weight/(g_weight+r_weight)*img_arr_nrm_bgr[..., 1] + r_weight/(g_weight+r_weight)*img_arr_nrm_bgr[..., 2]
      img_arr_nrm_out = np.stack([img_arr_nrm_bgr[..., 0], gr_merged_slice, gr_merged_slice], axis=-1)
  else:
    bgr_merged_slice = b_weight*img_arr_nrm_bgr[..., 0] + g_weight*img_arr_nrm_bgr[..., 1] + r_weight*img_arr_nrm_bgr[..., 2]
    img_arr_nrm_out = np.stack([bgr_merged_slice, bgr_merged_slice, bgr_merged_slice], axis=-1)

  return img_arr_nrm_out



