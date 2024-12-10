""" Dataset helper functions """
import numpy as np
import torch
from utils.image import crop_and_pad, normalise_intensity
from utils.image_io import load_nifti
import nibabel as nib


def _to_tensor(data_dict):
    # cast to Pytorch Tensor
    for name, data in data_dict.items():
        data_dict[name]['img'] = torch.from_numpy(data['img']).float()
    return data_dict


def _crop_and_pad(data_dict, crop_size):
    # cropping and padding
    for name, data in data_dict.items():
        # only crop-pad the full-sized fixed image to even size
        if 'fixed' in name:
            data_dict[name]['img'] = crop_and_pad(data['img'], new_size=crop_size['fixed'])
        else:
            data_dict[name]['img'] = crop_and_pad(data['img'], new_size=crop_size['moving'])
        # elif name == 'moving':
        #     data_dict[name]['img'] = crop_and_pad(data['img'], new_size=crop_size_moving)
        # else:
        #     pass    # do nothing

    return data_dict


def _normalise_intensity(data_dict, keys=None, vmin=0., vmax=1.):
    """ Normalise intensity of data in `data_dict` with `keys` """
    if keys is None:
        keys = {'fixed', 'moving', 'fixed_original'}

    # images in one pairing should be normalised using the same scaling
    vmin_in = min(np.min(np.array(data_dict['fixed']['img'])), np.min(np.array(data_dict['moving']['img'])))
    vmax_in = max(np.max(np.array(data_dict['fixed']['img'])), np.max(np.array(data_dict['moving']['img'])))    
    # vmin_in = np.amin(np.array([data_dict[k]['img'] for k in keys]))
    # vmax_in = np.amax(np.array([data_dict[k]['img'] for k in keys]))

    for name, data in data_dict.items():
        if name in keys:
            data_dict[name]['img'] = normalise_intensity(data['img'],
                                               min_in=vmin_in, max_in=vmax_in,
                                               min_out=vmin, max_out=vmax,
                                               mode="minmax", clip=True)
    return data_dict


def _load3d(data_path_dict):
    """
    Load 3D data and metadata from given paths.

    Parameters:
    - data_path_dict (dict): Dictionary with keys like 'fixed', 'moving', etc., and their corresponding file paths.

    Returns:
    - data_dict (dict): Dictionary containing image data and metadata.
    """
    data_dict = {}
    for key, path in data_path_dict.items():
        img = nib.load(path)
        data = img.get_fdata()
        data_dict[key] = {
            'img': data[np.newaxis, ...],
            'header': img.header,
            'affine': img.affine
        }
    return data_dict

