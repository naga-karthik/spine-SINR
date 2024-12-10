import os
import random
from torch.utils.data import Dataset
from data.utils import _load3d, _crop_and_pad, _normalise_intensity, _to_tensor
from utils.image_io import save_nifti
from loguru import logger


class _BaseDataset(Dataset):
    """Base dataset class"""
    def __init__(self, data_dir_path):
        super(_BaseDataset, self).__init__()
        self.data_dir = data_dir_path
        assert os.path.exists(data_dir_path), f"Data dir does not exist: {data_dir_path}"
        self.subject_list = sorted(os.listdir(self.data_dir))
        self.data_path_dict = dict()

    def _set_path(self, index):
        """ Set the paths of data files to load and the keys in data_dict"""
        raise NotImplementedError

    def __getitem__(self, index):
        """ Load data and pre-process """
        raise NotImplementedError

    def __len__(self):
        return len(self.subject_list)


class BrainMRInterSubj3D(_BaseDataset):
    def __init__(self,
                 data_dir_path,
                 crop_size,
                 evaluate=False,
                 modality='t1t1',
                 atlas_path=None):
        super(BrainMRInterSubj3D, self).__init__(data_dir_path)
        self.evaluate = evaluate
        self.crop_size = crop_size
        # self.voxel_size = voxel_size
        self.img_keys = ['fixed', 'moving']
        self.modality = modality
        self.atlas_path = atlas_path

    def _set_path(self, index):
        # choose the fixed and moving subjects/paths
        if self.atlas_path is None:
            self.tar_subj_id = self.subject_list[index]
            self.tar_subj_path = f'{self.data_dir}/{self.tar_subj_id}'
        else:
            self.tar_subj_path = self.atlas_path

        # random.choice can also pick the same target subject as the source but 
        # that is to also learn the identity transformation
        self.src_subj_id = self.subject_list[~self.subject_list.index(self.tar_subj_id)]    # always doing inter-subject
        self.src_subj_path = f'{self.data_dir}/{self.src_subj_id}'
        logger.info(f"Source path: {self.src_subj_path}")
        logger.info(f"Target path: {self.tar_subj_path}")

        # self.data_path_dict['fixed'] = f'{self.tar_subj_path}/{self.tar_subj_id}_T1w_norm_crop.nii.gz'
        self.data_path_dict['fixed'] = f'{self.tar_subj_path}/{self.tar_subj_id}_T1w_norm_cropped.nii.gz'
        # self.data_path_dict['fixed'] = f'{self.tar_subj_path}/T1_brain_norm.nii.gz'

        logger.info(f"Registering \t Moving: {self.src_subj_id} to Fixed: {self.tar_subj_id}")

        # Images
        if self.modality == 't1t1':
            # self.data_path_dict['moving'] = f'{self.src_subj_path}/T1_brain_norm_yshift.nii.gz'
            self.data_path_dict['moving'] = f'{self.src_subj_path}/{self.src_subj_id}_T1w_norm_cropped.nii.gz'
            # self.data_path_dict['moving'] = f'{self.src_subj_path}/{self.src_subj_id}_T1w_norm_cropped_2x2x2_yshift20.nii.gz'

        elif self.modality == 't1t2':
            # self.data_path_dict['moving'] = f'{self.src_subj_path}/T2_brain_norm_half.nii.gz'
            self.data_path_dict['moving'] = f'{self.src_subj_path}/{self.src_subj_id}_T2w_norm_crop_sc.nii.gz'

        else:
            raise ValueError(f'Modality ({self.modality}) not recognised.')

        # Masks
        if self.modality == 't1t1':
            # self.data_path_dict['fixed_mask'] = f'{self.tar_subj_path}/T1_brain_mask.nii.gz'
            # self.data_path_dict['moving_mask'] = f'{self.src_subj_path}/T1_brain_mask_2x2x2_half.nii.gz'
            self.data_path_dict['fixed_mask'] = f'{self.tar_subj_path}/{self.tar_subj_id}_T1w_mask_crop_sc.nii.gz'
            self.data_path_dict['moving_mask'] = f'{self.src_subj_path}/{self.src_subj_id}_T1w_mask_crop_sc.nii.gz'
            self.data_path_dict['fixed_seg'] = f'{self.tar_subj_path}/{self.tar_subj_id}_T1w_totalseg_cropped.nii.gz'
            self.data_path_dict['moving_seg'] = f'{self.src_subj_path}/{self.src_subj_id}_T1w_totalseg_cropped.nii.gz'
            # self.data_path_dict['moving_seg'] = f'{self.src_subj_path}/{self.src_subj_id}_T1w_totalseg_cropped_2x2x2_yshift20.nii.gz'

        elif self.modality == 't1t2':
            # # brain
            # self.data_path_dict['fixed_mask'] = f'{self.tar_subj_path}/T1_brain_mask.nii.gz'
            # self.data_path_dict['moving_mask'] = f'{self.src_subj_path}/T2_brain_mask_half.nii.gz'
            # spine
            self.data_path_dict['fixed_mask'] = f'{self.tar_subj_path}/{self.tar_subj_id}_T1w_mask_crop_sc.nii.gz'
            self.data_path_dict['moving_mask'] = f'{self.src_subj_path}/{self.src_subj_id}_T2w_mask_crop_sc.nii.gz'
            self.data_path_dict['fixed_seg'] = f'{self.tar_subj_path}/{self.tar_subj_id}_T1w_seg_crop_sc.nii.gz'
            self.data_path_dict['moving_seg'] = f'{self.src_subj_path}/{self.src_subj_id}_T2w_seg_crop_sc.nii.gz'

        else:
            raise ValueError(f'Modality ({self.modality}) not recognised.')

        # # brain
        # self.data_path_dict['fixed_seg'] = f'{self.tar_subj_path}/T1_brain_MALPEM_tissues.nii.gz'
        # self.data_path_dict['moving_seg'] = f'{self.src_subj_path}/T1_brain_MALPEM_tissues_2x2x2_half.nii.gz'

        logger.info(f"Fixed Image: {self.data_path_dict['fixed'].replace('/home/GRAMES.POLYMTL.CA/u114716/', '~/')}")
        logger.info(f"Moving Image: {self.data_path_dict['moving'].replace('/home/GRAMES.POLYMTL.CA/u114716/', '~/')}")
        logger.info(f"Fixed Seg: {self.data_path_dict['fixed_seg'].replace('/home/GRAMES.POLYMTL.CA/u114716/', '~/')}")
        logger.info(f"Moving Seg: {self.data_path_dict['moving_seg'].replace('/home/GRAMES.POLYMTL.CA/u114716/', '~/')}")
        # logger.info(f"Fixed Mask: {self.data_path_dict['fixed_mask'].replace('/home/GRAMES.POLYMTL.CA/u114716/', '~/')}")
        # logger.info(f"Moving Mask: {self.data_path_dict['moving_mask'].replace('/home/GRAMES.POLYMTL.CA/u114716/', '~/')}")

    def __getitem__(self, index):
        # output_dir = "/home/GRAMES.POLYMTL.CA/u114716/inrs-registration-superres/SINR/outputs"
        self._set_path(index)
        data_dict = _load3d(self.data_path_dict)
        data_dict = _crop_and_pad(data_dict, self.crop_size)
        # save_nifti(data_dict['moving'].squeeze(), f"{output_dir}/{self.src_subj_id}_moving_crop_pad.nii.gz")
        data_dict = _normalise_intensity(data_dict, self.img_keys)
        return _to_tensor(data_dict)

