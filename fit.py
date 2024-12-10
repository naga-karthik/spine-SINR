"""Run model inference and save outputs for analysis"""
import os
import hydra
from omegaconf import DictConfig, omegaconf, OmegaConf
import wandb
from loguru import logger

import torch
from torch.utils.data import DataLoader

from models import models
from data.datasets import BrainMRInterSubj3D
from utils.image_io import save_nifti
from utils.misc import setup_dir

import random
random.seed(7)

def custom_collate(batch):
    batch_data = {}
    for key in batch[0]:
        if isinstance(batch[0][key], dict):
            batch_data[key] = {subkey: [item[key][subkey] for item in batch] for subkey in batch[0][key]}
        else:
            batch_data[key] = torch.stack([item[key] for item in batch], dim=0)
    return batch_data

def get_inference_dataloader(cfg, pin_memory=False):
    logger.info(f'Loading dataset: {cfg.data.name}')
    # if cfg.data.name == 'brain_camcan':
    if cfg.data.name in ['brain_camcan', 'spine_generic']:
        dataset = BrainMRInterSubj3D(data_dir_path=cfg.data.data_dir_path,
                                        crop_size=cfg.data.crop_size,
                                        modality=cfg.data.modality,
                                        atlas_path=cfg.data.atlas_path,
                                        # voxel_size=cfg.data.voxel_size,
                                        evaluate=True)
    else:
        raise ValueError(f'Dataset config ({cfg.data}) not recognised.')
    return DataLoader(dataset, shuffle=False, collate_fn=custom_collate, 
                      pin_memory=pin_memory, batch_size=cfg.data.batch_size, 
                      num_workers=cfg.data.num_workers)


def set_up_kwargs(cfg):
    kwargs = {}
    kwargs["epochs"] = cfg.training.epochs
    kwargs["affine_epochs"] = cfg.training.affine_epochs
    kwargs["batch_size"] = cfg.data.batch_size
    kwargs["coords_batch_size"] = cfg.data.coords_batch_size
    logger.info(f'Coodinates batch size: {cfg.data.coords_batch_size}')
    kwargs["verbose"] = cfg.verbose

    if cfg.regularization.type == 'bending':
        kwargs["bending_regularization"] = True
        kwargs["hyper_regularization"] = False
        kwargs["jacobian_regularization"] = False
        kwargs["alpha_bending"] = cfg.regularization.alpha_bending
        logger.info(f'Regularization: {cfg.regularization.type}')
        logger.info(f'kwargs["hyper_regularization"]: {kwargs["hyper_regularization"]}')
        logger.info(f'LambdaReg: {cfg.regularization.alpha_bending}')

    kwargs["registration_type"] = cfg.network.type
    kwargs["network_type"] = cfg.network.activation
    kwargs["factor"] = cfg.network.factor
    kwargs["scale"] = cfg.network.scale
    kwargs["mapping_size"] = cfg.network.mapping_size
    kwargs["positional_encoding"] = cfg.network.positional_encoding

    kwargs["log_interval"] = cfg.training.log_interval
    kwargs["deformable_layers"] = cfg.network.deformable_layers
    kwargs["def_lr"] = cfg.training.def_lr
    kwargs["omega"] = cfg.network.omega
    kwargs["save_folder"] = setup_dir(os.getcwd() + '/logs')
    
    kwargs["loss_function"] = cfg.loss.type
    logger.info(f'Loss function: {cfg.loss.type}')
    
    kwargs["use_mask"] = cfg.data.use_mask
    kwargs["image_shape"] = cfg.data.crop_size
    # kwargs["voxel_size"] = cfg.data.voxel_size
    
    kwargs["transformation_type"] = cfg.transformation.type
    logger.info(f'Transformation type: {cfg.transformation.type}')
    
    kwargs["cps"] = cfg.transformation.config.cps
    logger.info(f'Control points: {cfg.transformation.config.cps}')

    wandb.config = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    wandb.init(
        entity=cfg.wandb_cfg.setup.entity, 
        project=f"sinr_{cfg.data.name.split('_')[0]}_{cfg.data.modality}",
        config=wandb.config,
        # name=run_name,
    )

    return kwargs


def inr_inference(cfg, dataloader=None, out_dir='', device=torch.device('cpu'), save=False):

    # create logs.txt in output directory
    logger.add(out_dir + '/logs.txt', rotation='10 MB', level="INFO")

    # setup arguments for the model
    kwargs = set_up_kwargs(cfg)

    for idx, batch_dict in enumerate(dataloader):
        logger.info(f'\nSubject: {idx+1}/{len(dataloader)} \n')
        # for all keys in batch, split the batch into two dicts, one for images and one for metadata
        batch = {k: v['img'][0] for k, v in batch_dict.items()}
        metadata = {k: {'affine': v['affine'][0], 'header': v['header'][0]} for k, v in batch_dict.items()}

        for k, x in batch.items():
            # reshape data for inference
            # 3d: (N=1, 1, H, W, D) -> (1, N=1, H, W, D)
            batch[k] = x.to(device=device).squeeze()

        ImpReg = models.ImplicitRegistrator(batch, idx, **kwargs)
        ImpReg.fit()

        # save the outputs
        subj_id = dataloader.dataset.subject_list[idx]
        output_id_dir = setup_dir(out_dir + f'/{subj_id}')
        for k, x in ImpReg.batch.items():
            x = x.detach().cpu().numpy()
            x = x.squeeze()
            if k in ["fixed", "fixed_mask", "fixed_seg"]:
                save_nifti(x, path=output_id_dir + f'/{k}.nii.gz', nim=metadata["fixed"], verbose=True)
            elif k in ["moving", "moving_mask", "moving_seg"]:
                save_nifti(x, path=output_id_dir + f'/{k}.nii.gz', nim=metadata["moving"], verbose=True)
            elif k in ["warped_moving", "warped_seg", "disp_pred"]:
                save_nifti(x, path=output_id_dir + f'/{k}.nii.gz', nim=metadata["fixed"], verbose=True)
            
            # save_nifti(x, path=output_id_dir + f'/{k}.nii.gz', verbose=True)
            
        torch.save(ImpReg.deform_net, f'{kwargs["save_folder"]}/{subj_id}_deform_net.pl')




def check_set_gpu(override=None):
    if override is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
            print(f"Using MPS: {torch.backends.mps.is_available()}")
        else:
            device = torch.device('cpu')
            print(f"Using CPU: {torch.device('cpu')}")
    else:
        device = torch.device(override)
    return device


@hydra.main(version_base=None, config_path="conf/", config_name="config")
def main(cfg: DictConfig) -> None:

    # os.environ['WANDB_DISABLED'] = cfg.wandb_disable

    # configure GPU
    gpu = cfg.gpu
    if gpu is not None and isinstance(gpu, int):
        if not cfg.slurm:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    # configure dataset & model
    dataloader = get_inference_dataloader(cfg, pin_memory=(device is torch.device('cuda')))

    # run inference
    if not cfg.analyse_only:
        output_dir = setup_dir(os.getcwd() + '/outputs')  # cwd = hydra.run.dir
        inr_inference(cfg, dataloader, output_dir, device=device)
    else:
        # TODO: add path
        output_dir = ''



if __name__ == '__main__':
    main()
