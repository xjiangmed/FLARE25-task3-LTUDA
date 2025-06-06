import random

import numpy as np
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.color_transforms import (
    BrightnessMultiplicativeTransform, BrightnessTransform,
    ContrastAugmentationTransform, GammaTransform)
from batchgenerators.transforms.noise_transforms import (
    GaussianBlurTransform, GaussianNoiseTransform)
from batchgenerators.transforms.resample_transforms import \
    SimulateLowResolutionTransform
from batchgenerators.transforms.spatial_transforms import (MirrorTransform,
                                                           SpatialTransform)
from batchgenerators.transforms.utility_transforms import NumpyToTensor
from batchgenerators.utilities.file_and_folder_operations import *
from torch.utils.data import Dataset

from .data_augmentation import (DownsampleSegForDSTransform,
                                default_2D_augmentation_params,
                                default_3D_augmentation_params, get_patch_size)
import torch

class flare22_dataset(Dataset):
    def __init__(
        self,
        config,
        data_size,
        data_path,
        unlab_data_path,
        pool_op_kernel_sizes,
        num_each_epoch,
        is_train=True,
        is_deep_supervision=True,
    ):
        self.config = config
        self.is_select_import_voxel = config.TRAIN.SELECT_IMPORT_VOXEL.IS_OPEN
        self.data_path = data_path
        self.data_size = data_size
        self.unlab_data_path = unlab_data_path
        self.pool_op_kernel_sizes = pool_op_kernel_sizes
        self.num_each_epoch = num_each_epoch
        self.series_ids = [file for file in os.listdir(data_path) if file.endswith('.npz')]
        print('len(labeled_data):', len(self.series_ids))
        # self.series_ids = subfiles(data_path, join=False, suffix="npz") # 用这个读不到数据
        self.setup_DA_params()

        self.transforms = self.get_augmentation(
            data_size,
            self.data_aug_params,
            is_train=is_train,
            deep_supervision_scales=self.deep_supervision_scales
            if is_deep_supervision
            else None,
        )
        self.transforms_mr = self.get_augmentation(
            data_size,
            self.data_aug_params,
            is_train=is_train,
            deep_supervision_scales=None,
        )

    def __getitem__(self, idx):
        data_id = self.series_ids[random.randint(0, len(self.series_ids) - 1)]
        data_load_ori = np.load(join(self.data_path, data_id))
        if self.is_select_import_voxel:
            if "softmax_iamge" in data_load_ori.keys() and data_load_ori['softmax_iamge'].max(axis=1).any() > 0:
                seg_gt = data_load_ori["seg"]
                ignore_mask = data_load_ori['softmax_iamge'].max(axis=1) < 0.85
                seg_gt[ignore_mask] = self.config.TRAIN.SELECT_IMPORT_VOXEL.IGNORE_LABEL
                data_load_ori = {
                    "data": data_load_ori["data"].transpose(0, 3, 1, 2),
                    "seg": seg_gt.transpose(0, 3, 1, 2),
                }
            else:
                data_load_ori = {
                    "data": data_load_ori["data"].transpose(0, 3, 1, 2),
                    "seg": data_load_ori["seg"].transpose(0, 3, 1, 2),
                }
        else:
            data_load_ori = {
                "data": data_load_ori["data"].transpose(0, 3, 1, 2),
                "seg": data_load_ori["seg"].transpose(0, 3, 1, 2),
            }
        data_trans_ori = self.transforms(**data_load_ori)
        # 检查转换后是否正常

        data_infos = {
            "modality": int(data_id.split(".")[0].split("_")[-1]),
        }
        flag = 1
        
        
        assert not torch.isnan(data_trans_ori["data"]).any(), str(data_id)+" 输入数据包含NaN！"
        assert not torch.isinf(data_trans_ori["data"]).any(), str(data_id)+" 输入数据包含Inf！"
        assert not torch.isnan(data_trans_ori["seg"][0]).any(), str(data_id)+" 标签数据包含NaN！"
        assert not torch.isinf(data_trans_ori["seg"][0]).any(), str(data_id)+" 标签数据包含Inf！"
        
        return data_trans_ori, flag, data_infos

    def __len__(self):
        # return self.num_each_epoch
        return len(self.series_ids) # for large-scale data debugging!!

    def setup_DA_params(self):
        if self.config.MODEL.DEEP_SUPERVISION:
            self.deep_supervision_scales = [[1, 1, 1]] + list(
                list(i)
                for i in 1 / np.cumprod(np.vstack(self.pool_op_kernel_sizes), axis=0)
            )[:-1]
        self.data_aug_params = default_3D_augmentation_params
        self.data_aug_params["rotation_x"] = (
            -30.0 / 360 * 2.0 * np.pi,
            30.0 / 360 * 2.0 * np.pi,
        )
        self.data_aug_params["rotation_y"] = (
            -30.0 / 360 * 2.0 * np.pi,
            30.0 / 360 * 2.0 * np.pi,
        )
        self.data_aug_params["rotation_z"] = (
            -30.0 / 360 * 2.0 * np.pi,
            30.0 / 360 * 2.0 * np.pi,
        )

        if self.config.DATASET.DA.DO_2D_AUG:
            if self.config.DATASET.DA.DO_ELASTIC:
                self.data_aug_params[
                    "elastic_deform_alpha"
                ] = default_2D_augmentation_params["elastic_deform_alpha"]
                self.data_aug_params[
                    "elastic_deform_sigma"
                ] = default_2D_augmentation_params["elastic_deform_sigma"]
                self.data_aug_params["rotation_x"] = default_2D_augmentation_params[
                    "rotation_x"
                ]

        if self.config.DATASET.DA.DO_2D_AUG:
            self.basic_generator_patch_size = get_patch_size(
                self.data_size[1:],
                self.data_aug_params["rotation_x"],
                self.data_aug_params["rotation_y"],
                self.data_aug_params["rotation_z"],
                self.data_aug_params["scale_range"],
            )
            self.basic_generator_patch_size = np.array(
                [self.data_size[0]] + list(self.basic_generator_patch_size)
            )
        else:
            self.basic_generator_patch_size = get_patch_size(
                self.data_size,
                self.data_aug_params["rotation_x"],
                self.data_aug_params["rotation_y"],
                self.data_aug_params["rotation_z"],
                self.data_aug_params["scale_range"],
            )

    def get_augmentation(
        self,
        patch_size,
        params=default_3D_augmentation_params,
        is_train=True,
        border_val_seg=-1,
        order_seg=1,
        order_data=3,
        deep_supervision_scales=None,
    ):
        transforms = []
        if is_train:
            if self.config.DATASET.DA.DO_2D_AUG:
                ignore_axes = (1,)

                patch_size_spatial = patch_size[1:]
            else:
                patch_size_spatial = patch_size
                ignore_axes = None

            transforms.append(
                SpatialTransform(
                    patch_size_spatial,
                    patch_center_dist_from_border=None,
                    do_elastic_deform=self.config.DATASET.DA.DO_ELASTIC,
                    alpha=params.get("elastic_deform_alpha"),
                    sigma=params.get("elastic_deform_sigma"),
                    do_rotation=self.config.DATASET.DA.DO_ROTATION,
                    angle_x=params.get("rotation_x"),
                    angle_y=params.get("rotation_y"),
                    angle_z=params.get("rotation_z"),
                    p_rot_per_axis=params.get("rotation_p_per_axis"),
                    do_scale=self.config.DATASET.DA.DO_SCALING,
                    scale=params.get("scale_range"),
                    border_mode_data=params.get("border_mode_data"),
                    border_cval_data=0,
                    order_data=order_data,
                    border_mode_seg="constant",
                    border_cval_seg=border_val_seg,
                    order_seg=order_seg,
                    random_crop=self.config.DATASET.DA.RANDOM_CROP,
                    p_el_per_sample=params.get("p_eldef"),
                    p_scale_per_sample=params.get("p_scale"),
                    p_rot_per_sample=params.get("p_rot"),
                    independent_scale_for_each_axis=params.get(
                        "independent_scale_factor_for_each_axis"
                    ),
                )
            )

            transforms.append(
                GaussianNoiseTransform(noise_variance=(0, 0.15), p_per_sample=0.1)
            )
            transforms.append(
                GaussianBlurTransform(
                    (0.5, 1.0),
                    different_sigma_per_channel=True,
                    p_per_sample=0.2,
                    p_per_channel=0.5,
                )
            )
            transforms.append(
                BrightnessMultiplicativeTransform(
                    multiplier_range=(0.75, 1.25), p_per_sample=0.15
                )
            )

            if self.config.DATASET.DA.DO_ADDITIVE_BRIGHTNESS:
                transforms.append(
                    BrightnessTransform(
                        params.get("additive_brightness_mu"),
                        params.get("additive_brightness_sigma"),
                        True,
                        p_per_sample=params.get("additive_brightness_p_per_sample"),
                        p_per_channel=params.get("additive_brightness_p_per_channel"),
                    )
                )

            transforms.append(
                ContrastAugmentationTransform((0.6, 1.5), p_per_sample=0.15)
            )
            transforms.append(
                SimulateLowResolutionTransform(
                    zoom_range=(0.5, 1),
                    per_channel=True,
                    p_per_channel=0.5,
                    order_downsample=0,
                    order_upsample=3,
                    p_per_sample=0.25,
                    ignore_axes=ignore_axes,
                )
            )
            transforms.append(
                GammaTransform(
                    params.get("gamma_range"),
                    True,
                    True,
                    retain_stats=params.get("gamma_retain_stats"),
                    p_per_sample=0.1,
                )
            )  # inverted gamma

            if self.config.DATASET.DA.DO_GAMMA:
                transforms.append(
                    GammaTransform(
                        params.get("gamma_range"),
                        False,
                        True,
                        retain_stats=params.get("gamma_retain_stats"),
                        p_per_sample=params["p_gamma"],
                    )
                )

            if self.config.DATASET.DA.DO_MIRROR:
                transforms.append(MirrorTransform(params.get("mirror_axes")))

        if deep_supervision_scales is not None:
            transforms.append(
                DownsampleSegForDSTransform(
                    deep_supervision_scales, 0, input_key="seg", output_key="seg"
                )
            )


        transforms.append(NumpyToTensor(["data", "seg"], "float"))

        transforms = Compose(transforms)

        return transforms


class flare22_dataset_lnul(Dataset):
    def __init__(
        self,
        config,
        data_size,
        data_path,
        unlab_data_path,
        pool_op_kernel_sizes,
        num_each_epoch,
        is_train=True,
        is_deep_supervision=True,
        mode='labeled',
    ):
        self.config = config
        self.is_select_import_voxel = config.TRAIN.SELECT_IMPORT_VOXEL.IS_OPEN
        self.data_path = data_path
        self.data_size = data_size
        self.unlab_data_path = unlab_data_path
        self.mode = mode
        self.pool_op_kernel_sizes = pool_op_kernel_sizes
        self.num_each_epoch = num_each_epoch
        if self.mode == 'labeled':
            self.series_ids = [file for file in os.listdir(data_path) if file.endswith('.npz')]
            print('len(labeled_data):', len(self.series_ids))
        elif self.mode == 'unlabeled':
            self.series_ids = [file for file in os.listdir(unlab_data_path) if file.endswith('.npz')]
            print('len(unlabeled_data):', len(self.series_ids))
        
        self.setup_DA_params()

        self.transforms = self.get_augmentation(
            data_size,
            self.data_aug_params,
            is_train=is_train,
            deep_supervision_scales=self.deep_supervision_scales
            if is_deep_supervision
            else None,
        )
        # self.transforms_mr = self.get_augmentation(
        self.transforms_ul = self.get_augmentation(
            data_size,
            self.data_aug_params,
            is_train=is_train,
            deep_supervision_scales=None,
        )

    def __getitem__(self, idx):
    
        if self.mode == 'labeled':
            data_id = self.series_ids[random.randint(0, len(self.series_ids) - 1)]
            data_load_ori = np.load(join(self.data_path, data_id))
            if self.is_select_import_voxel:
                if "softmax_iamge" in data_load_ori.keys() and data_load_ori['softmax_iamge'].max(axis=1).any() > 0:
                    seg_gt = data_load_ori["seg"]
                    ignore_mask = data_load_ori['softmax_iamge'].max(axis=1) < 0.85
                    seg_gt[ignore_mask] = self.config.TRAIN.SELECT_IMPORT_VOXEL.IGNORE_LABEL
                    data_load_ori = {
                        "data": data_load_ori["data"].transpose(0, 3, 1, 2),
                        "seg": seg_gt.transpose(0, 3, 1, 2),
                    }
                else:
                    data_load_ori = {
                        "data": data_load_ori["data"].transpose(0, 3, 1, 2),
                        "seg": data_load_ori["seg"].transpose(0, 3, 1, 2),
                    }
            else:
                data_load_ori = {
                    "data": data_load_ori["data"].transpose(0, 3, 1, 2),
                    "seg": data_load_ori["seg"].transpose(0, 3, 1, 2),
                }
            data_trans_ori = self.transforms(**data_load_ori)
            
            flag = 1
            
            data_infos = {
                "modality": int(data_id.split(".")[0].split("_")[-1]),
            }
            assert not torch.isnan(data_trans_ori["data"]).any(), str(data_id)+" 输入数据包含NaN！"
            assert not torch.isinf(data_trans_ori["data"]).any(), str(data_id)+" 输入数据包含Inf！"
            assert not torch.isnan(data_trans_ori["seg"][0]).any(), str(data_id)+" 标签数据包含NaN！"
            assert not torch.isinf(data_trans_ori["seg"][0]).any(), str(data_id)+" 标签数据包含Inf！"
            
        elif self.mode == 'unlabeled': #for mri and pet
            data_id = self.series_ids[random.randint(0, len(self.series_ids) - 1)]
            data_load_ori = np.load(join(self.unlab_data_path, data_id))
            data_load_ori = {
                "data": data_load_ori["data"].transpose(0, 3, 1, 2),
            }
            data_trans_ori = self.transforms_ul(**data_load_ori)
            
            flag = 0 #TODO:check 
            
            if ('amos' in data_id) or ('MR' in data_id):
                data_infos = {
                    "modality": 'mr',
                }
            elif ('fdg' in data_id) or ('psma' in data_id):
                data_infos = {
                    "modality": 'pet',
                }
        
        
        return data_trans_ori, flag, data_infos

    def __len__(self):
        # return self.num_each_epoch
        return len(self.series_ids)

    def setup_DA_params(self):
        if self.config.MODEL.DEEP_SUPERVISION:
            self.deep_supervision_scales = [[1, 1, 1]] + list(
                list(i)
                for i in 1 / np.cumprod(np.vstack(self.pool_op_kernel_sizes), axis=0)
            )[:-1]
        self.data_aug_params = default_3D_augmentation_params
        self.data_aug_params["rotation_x"] = (
            -30.0 / 360 * 2.0 * np.pi,
            30.0 / 360 * 2.0 * np.pi,
        )
        self.data_aug_params["rotation_y"] = (
            -30.0 / 360 * 2.0 * np.pi,
            30.0 / 360 * 2.0 * np.pi,
        )
        self.data_aug_params["rotation_z"] = (
            -30.0 / 360 * 2.0 * np.pi,
            30.0 / 360 * 2.0 * np.pi,
        )

        if self.config.DATASET.DA.DO_2D_AUG:
            if self.config.DATASET.DA.DO_ELASTIC:
                self.data_aug_params[
                    "elastic_deform_alpha"
                ] = default_2D_augmentation_params["elastic_deform_alpha"]
                self.data_aug_params[
                    "elastic_deform_sigma"
                ] = default_2D_augmentation_params["elastic_deform_sigma"]
                self.data_aug_params["rotation_x"] = default_2D_augmentation_params[
                    "rotation_x"
                ]

        if self.config.DATASET.DA.DO_2D_AUG:
            self.basic_generator_patch_size = get_patch_size(
                self.data_size[1:],
                self.data_aug_params["rotation_x"],
                self.data_aug_params["rotation_y"],
                self.data_aug_params["rotation_z"],
                self.data_aug_params["scale_range"],
            )
            self.basic_generator_patch_size = np.array(
                [self.data_size[0]] + list(self.basic_generator_patch_size)
            )
        else:
            self.basic_generator_patch_size = get_patch_size(
                self.data_size,
                self.data_aug_params["rotation_x"],
                self.data_aug_params["rotation_y"],
                self.data_aug_params["rotation_z"],
                self.data_aug_params["scale_range"],
            )

    def get_augmentation(
        self,
        patch_size,
        params=default_3D_augmentation_params,
        is_train=True,
        border_val_seg=-1,
        order_seg=1,
        order_data=3,
        deep_supervision_scales=None,
    ):
        transforms = []
        if is_train:
            if self.config.DATASET.DA.DO_2D_AUG:
                ignore_axes = (1,)

                patch_size_spatial = patch_size[1:]
            else:
                patch_size_spatial = patch_size
                ignore_axes = None

            transforms.append(
                SpatialTransform(
                    patch_size_spatial,
                    patch_center_dist_from_border=None,
                    do_elastic_deform=self.config.DATASET.DA.DO_ELASTIC,
                    alpha=params.get("elastic_deform_alpha"),
                    sigma=params.get("elastic_deform_sigma"),
                    do_rotation=self.config.DATASET.DA.DO_ROTATION,
                    angle_x=params.get("rotation_x"),
                    angle_y=params.get("rotation_y"),
                    angle_z=params.get("rotation_z"),
                    p_rot_per_axis=params.get("rotation_p_per_axis"),
                    do_scale=self.config.DATASET.DA.DO_SCALING,
                    scale=params.get("scale_range"),
                    border_mode_data=params.get("border_mode_data"),
                    border_cval_data=0,
                    order_data=order_data,
                    border_mode_seg="constant",
                    border_cval_seg=border_val_seg,
                    order_seg=order_seg,
                    random_crop=self.config.DATASET.DA.RANDOM_CROP,
                    p_el_per_sample=params.get("p_eldef"),
                    p_scale_per_sample=params.get("p_scale"),
                    p_rot_per_sample=params.get("p_rot"),
                    independent_scale_for_each_axis=params.get(
                        "independent_scale_factor_for_each_axis"
                    ),
                )
            )

            transforms.append(
                GaussianNoiseTransform(noise_variance=(0, 0.15), p_per_sample=0.1)
            )
            transforms.append(
                GaussianBlurTransform(
                    (0.5, 1.0),
                    different_sigma_per_channel=True,
                    p_per_sample=0.2,
                    p_per_channel=0.5,
                )
            )
            transforms.append(
                BrightnessMultiplicativeTransform(
                    multiplier_range=(0.75, 1.25), p_per_sample=0.15
                )
            )

            if self.config.DATASET.DA.DO_ADDITIVE_BRIGHTNESS:
                transforms.append(
                    BrightnessTransform(
                        params.get("additive_brightness_mu"),
                        params.get("additive_brightness_sigma"),
                        True,
                        p_per_sample=params.get("additive_brightness_p_per_sample"),
                        p_per_channel=params.get("additive_brightness_p_per_channel"),
                    )
                )

            transforms.append(
                ContrastAugmentationTransform((0.6, 1.5), p_per_sample=0.15)
            )
            transforms.append(
                SimulateLowResolutionTransform(
                    zoom_range=(0.5, 1),
                    per_channel=True,
                    p_per_channel=0.5,
                    order_downsample=0,
                    order_upsample=3,
                    p_per_sample=0.25,
                    ignore_axes=ignore_axes,
                )
            )
            transforms.append(
                GammaTransform(
                    params.get("gamma_range"),
                    True,
                    True,
                    retain_stats=params.get("gamma_retain_stats"),
                    p_per_sample=0.1,
                )
            )  # inverted gamma

            if self.config.DATASET.DA.DO_GAMMA:
                transforms.append(
                    GammaTransform(
                        params.get("gamma_range"),
                        False,
                        True,
                        retain_stats=params.get("gamma_retain_stats"),
                        p_per_sample=params["p_gamma"],
                    )
                )

            if self.config.DATASET.DA.DO_MIRROR:
                transforms.append(MirrorTransform(params.get("mirror_axes")))

        if deep_supervision_scales is not None:
            transforms.append(
                DownsampleSegForDSTransform(
                    deep_supervision_scales, 0, input_key="seg", output_key="seg"
                )
            )

        if deep_supervision_scales is not None:
            transforms.append(NumpyToTensor(["data", "seg"], "float"))
        else:
            transforms.append(NumpyToTensor(["data"], "float"))

        transforms = Compose(transforms)

        return transforms
    
