import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import shutil
import traceback
from collections import OrderedDict
from datetime import datetime
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from batchgenerators.utilities.file_and_folder_operations import *
from filter_plabel import process_fake_labels, static_gt_each_labels_num
from preprocess_flare24_real_mr_dataset import (preprocess_amos_datset_plabel,
                                                preprocess_lld_datset_plabel,
                                                process_lld_data_img)
from regis_batch import regis_data
from skimage.transform import resize

from engines.common import Inference
from engines.dataloader.utils import (change_axes_of_image,
                                      clip_and_normalize_mean_std,
                                      create_two_class_mask,
                                      crop_image_according_to_mask, load_data,
                                      resize_segmentation)
from utils.common_function import (crop_bbox_by_stand_spacing, load_checkpoint,
                                   parse_option)


def run_prepare_data(config, is_multiprocessing=True):
    data_prepare = model_train_data_process(config)
    if is_multiprocessing:
        pool = Pool(int(cpu_count() * 0.3))
        for data in data_prepare.data_list:
            try:
                pool.apply_async(data_prepare.process, (data,))
            except Exception as err:
                traceback.print_exc()
                print(
                    "Create image/label throws exception %s, with series_id %s!"
                    % (err, data_prepare.data_info)
                )

        pool.close()
        pool.join()
    else:
        for data in data_prepare.data_list:
            data_prepare.process(data)


class model_train_data_process(object):
    def __init__(self, config):
        self.config = config
        self.train_type = self.config.TRAINING_TYPE
        print(self.train_type)
        self.coarse_size = self.config.DATASET.COARSE.PREPROCESS_SIZE
        self.fine_size = self.config.DATASET.FINE.PREPROCESS_SIZE
        self.nor_dir = self.config.DATASET.IS_NORMALIZATION_DIRECTION
        self.extend_size = self.config.DATASET.EXTEND_SIZE
 
        self.image_path = os.path.join(
            config.DATASET.BASE_DIR, config.DATASET.TRAIN_IMAGE_PATH
        )
        # self.mask_path = os.path.join(
        #     config.DATASET.BASE_DIR, config.DATASET.TRAIN_MASK_PATH
        # )
        self.preprocess_coarse_path = os.path.join(
            config.DATASET.BASE_DIR, config.DATASET.COARSE.PROPRECESS_PATH
        )
        self.preprocess_fine_path = os.path.join(
            config.DATASET.BASE_DIR, config.DATASET.FINE.PROPRECESS_PATH
        )
        self.data_list = os.listdir(self.image_path)
        os.makedirs(self.preprocess_coarse_path, exist_ok=True)
        os.makedirs(self.preprocess_fine_path, exist_ok=True)
        self.is_abdomen_crop = config.DATASET.IS_ABDOMEN_CROP

    def process(self, image_id):
        is_softmax_exists = False
        data_id = image_id.split("_0000.nii.gz")[0]
        image, image_spacing, image_direction, image_itk_info = load_data(
            join(self.image_path, data_id + "_0000.nii.gz"))
        # mask, _, mask_direction, label_itk_info = load_data(
        #     join(self.mask_path, data_id + ".nii.gz")
        # )
        # # if softmax matrix exists
        # if os.path.exists(os.path.join(self.mask_path, data_id + "_softmax.npy")):
        #     softmax_image = np.load(
        #         os.path.join(self.mask_path, data_id + "_softmax.npy")
        #     )
        #     is_softmax_exists = True
        # assert image_direction.all() == mask_direction.all()
        if self.is_abdomen_crop:
            (
                image,
                crop_start_slice,
                crop_end_slice,
                original_start,
                original_end,
            ) = crop_bbox_by_stand_spacing(image, image_spacing)
            # mask, _, _, _, _ = crop_bbox_by_stand_spacing(mask, image_spacing)
        #
        image = image.transpose(1, 2, 0)
        # mask = mask.transpose(1, 2, 0)
        if is_softmax_exists:
            softmax_image = softmax_image.transpose(0, 2, 3, 1)

        if self.nor_dir:
            image = change_axes_of_image(image, image_direction)
            # mask = change_axes_of_image(mask, mask_direction)
            # if is_softmax_exists:
            #     for s_i in range(softmax_image.shape[0]):
            #         softmax_image[s_i] = change_axes_of_image(
            #             softmax_image[s_i], mask_direction
            #         )
        
        crop_image = image
        # crop_mask = mask
        if is_softmax_exists:
            crop_softmax_image = softmax_image

        if "fine" in self.train_type:
            data_info_crop = OrderedDict()
            data_info_crop["raw_shape"] = image.shape
            data_info_crop["crop_shape"] = crop_image.shape
            data_info_crop["raw_spacing"] = image_spacing
            resize_crop_spacing = image_spacing * crop_image.shape / self.fine_size
            data_info_crop["resize_crop_spacing"] = resize_crop_spacing
            data_info_crop["image_direction"] = image_direction
            with open(
                os.path.join(self.preprocess_fine_path, "%s_info.pkl" % data_id), "wb"
            ) as f:
                pickle.dump(data_info_crop, f)

            crop_image_resize = resize(
                crop_image, self.fine_size, order=3, mode="edge", anti_aliasing=False
            )
            # crop_mask_resize = resize_segmentation(crop_mask, self.fine_size, order=0)
            crop_image_normal = clip_and_normalize_mean_std(crop_image_resize)
            if is_softmax_exists:
                crop_softmax_resize = resize(
                    crop_softmax_image,
                    [crop_softmax_image.shape[0], *self.fine_size],
                    order=1,
                    mode="edge",
                    anti_aliasing=False,
                )
            else:
                if len(crop_image_resize.shape) == (len(self.fine_size) + 1):
                    crop_softmax_resize = np.zeros([*crop_image_resize.shape]).astype(
                        np.int8
                    )
                else:
                    crop_softmax_resize = np.zeros(
                        [1, *crop_image_resize.shape]
                    ).astype(np.int8)

            np.savez_compressed(
                os.path.join(self.preprocess_fine_path, "%s.npz" % data_id),
                data=crop_image_normal[None, ...],
                # seg=crop_mask_resize[None, ...],
                softmax_iamge=crop_softmax_resize[None, ...],
            )
            print("End processing %s." % data_id)


if __name__ == "__main__":
    _, config = parse_option("other")
    run_prepare_data(config, True)

