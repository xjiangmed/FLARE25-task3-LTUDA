#!/usr/bin/env python
# ref: https://github.com/JunMa11/FLARE/blob/aad2cc2813d11135d014bf578d4f62cea84ab865/FLARE23/FLARE23_DSC_NSD_Eval.py#L4

import sys
import os
import nibabel as nb
import numpy as np
import glob
import gc
from collections import OrderedDict
from engines.FLARE_EVAL.SurfaceDice import compute_surface_distances, compute_surface_dice_at_tolerance, compute_dice_coefficient
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.common_function import parse_option, is_directory_only_symlinks

def find_lower_upper_zbound(organ_mask):
    """
    Parameters
    ----------
    seg : TYPE
        DESCRIPTION.
    Returns
    -------
    z_lower: lower bound in z axis: int
    z_upper: upper bound in z axis: int
    """
    organ_mask = np.uint8(organ_mask)
    assert np.max(organ_mask) == 1, print('mask label error!')
    z_index = np.where(organ_mask > 0)[2]
    z_lower = np.min(z_index)
    z_upper = np.max(z_index)

    return z_lower, z_upper

if __name__ == '__main__':
    # get config
    _, config = parse_option("other")
    # Check input directories.
    submit_dir = config.VAL_OUTPUT_PATH
    truth_dir = os.path.join(config.DATASET.BASE_DIR, config.DATASET.VAL_MASK_PATH)
    if not os.path.isdir(submit_dir):
        print("submit_dir {} doesn't exist".format(submit_dir))
        sys.exit()
    if not os.path.isdir(truth_dir):
        print("truth_dir {} doesn't exist".format(truth_dir))
        sys.exit()

    # Create output directory.
    output_dir = config.VAL.EVAL_OUTPUT_RESULT
    os.makedirs(output_dir, exist_ok=True)

    # -------------------------- flare metrics
    seg_metrics = OrderedDict()
    seg_metrics['Name'] = list()

    label_tolerance = OrderedDict({'Liver': 5, 'RK': 3, 'Spleen': 3, 'LK': 3})
    label_dict = OrderedDict({'Liver': 1, 'RK': 2, 'Spleen': 3, 'LK': 4})
    sub_dict =  OrderedDict({'Liver': 1, 'RK': 2, 'Spleen': 3, 'LK': 13})
    for organ in label_tolerance.keys():
        seg_metrics['{}_DSC'.format(organ)] = list()
    for organ in label_tolerance.keys():
        seg_metrics['{}_NSD'.format(organ)] = list()
    # -------------------------- flare metrics

    # Iterate over all volumes in the reference list.
    reference_volume_list = sorted(glob.glob(truth_dir + '/*.nii.gz'))
    for reference_volume_fn in reference_volume_list:
        print("Starting with volume {}".format(reference_volume_fn))
        submission_volume_path = os.path.join(submit_dir, os.path.basename(reference_volume_fn))
        if not os.path.exists(submission_volume_path):
            raise ValueError("Submission volume not found - terminating!\n"
                             "Missing volume: {}".format(submission_volume_path))
        print("Found corresponding submission file {} for reference file {}"
              "".format(reference_volume_fn, submission_volume_path))
        print('-'*50)

        # Load reference and submission volumes with Nibabel.
        reference_volume = nb.load(reference_volume_fn)
        submission_volume = nb.load(submission_volume_path)

        # Get the current voxel spacing.
        voxel_spacing = reference_volume.header.get_zooms()[:3]

        # Get Numpy dataloader and compress to int8.
        reference_volume = (reference_volume.get_fdata()).astype(np.int8)
        submission_volume = (submission_volume.get_fdata()).astype(np.int8)

        # Ensure that the shapes of the masks match.
        if submission_volume.shape != reference_volume.shape:
            raise AttributeError("Shapes do not match! Prediction mask {}, "
                                 "ground truth mask {}"
                                 "".format(submission_volume.shape,
                                           reference_volume.shape))

        # ----------------------- flare metrics
        seg_metrics['Name'].append(os.path.basename(reference_volume_fn))
        for i, organ in enumerate(label_tolerance.keys(), 1):
            # reference_volum == 1,2,3,13
            print('organ, label_dict[organ], sub_dict[organ]:', organ, label_dict[organ], sub_dict[organ])
            if np.sum(reference_volume == label_dict[organ]) == 0 and np.sum(submission_volume == sub_dict[organ]) == 0:
                DSC_i = 1
                NSD_i = 1
            elif np.sum(reference_volume == label_dict[organ]) == 0 and np.sum(submission_volume == sub_dict[organ]) > 0:
                DSC_i = 0
                NSD_i = 0
            elif np.sum(reference_volume == label_dict[organ]) > 0 and np.sum(submission_volume == sub_dict[organ]) == 0:
                DSC_i = 0
                NSD_i = 0
            else:
                organ_i_gt, organ_i_seg = reference_volume == label_dict[organ], submission_volume == sub_dict[organ]
                DSC_i = compute_dice_coefficient(organ_i_gt, organ_i_seg)
                if DSC_i < 0.2:
                    NSD_i = 0
                else:
                    surface_distances = compute_surface_distances(organ_i_gt, organ_i_seg, voxel_spacing)
                    NSD_i = compute_surface_dice_at_tolerance(surface_distances, label_tolerance[organ])
                    
            seg_metrics['{}_DSC'.format(organ)].append(round(DSC_i, 4))
            seg_metrics['{}_NSD'.format(organ)].append(round(NSD_i, 4))
        gc.collect()

    overall_metrics = {}
    for key, value in seg_metrics.items():
        if 'Name' not in key:
            overall_metrics[key] = round(np.mean(value), 4)
            overall_metrics[key+'_std'] = round(np.std(value), 4)

    organ_dsc = []
    organ_nsd = []
    for key, value in overall_metrics.items():
        if 'Lesion' not in key:
            if 'DSC' in key and '_std' not in key:
                organ_dsc.append(value)
            if 'NSD' in key and '_std' not in key:
                organ_nsd.append(value)
    overall_metrics['Organ_DSC'] = round(np.mean(organ_dsc), 4)
    overall_metrics['Organ_DSC_std'] = round(np.std(organ_dsc), 4)
    overall_metrics['Organ_NSD'] = round(np.mean(organ_nsd), 4)
    overall_metrics['Organ_NSD_std'] = round(np.std(organ_nsd), 4)


    print("Computed metrics:")
    for key, value in overall_metrics.items():
        print("{}: {:.4f}".format(key, float(value)))

    # Write metrics to file.
    output_filename = os.path.join(output_dir, 'scores.txt')
    output_file = open(output_filename, 'w')
    for key, value in overall_metrics.items():
        output_file.write("{}: {:.4f}\n".format(key, float(value)))
    output_file.close()