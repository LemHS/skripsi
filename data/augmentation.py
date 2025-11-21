from typing import Callable
from toolz import *
from toolz.curried import *
from toolz.curried.operator import *
from utils.definitions import *
from monai.transforms import (
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    ScaleIntensityRangePercentilesd,
    Spacingd,
    ToTensord,
    EnsureChannelFirstd,
    SelectItemsd,
    RandAffined,
    RandSimulateLowResolutiond,
    RandAdjustContrastd,
    RandGaussianSmoothd,
    RandGaussianNoised,
    RandScaleIntensityd,
    IdentityD,
    DivisiblePadD,
    ResizeWithPadOrCropd,
    Resized,
    SpatialPadD,
    RandCropByPosNegLabelD,
    CopyItemsD,
    RandGridDistortiond,
    ConcatItemsD,
)

from data.transforms.transform import MergeLabelD
from pathlib import Path
import torch

def flatten_list(lst):
    result = []
    for item in lst:
        if isinstance(item, list):
            result.extend(flatten_list(item))
        else:
            result.append(item)
    return result


def aug(
    spacing: list[float],
    local_roi_size: list[int],
    global_roi_size: list[int],
    intensity_scale: list[float],
    old_label_rule: list[str],
    new_label_rule: dict[str, list[str]] | None = None,
    global_divisible_k: list[int] = [32, 32, 48],  # 16*2 for h,w and # 16*3 for global
    orientation_mode: str = "RAS",
    rand_aug_type=["none", "heavy"][-1],
    do_deform: bool = True,  # grid deformation if the dataset has little training sample,
    vol: list[str] | str = VOL,
    lab: str = LAB,
    source: str = VOL,
) -> Callable[[dict[str, Path]], dict[str, torch.Tensor]]:
    
    patch_vol_key = PATCH + "_" + vol if isinstance(vol, str) else [PATCH + "_" + v for v in vol]
    patch_lab_key = PATCH + "_" + lab
    vol_lab_key = flatten_list([vol, lab])
    patch_vol_lab_key = flatten_list([patch_vol_key, patch_lab_key])

    pre_aug = [
        LoadImaged(keys=vol_lab_key),
        EnsureChannelFirstd(keys=vol_lab_key),
        Orientationd(keys=vol_lab_key, axcodes=orientation_mode),
        ScaleIntensityRanged(
            keys=vol,
            a_min=intensity_scale[0],
            a_max=intensity_scale[1],
            b_min=intensity_scale[2],
            b_max=intensity_scale[3],
            clip=True,
        ),
        CropForegroundd(keys=vol_lab_key, source_key=source),
        Spacingd(keys=vol_lab_key, pixdim=spacing, mode=tuple(["bilinear"] * (len(vol_lab_key) - 1) + ["nearest"])),
        SelectItemsd(vol_lab_key + [CASE_ID]),
        ############################################
        # PAD IF VOLUME IS SMALLLER THAN ROIs
        SpatialPadD(
            keys=[vol] if isinstance(vol, str) else vol,
            spatial_size=local_roi_size,
            mode=("constant",) if isinstance(vol, str) else ("constant",) * len(vol),
            value=-0.1,
        ),
        SpatialPadD(
            keys=[lab],
            spatial_size=local_roi_size,
            mode=("constant",),
            value=0,
        ),
        ToTensord(keys=vol_lab_key),
    ]

    if rand_aug_type == "heavy":
        # Closely follows nnunet's augmentation rule
        random_aug = [
            RandShiftIntensityd(
                keys=[vol] if isinstance(vol, str) else vol,
                offsets=0.10,
                prob=0.20,
            ),
            RandAffined(
                keys=vol_lab_key,
                prob=1.0,
                mode=["bilinear"] * (len(vol_lab_key) - 1) + ["nearest"],
                padding_mode="zeros",
                translate_range=[50.0, 50.0, 50.0],
            ),  # this is needed for more variation exposure to nmsw-net
            RandAffined(
                keys=vol_lab_key,
                prob=0.3,
                mode=["bilinear"] * (len(vol_lab_key) - 1) + ["nearest"],
                padding_mode="zeros",
                rotate_range=[0.52, 0.52, 0.52],
                scale_range=[0.2, 0.2, 0.3],  # more zooming in axial direction
                translate_range=[10.0, 10.0, 50.0],
            ),
            RandGaussianNoised(prob=0.1, keys=[vol] if isinstance(vol, str) else vol),
            RandGaussianSmoothd(
                prob=0.2,
                sigma_x=(0.5, 1.0),
                sigma_y=(0.5, 1.0),
                sigma_z=(0.5, 1.0),
                keys=[vol] if isinstance(vol, str) else vol,
            ),
            RandScaleIntensityd(
                prob=0.15,
                factors=(-0.25, 0.25),
                keys=[vol] if isinstance(vol, str) else vol,
            ),
            RandSimulateLowResolutiond(
                prob=0.25,
                zoom_range=(0.5, 1),
                keys=[vol] if isinstance(vol, str) else vol,
            ),
            RandAdjustContrastd(
                prob=0.1,
                gamma=(0.7, 1.5),
                invert_image=True,
                retain_stats=True,
                keys=[vol] if isinstance(vol, str) else vol,
            ),
            RandAdjustContrastd(
                prob=0.3,
                gamma=(0.7, 1.5),
                invert_image=False,
                retain_stats=True,
                keys=[vol] if isinstance(vol, str) else vol,
            ),
            (
                RandGridDistortiond(
                    prob=0.3,
                    num_cells=5,
                    distort_limit=(-0.05, 0.05),
                    keys=vol_lab_key,
                    mode=["bilinear"] * (len(vol_lab_key) - 1) + ["nearest"],
                )
                if do_deform
                else IdentityD(keys=vol_lab_key)
            ),
        ]

    elif rand_aug_type == "none":
        random_aug = [
            # random to save before padding
            RandShiftIntensityd(
                keys=[vol] if isinstance(vol, str) else vol,
                offsets=0.0,
                prob=0.01,
            ),
            IdentityD(
                keys=vol_lab_key,
            ),
        ]

    else:
        raise ValueError(
            f"`rand_aug_type`:{rand_aug_type} is not one of [heavy | light | none]"
        )

    post_aug = [
        {
            1: MergeLabelD(
                keys=lab,
                old_label_rule=old_label_rule,
                new_label_rule=new_label_rule,
            ),
            0: IdentityD(keys=lab),
        }[new_label_rule != None],
        ############################################
        # MAKE SURE THE VOL CAN BE DOWN_SAMPLED ATLEAST 4TIMES, AS REQUIRED BY MOST SEGMENTATION BACKBONE.
        ############################################
        DivisiblePadD(
            keys=vol_lab_key,
            k=global_divisible_k,
            mode=["constant"] * (len(vol_lab_key) - 1) + ["constant"],
        ),
        ############################################
        # UNIFY the volume size.
        ############################################
        ResizeWithPadOrCropd(
            keys=[vol] if isinstance(vol, str) else vol,
            spatial_size=global_roi_size,
            mode=("constant") if isinstance(vol, str) else ("constant",) * len(vol),
            value=-0.1,
        ),  #
        ResizeWithPadOrCropd(
            keys=[lab],
            spatial_size=global_roi_size,
            mode="constant" if isinstance(lab, str) else ("constant",),
            value=0,
        ),
        CopyItemsD(
            keys=vol_lab_key,
            names=patch_vol_lab_key,
        ),
        RandCropByPosNegLabelD(
            keys=patch_vol_lab_key,
            label_key=patch_lab_key,
            spatial_size=local_roi_size,
            pos=1,
            neg=1,
            num_samples=1,
        ),
        (
            ConcatItemsD(
                keys=vol,
                name=VOL,
                dim=0,
            ) 
            if isinstance(vol, list) 
            else IdentityD(keys=vol)
        ),
        (
            ConcatItemsD(
                keys=patch_vol_key,
                name=PATCH + "_" + VOL,
                dim=0,
            ) 
            if isinstance(vol, list) 
            else IdentityD(keys=patch_vol_key)
        ),
        (
            CopyItemsD(
                keys=[lab, patch_lab_key],
                names=[LAB, PATCH + "_" + LAB],
            )
            if lab != LAB
            else IdentityD(keys=LAB)
        ),
        SelectItemsd(keys=[VOL, LAB, PATCH + "_" + VOL, PATCH + "_" + LAB, CASE_ID]),
    ]
    return Compose(pre_aug + random_aug + post_aug)

def aug_brats(
    spacing: list[float],
    local_roi_size: list[int],
    global_roi_size: list[int],
    intensity_scale: list[float],
    old_label_rule: list[str],
    new_label_rule: dict[str, list[str]] | None = None,
    global_divisible_k: list[int] = [32, 32, 48],  # 16*2 for h,w and # 16*3 for global
    orientation_mode: str = "RAS",
    rand_aug_type=["none", "heavy"][-1],
    do_deform: bool = True,  # grid deformation if the dataset has little training sample,
    vol: list[str] | str = VOL,
    lab: str = LAB,
    source: str = VOL,
) -> Callable[[dict[str, Path]], dict[str, torch.Tensor]]:
    
    patch_vol_key = PATCH + "_" + vol if isinstance(vol, str) else [PATCH + "_" + v for v in vol]
    patch_lab_key = PATCH + "_" + lab
    vol_lab_key = flatten_list([vol, lab])
    patch_vol_lab_key = flatten_list([patch_vol_key, patch_lab_key])

    pre_aug = [
        LoadImaged(keys=vol_lab_key),
        EnsureChannelFirstd(keys=vol_lab_key),
        Orientationd(keys=vol_lab_key, axcodes=orientation_mode),
        ScaleIntensityRangePercentilesd(
            keys=vol,
            lower=intensity_scale[0],
            upper=intensity_scale[1],
            b_min=intensity_scale[2],
            b_max=intensity_scale[3],
            clip=True,
        ),
        ToTensord(keys=vol_lab_key),
    ]

    post_aug = [
        {
            1: MergeLabelD(
                keys=lab,
                old_label_rule=old_label_rule,
                new_label_rule=new_label_rule,
            ),
            0: IdentityD(keys=lab),
        }[new_label_rule != None],
        ############################################
        # MAKE SURE THE VOL CAN BE DOWN_SAMPLED ATLEAST 4TIMES, AS REQUIRED BY MOST SEGMENTATION BACKBONE.
        ############################################
        DivisiblePadD(
            keys=vol_lab_key,
            k=global_divisible_k,
            mode=["constant"] * (len(vol_lab_key) - 1) + ["constant"],
        ),
        ############################################
        # UNIFY the volume size.
        ############################################
        ResizeWithPadOrCropd(
            keys=[vol] if isinstance(vol, str) else vol,
            spatial_size=global_roi_size,
            mode=("constant") if isinstance(vol, str) else ("constant",) * len(vol),
            value=-0.1,
        ),  #
        ResizeWithPadOrCropd(
            keys=[lab],
            spatial_size=global_roi_size,
            mode="constant" if isinstance(lab, str) else ("constant",),
            value=0,
        ),
        Resized(
            keys=vol_lab_key,
            spatial_size=global_roi_size,
            mode=["bilinear"] * (len(vol_lab_key) - 1) + ["nearest"],
        ),
        CopyItemsD(
            keys=vol_lab_key,
            names=patch_vol_lab_key,
        ),
        RandCropByPosNegLabelD(
            keys=patch_vol_lab_key,
            label_key=patch_lab_key,
            spatial_size=local_roi_size,
            pos=1,
            neg=1,
            num_samples=1,
        ),
        (
            ConcatItemsD(
                keys=vol,
                name=VOL,
                dim=0,
            ) 
            if isinstance(vol, list) 
            else IdentityD(keys=vol)
        ),
        (
            ConcatItemsD(
                keys=patch_vol_key,
                name=PATCH + "_" + VOL,
                dim=0,
            ) 
            if isinstance(vol, list) 
            else IdentityD(keys=patch_vol_key)
        ),
        (
            CopyItemsD(
                keys=[lab, patch_lab_key],
                names=[LAB, PATCH + "_" + LAB],
            )
            if lab != LAB
            else IdentityD(keys=LAB)
        ),
        SelectItemsd(keys=[VOL, LAB, PATCH + "_" + VOL, PATCH + "_" + LAB, CASE_ID]),
    ]
    return Compose(pre_aug + post_aug)


if __name__ == "__main__":

    from utils.visualzation import VisVolLab
    from matplotlib import pyplot as plt

    TEST_DATASET = True

    if TEST_DATASET:

        data = {
            "vol": "/hpc/home/jeon74/no-more-sw/data/datasets/processed/Word/imagesTr/word_0002.nii.gz",
            "lab": "/hpc/home/jeon74/no-more-sw/data/datasets/processed/Word/labelsTr/word_0002.nii.gz",
            "case_id": "shit",
        }

        mode = TRAIN
        rand_aug_type = "heavy"

        new_label_rule = None
        old_label_rule = [
            "background",
            "liver",
            "spleen",
            "left_kidney",
            "right_kidney",
            "stomach",
            "gallbladder",
            "esophagus",
            "pancreas",
            "duodenum",
            "colon",
            "intestine",
            "adrenal",
            "rectum",
            "bladder",
            "Head_of_femur_L",
            "Head_of_femur_R",
        ]

        augmentation = aug(
            spacing=[1, 1, 1],
            intensity_scale=(-250, 500, 0.0, 1.0),
            old_label_rule=old_label_rule,
            new_label_rule=new_label_rule,
            local_roi_size=(128, 128, 128),
            global_roi_size=(256, 256, 512),
            rand_aug_type=rand_aug_type,
        )

        data = augmentation(data)

        visualizer = VisVolLab(num_classes=20)

        print(data.keys())
        print(data[VOL].shape)
        print(data[LAB].shape)

        img = visualizer.vis(
            data[VOL].detach().cpu(),
            data[LAB].detach().cpu(),
        )
        plt.figure(figsize=(30, 10))
        plt.title(data["case_id"])
        plt.imshow(img)
        plt.show()
