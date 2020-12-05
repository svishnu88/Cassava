import albumentations as A
from albumentations.augmentations.transforms import GridDistortion, OpticalDistortion
from albumentations.pytorch.transforms import ToTensor


def get_augmentations(p=0.5, image_size=224):
    imagenet_stats = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
    train_tfms = A.Compose(
        [
            A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.4, rotate_limit=45, p=p),
            A.Cutout(p=p),
            A.RandomRotate90(p=p),
            A.Flip(p=p),
            A.OneOf(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=0.2, contrast_limit=0.2,
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=20, sat_shift_limit=50, val_shift_limit=50
                    ),
                ],
                p=p,
            ),
            A.OneOf([A.IAAAdditiveGaussianNoise(), A.GaussNoise(),], p=p),
            A.CoarseDropout(max_holes=10, p=p),
            A.OneOf(
                [
                    A.MotionBlur(p=0.2),
                    A.MedianBlur(blur_limit=3, p=0.1),
                    A.Blur(blur_limit=3, p=0.1),
                ],
                p=p,
            ),
            #         A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.4, rotate_limit=45, p=p),
            A.OneOf(
                [
                    A.OpticalDistortion(p=0.3),
                    A.GridDistortion(p=0.1),
                    A.IAAPiecewiseAffine(p=0.3),
                ],
                p=p,
            ),
            A.Resize(image_size, image_size),
            ToTensor(normalize=imagenet_stats),
        ]
    )

    test_tfms = A.Compose([ToTensor(normalize=imagenet_stats)])
    return train_tfms, test_tfms
