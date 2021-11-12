import os
import numpy as np
import torch
import torchvision
import albumentations as A
from albumentations.pytorch import ToTensorV2

from utils.misc import dataset_dir

VOC_COLORMAP = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 12, 13, 14, 15, 16, 17, 18, 19, 20
]


class PascalVOC(torch.utils.data.Dataset):
    def __init__(self, root, image_set, download,
                 transform):
        super().__init__()
        self.pre_loaded_data = torchvision.datasets.VOCSegmentation(
            root=root,
            year='2012',
            image_set=image_set,
            download=download,
            transform=None
        )
        self.transform = transform

    def __len__(self):
        return len(self.pre_loaded_data)

    def __getitem__(self, index):
        image, mask = self.pre_loaded_data.__getitem__(index)
        image = np.array(image)
        mask = np.array(mask)
        mask = self._one_hot_encode(mask)

        transformed = self.transform(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask']

        return image, mask

    @staticmethod
    def _one_hot_encode(mask: torch.Tensor):
        """
        Output shape [height, width, num_classes].
        Each channel in the mask should encode values for a single class.
        Pixel in a mask channel should have a value of 1.0 if the pixel
        of the image belongs to this class and 0.0 otherwise.
        """
        height, width = mask.shape
        encoding = np.zeros((height, width, len(VOC_COLORMAP)),
                            dtype=np.float32)
        for label_index, label in enumerate(VOC_COLORMAP):
            encoding[:, :, label_index] = (mask == label)
        return encoding


def pascal_loader(batch_size):
    data_directory = os.path.join(dataset_dir(), 'pascal')

    train_transform = A.Compose([
        A.PadIfNeeded(min_height=256, min_width=256),
        A.CenterCrop(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(transpose_mask=True),
    ])
    test_transform = A.Compose([
        A.PadIfNeeded(min_height=256, min_width=256),
        A.CenterCrop(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(transpose_mask=True),
    ])
    trainset = PascalVOC(
        root=data_directory,
        image_set='train',
        download=False,
        transform=train_transform
    )
    testset = PascalVOC(
        root=data_directory,
        image_set='val',
        download=False,
        transform=test_transform,
    )
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    return trainloader, testloader


def remove_one_hot_enc(mask: torch.Tensor):
    """
    Function to remove one-hot-encoding from the mask.
    Expected tensor of shape (N, C, H, W).
    """
    category = torch.any(mask, dim=1).int().float()
    category = torch.abs((category - 1) * 255)
    _, wrong_mask = torch.max(mask, dim=1)
    true_mask = category + wrong_mask

    return true_mask
