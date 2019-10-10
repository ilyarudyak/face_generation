from pathlib import Path

import torch
from torchvision import datasets
from torchvision.transforms import transforms


def get_dataloader(batch_size, image_size,
                   data_dir=Path.home()/'data/processed_celeba_small/'):
    """
    Batch the neural network data using DataLoader
    :param batch_size: The size of each batch; the number of images in a batch
    :param img_size: The square size of the image data (x, y)
    :param data_dir: Directory where image data is located
    :return: DataLoader with batched data
    """

    data_transform = transforms.Compose([transforms.Resize(image_size),
                                         transforms.ToTensor()])
    train_data = datasets.ImageFolder(data_dir,transform=data_transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                               num_workers=0, shuffle=True)

    return train_loader