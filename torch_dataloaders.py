import numpy as np
import pandas as pd
import math
import torch
from torch import tensor
from torch.utils.data import Dataset, DataLoader, IterableDataset
import kornia as ko


from pathlib import Path
from astropy.io import fits
from matplotlib import pyplot as plt

class SyntheticDataset(Dataset):
    def __init__(self, data, batch_size):
        super().__init__()
        self.data = data
        self.batch_size = batch_size

    def __len__(self):
        return self.batch_size

    def __getitem__(self, idx):
        try:
            return self.data.expand(len(idx), -1, -1, -1)
        except TypeError:
            return self.data


def data_loader_synthetic(n_images=1, batch_size=128):
    from galaxy_models import galaxy_model

    dxs = 4. * (torch.rand(n_images) - 0.5)
    dys = 4. * (torch.rand(n_images) - 0.5)
    amplitude_bulge = tensor([0.094])
    r_eff = tensor([4.53])
    n = tensor([0.8])
    ellip_bulge = tensor([0.284])
    theta_bulge = tensor([15.08])
    amplitude_disk = tensor([0.129])
    scale_height = tensor([20.77])
    ellip_disk = tensor([0.625])
    theta_disk = tensor([26.74])

    truths = {
        "amplitude_bulge": amplitude_bulge.item(),
        "r_eff": r_eff.item(),
        "n": n.item(),
        "ellip_bulge": ellip_bulge.item(),
        "theta_bulge": theta_bulge.item(),
        "amplitude_disk": amplitude_disk.item(),
        "scale_height": scale_height.item(),
        "ellip_disk": ellip_disk.item(),
        "theta_disk": theta_disk.item(),
    }

    if len(dxs) > 1:
        for i, (ddx, ddy) in enumerate(zip(dxs, dys)):
            truths[f"dx{i}"] = ddx.cpu().numpy()
            truths[f"dy{i}"] = ddy.cpu().numpy()
    else:
        truths[f"dx"] = dxs.item()
        truths[f"dy"] = dys.item()

    yerr = 3.e-3
    tau = 1.e-4

    # IMAGE GENERATION
    true_galaxy_images = galaxy_model(dxs, dys,
                                      amplitude_bulge, r_eff, n, ellip_bulge, theta_bulge,
                                      amplitude_disk, scale_height, ellip_disk, theta_disk)
    noisy_galaxy_images = torch.normal(true_galaxy_images, yerr)

    dataset = SyntheticDataset(noisy_galaxy_images, batch_size)

    train_dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size
    )

    test_dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size
    )

    return train_dataloader, test_dataloader

if __name__ == "__main__":
    train_dataloader, test_dataloader = data_loader_synthetic(n_images=1, batch_size=128)

    for batch in train_dataloader:
        print(batch.shape)