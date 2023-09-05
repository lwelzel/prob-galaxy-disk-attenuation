import numpy as np
import pandas as pd
import torch
from torch import tensor
from torch.utils.data import DataLoader, Dataset
import kornia as ko


from pathlib import Path
from astropy.io import fits
from matplotlib import pyplot as plt

CUT_IMG_X = 512  # default: 1023
CUT_IMG_Y = 426  # default: 426

def load_observations():
    observations_path = Path("/home/lwelzel/Documents/git/debris-disk-photometry/data/raw_data/hd107146_reduced/")
    observations = sorted(list(observations_path.glob("*.fits")))
    observations = [tensor(fits.open(path)[0].data[:CUT_IMG_Y, :CUT_IMG_X].astype(np.float32)) for path in observations]

    return observations

def load_disk_model():
    disk_model_path = Path("/home/lwelzel/Documents/git/debris-disk-photometry/data/raw_data/hd107146_disk_model/MODEL20_HD107146_16.5_47_h=0.3_NUP.fits")
    disk_model = tensor(fits.open(disk_model_path)[0].data.astype(np.float32))
    disk_model = ko.geometry.translate(disk_model[None, ...], translation=tensor([-540.45, -541.08])[None, ...])
    disk_model = torch.squeeze(disk_model)[:CUT_IMG_Y, :CUT_IMG_X]
    return disk_model


def load_position_guess():
    csv_path = Path("/home/lwelzel/Documents/git/debris-disk-photometry/data/est_pos_galaxy.csv")
    df = pd.read_csv(csv_path)

    return df

def load_all_data():
    x_range = 128
    y_range = 128
    x_0 = 300
    x_space = [x_0, x_0 + x_range]
    y_0 = 60
    y_space = [y_0, y_0 + y_range]

    observations = load_observations()
    observations = [obs[y_space[0]:y_space[1], x_space[0]:x_space[1]] for obs in observations]

    disk_model = load_disk_model()
    disk_model = disk_model[y_space[0]:y_space[1], x_space[0]:x_space[1]]

    galaxy_positions = load_position_guess()
    galaxy_positions["x_est"] = galaxy_positions["x_est"] - x_0 - 2
    galaxy_positions["y_est"] = galaxy_positions["y_est"] - y_0 - 2

    return observations, disk_model, galaxy_positions


def load_and_center_data():
    trim = 32 + 16 + 4

    observations, disk_model, galaxy_positions = load_all_data()

    current_center = observations[0].shape[0] / 2

    translations_x = [
        (current_center - x) for x in galaxy_positions["x_est"]
    ]
    translations_y = [
        (current_center - y) for y in galaxy_positions["y_est"]
    ]
    translations = tensor([[x, y] for x, y in zip(translations_x, translations_y)])

    observations = torch.unsqueeze(torch.stack(observations), 1)
    observations = ko.geometry.translate(observations, translation=translations)
    observations = torch.squeeze(observations)[:, trim:-trim, trim:-trim]

    disk_models = torch.unsqueeze(torch.expand_copy(disk_model, (8, -1, -1)), 1)
    disk_models = ko.geometry.translate(disk_models, translation=translations)
    disk_models = torch.squeeze(disk_models)[:, trim:-trim, trim:-trim]

    galaxy_positions["x_est"] = galaxy_positions["x_est"] + translations_x - trim
    galaxy_positions["y_est"] = galaxy_positions["y_est"] + translations_y - trim


    return observations, disk_models, galaxy_positions

def make_ellipse_mask(a, b, theta, h=24, w=24, n_images=1):
    theta = theta * np.pi / 180
    a, b, theta = tensor([a]), tensor([b]), tensor([theta])

    __, y, x = torch.meshgrid(
        [
            torch.ones(n_images),
            torch.linspace(-h//2, h//2, h),
            torch.linspace(-w//2, w//2, w)
         ],
                          indexing='ij')
    x = x.to(torch.float32)
    y = y.to(torch.float32)
    ellipse = ((x * torch.cos(theta) + y * torch.sin(theta)) ** 2) / a ** 2 \
             + ((x * torch.sin(theta) - y * torch.cos(theta)) ** 2) / b ** 2
    ellipse = (ellipse <= 1).to(torch.float32)  # this will return a binary mask
    return ellipse




if __name__ == "__main__":
    observations = load_observations()
    disk_model = load_disk_model()
    galaxy_positions = load_position_guess()

    fig, axes = plt.subplots(3, 3,
                             constrained_layout=True,
                             sharex=True, sharey=True
                             )

    axes = np.array(axes).flatten()

    for i, (ax, img) in enumerate(zip(axes, observations + [disk_model])):
        ax.imshow(img,
                  origin="lower",
                  vmin=-0.05, vmax=0.3)
        if i < 8:
            ax.scatter(galaxy_positions.iloc[i]["x_est"],
                       galaxy_positions.iloc[i]["y_est"],
                       marker="+", c="white", s=50, linewidths=0.75)
        else:
            ax.scatter(galaxy_positions["x_est"],
                       galaxy_positions["y_est"],
                       marker="+", c="white", s=20, linewidths=0.75)

    plt.show()

    observations, disk_model, galaxy_positions = load_all_data()

    fig, axes = plt.subplots(3, 3,
                             constrained_layout=True,
                             sharex=True, sharey=True
                             )

    axes = np.array(axes).flatten()

    for i, (ax, img) in enumerate(zip(axes, observations + [disk_model])):
        ax.imshow(img,
                  origin="lower",
                  vmin=-0.05, vmax=0.3)
        if i < 8:
            ax.scatter(galaxy_positions.iloc[i]["x_est"],
                       galaxy_positions.iloc[i]["y_est"],
                       marker="+", c="white", s=50, linewidths=0.75)
        else:
            ax.scatter(galaxy_positions["x_est"],
                       galaxy_positions["y_est"],
                       marker="+", c="white", s=20, linewidths=0.75)

    plt.show()

    observations, disk_models, galaxy_positions = load_and_center_data()
    ellipse_mask = make_ellipse_mask(9, 5, 35)
    print("Number of active pixels in the ellipse mask: " + str(torch.sum(ellipse_mask).item()))
    only_bg = (observations - disk_models) * ellipse_mask

    fig, axes = plt.subplots(3, 3,
                             constrained_layout=True,
                             sharex=True, sharey=True
                             )

    axes = np.array(axes).flatten()

    for i, (ax, img) in enumerate(zip(axes, only_bg)):
        ax.imshow(img,
                  origin="lower",
                  vmin=-0.05, vmax=0.3)
        if i < 8:
            ax.scatter(galaxy_positions.iloc[i]["x_est"],
                       galaxy_positions.iloc[i]["y_est"],
                       marker="+", c="white", s=50, linewidths=0.75)

    axes[-1].axis("off")

    plt.show()