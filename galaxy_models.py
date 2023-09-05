import numpy as np
from typing import Union
import torch
from torch import tensor, masked_select
from torch.nn import Module
from torch.masked import masked_tensor, as_masked_tensor
import pyro
from pyro import distributions as dist
from pyro.infer import Predictive, MCMC, NUTS
from kornia.geometry.transform import Affine

from dataloader import load_and_center_data, make_ellipse_mask

torch.set_default_device("cuda:0")
torch.set_default_tensor_type(torch.FloatTensor)  # float32 set explicitly

rad = 10. * np.pi / 180.
PARAMETER, LOWER, UPPER = zip(*[
    ["dx", -5., 5.],  # x translation in px
    ["dy", -5., 5.],  # y translation in px
    ["dtheta", -rad, rad],  # rotation in radians
    ["tau", 0., 1.],  # opacity
])

observations, disk_models, galaxy_positions = load_and_center_data()
observations, disk_models = observations.unsqueeze(1), disk_models.unsqueeze(1)
disk_subtracted_models = observations - disk_models
ellipse_mask = make_ellipse_mask(9, 5, 35)

N_IMAGES = len(observations)

class GalaxyModel(Module):
    def __init__(self, h=24, w=24, n_images=1):
        super().__init__()

        __, self.y_bulge, self.x_bulge = torch.meshgrid(
            [
                torch.ones(n_images),
                torch.linspace(-h // 2, h // 2, h),
                torch.linspace(-w // 2, w // 2, w)],
            indexing='ij')

        __, self.y_disk, self.x_disk = torch.meshgrid(
            [
                torch.ones(n_images),
                torch.linspace(-h // 2, h // 2, h),
                torch.linspace(-w // 2, w // 2, w)
            ],
            indexing='ij')


    def forward(self,dx, dy,
                amplitude_bulge, r_eff, n, ellip_bulge, theta_bulge,
                amplitude_disk, scale_height, ellip_disk, theta_disk):

        z_bulge = self.get_transformed_field(dx, dy,  theta_bulge, r_eff, ellip_bulge)
        z_disk = self.get_transformed_field(dx, dy,  theta_disk, scale_height, ellip_disk)

        bn = self.approx_bn(n)

        bulge = amplitude_bulge * torch.exp(-bn * (z_bulge ** (1 / n) - 1))
        disk = amplitude_disk * torch.exp(- z_disk / scale_height)

        return bulge + disk

    def get_transformed_field(self, dx, dy,  theta, scale, ellip):
        rad_theta = torch.deg2rad(theta)

        cos_theta = torch.cos(rad_theta)
        sin_theta = torch.sin(rad_theta)

        __ = (self.x_bulge - dx)
        __ = cos_theta
        __ = (self.y_bulge - dy)
        __ = sin_theta

        x_maj = (self.x_bulge - dx) * cos_theta + (self.y_bulge - dy) * sin_theta
        x_min = -(self.x_bulge - dx) * sin_theta + (self.y_bulge - dy) * cos_theta

        z = torch.sqrt((x_maj / scale) ** 2 + (x_min / ((1 - ellip) * scale)) ** 2)

        return z

    @staticmethod
    def approx_bn(n):
        """
        Calculate bn using polynomial approximation: https://en.wikipedia.org/wiki/S%C3%A9rsic_profile.
        :param n: 8 > n > 0.36 (there is an approximation for n > 8: https://en.wikipedia.org/wiki/S%C3%A9rsic_profile)
        :return: bn
        """

        p0 = + 2. * n
        p1 = - 1. / 3.
        p2 = + 4. / (405. * n)
        p3 = + 46. / (25515. * torch.pow(n, 2.))
        p4 = + 131. / (1148175. * torch.pow(n, 3.))
        p5 = - 2194697. / (30690717750. * torch.pow(n, 4.))

        return p0 + p1 + p2 + p3 + p4 + p5


class VecGalaxyModel(Module):
    def __init__(self, h=24, w=24):
        super().__init__()

        self._galaxy_model = GalaxyModel(h=h, w=w)

        self.forward = torch.vmap(self._galaxy_model)

nn_galaxy_model = GalaxyModel()


# class GalaxyImageLoss(Module):
#     def __init__(self, mask, truths=None, yerr=3.0e-3):
#         super().__init__()
#
#         self.mask = mask.bool()
#         self.yerr = yerr
#
#         if truths is None:
#             self.forward = self.var_fwd
#         else:
#             self.y = masked_tensor(truths, self.mask)
#             self.forward = self.static_fwd
#
#     def var_fwd(self, x, y):
#         chi_sqrd = torch.square(masked_tensor(y, self.mask) - masked_tensor(x, self.mask)) / self.yerr ** 2
#         loss = torch.sum(chi_sqrd)
#         return loss
#
#     def static_fwd(self, x):
#         chi_sqrd = torch.square(self.y - masked_tensor(x, self.mask)) / self.yerr ** 2
#         loss = torch.sum(chi_sqrd)
#         return loss

class GalaxyImageLoss(Module):
    def __init__(self, mask, truths=None, yerr=3.0e-3):
        super().__init__()

        self.mask = mask.bool()
        self.yerr = yerr

        if truths is None:
            self.forward = self.var_fwd
        else:
            self.y = masked_select(truths, self.mask)
            self.forward = self.static_fwd

    def var_fwd(self, x, y):
        chi_sqrd = torch.square(masked_select(y, self.mask) - masked_select(x, self.mask)) / self.yerr ** 2
        loss = torch.sum(chi_sqrd)
        return loss

    def static_fwd(self, x):
        chi_sqrd = torch.square(self.y - masked_select(x, self.mask)) / self.yerr ** 2
        loss = torch.sum(chi_sqrd)
        return loss

def approx_bn(n):
    bn = 2. * n \
         - 1. / 3. \
         + 4. / (405. * n) \
         + 46. / (25515. * torch.pow(n, 2.)) \
         + 131. / (1148175. * torch.pow(n, 3.)) \
         - 2194697. / (30690717750. * torch.pow(n,  4.))

    return bn

def galaxy_bulge_model(x, y,
                       amplitude_bulge=tensor([0.094]), r_eff=tensor([4.53]), n=tensor([0.8]),
                       ellip_bulge=tensor([0.284]), theta_bulge=tensor([15.08]),
                       dx=0., dy=0.):

    rad_theta_bulge = torch.deg2rad(theta_bulge)

    # print(n)
    # print(n)
    # print(n * 2.)

    # calculate bn using polynomial approximation: https://en.wikipedia.org/wiki/S%C3%A9rsic_profile
    # for 8 > n > 0.36
    # bn = 2.0 * n - 1. / 3. + 4 / (405 * n) + 46 / (25515 * torch.pow(n, 2)) + 131 / (1148175 * torch.pow(n, 3)) \
    #      - 2194697 / (30690717750 * torch.pow(n,  4))

    bn = approx_bn(n)

    # there is an approximation for n > 8: https://en.wikipedia.org/wiki/S%C3%A9rsic_profile

    a, b = r_eff, (1 - ellip_bulge) * r_eff
    cos_theta, sin_theta = torch.cos(rad_theta_bulge), torch.sin(rad_theta_bulge)
    x_maj = (x - dx) * cos_theta + (y - dy) * sin_theta
    x_min = -(x - dx) * sin_theta + (y - dy) * cos_theta
    z = torch.sqrt((x_maj / a) ** 2 + (x_min / b) ** 2)

    bulge = amplitude_bulge * torch.exp(-bn * (z ** (1 / n) - 1))

    return bulge


def galaxy_disk_model(x, y,
                      amplitude_disk=tensor([0.129]), scale_height=tensor([20.77]),
                      ellip_disk=tensor([0.625]), theta_disk=tensor([26.74]),
                      dx=0., dy=0.):
    rad_theta_disk = torch.deg2rad(theta_disk)

    a, b = scale_height, (1 - ellip_disk) * scale_height
    cos_theta, sin_theta = torch.cos(rad_theta_disk), torch.sin(rad_theta_disk)
    x_maj = (x - dx) * cos_theta + (y - dy) * sin_theta
    x_min = -(x - dx) * sin_theta + (y - dy) * cos_theta
    z = torch.sqrt((x_maj / a) ** 2 + (x_min / b) ** 2)

    disk = amplitude_disk * torch.exp(- z / scale_height)

    return disk


# def _galaxy_model(dx: Union[float, torch.Tensor] = tensor([0.]),
#                   dy: Union[float, torch.Tensor] = tensor([0.]),
#                   amplitude_bulge: Union[float, torch.Tensor] = tensor([0.094]),
#                   r_eff: Union[float, torch.Tensor] = tensor([4.53]),
#                   n: Union[float, torch.Tensor] = tensor([0.8]),
#                   ellip_bulge: Union[float, torch.Tensor] = tensor([0.284]),
#                   theta_bulge: Union[float, torch.Tensor] = tensor([15.08]),
#                   amplitude_disk: Union[float, torch.Tensor] = tensor([0.129]),
#                   scale_height: Union[float, torch.Tensor] = tensor([20.77]),
#                   ellip_disk: Union[float, torch.Tensor] = tensor([0.625]),
#                   theta_disk: Union[float, torch.Tensor] = tensor([26.74]),
#                   h: Union[int, torch.Tensor] = 24,
#                   w: Union[int, torch.Tensor] = 24):
#
#     # make to tensors if floats
#     # amplitude_bulge, r_eff, n = tensor(amplitude_bulge), tensor(r_eff), tensor(n)
#     # ellip_bulge, theta_bulge = tensor(ellip_bulge), tensor(theta_bulge)
#     # amplitude_disk, scale_height = tensor(amplitude_disk), tensor(scale_height),
#     # ellip_disk, theta_disk = tensor(ellip_disk), tensor(theta_disk)
#
#     y_bulge, x_bulge = torch.meshgrid([torch.linspace(-h // 2, h // 2, h),
#                                        torch.linspace(-w // 2, w // 2, w)],
#                                       indexing='ij')
#
#     y_disk, x_disk = torch.meshgrid([torch.linspace(-h // 2, h // 2, h),
#                                      torch.linspace(-w // 2, w // 2, w)],
#                                     indexing='ij')
#
#     bulge = galaxy_bulge_model(x_bulge, y_bulge,
#                                amplitude_bulge=amplitude_bulge, r_eff=r_eff, n=n,
#                                ellip_bulge=ellip_bulge, theta_bulge=theta_bulge,
#                                dx=dx, dy=dy)
#
#     disk = galaxy_disk_model(x_disk, y_disk,
#                              amplitude_disk=amplitude_disk, scale_height=scale_height,
#                              ellip_disk=ellip_disk, theta_disk=theta_disk,
#                              dx=dx, dy=dy)
#
#     galaxy = bulge + disk
#
#     return galaxy


def _galaxy_model(dx: Union[float, torch.Tensor] = tensor([0.]),
                  dy: Union[float, torch.Tensor] = tensor([0.]),
                  amplitude_bulge: Union[float, torch.Tensor] = tensor([0.094]),
                  r_eff: Union[float, torch.Tensor] = tensor([4.53]),
                  n: Union[float, torch.Tensor] = tensor([0.8]),
                  ellip_bulge: Union[float, torch.Tensor] = tensor([0.284]),
                  theta_bulge: Union[float, torch.Tensor] = tensor([15.08]),
                  amplitude_disk: Union[float, torch.Tensor] = tensor([0.129]),
                  scale_height: Union[float, torch.Tensor] = tensor([20.77]),
                  ellip_disk: Union[float, torch.Tensor] = tensor([0.625]),
                  theta_disk: Union[float, torch.Tensor] = tensor([26.74]),
                  h: Union[int, torch.Tensor] = 24,
                  w: Union[int, torch.Tensor] = 24):

    y_bulge, x_bulge = torch.meshgrid([torch.linspace(-h // 2, h // 2, h),
                                       torch.linspace(-w // 2, w // 2, w)],
                                      indexing='ij')

    y_disk, x_disk = torch.meshgrid([torch.linspace(-h // 2, h // 2, h),
                                     torch.linspace(-w // 2, w // 2, w)],
                                    indexing='ij')

    rad_theta_bulge = torch.deg2rad(theta_bulge)
    rad_theta_disk = torch.deg2rad(theta_disk)

    # calculate bn using polynomial approximation: https://en.wikipedia.org/wiki/S%C3%A9rsic_profile
    # for 8 > n > 0.36
    # there is an approximation for n > 8: https://en.wikipedia.org/wiki/S%C3%A9rsic_profile

    bn = 2. * n \
         - 1. / 3. \
         + 4. / (405. * n) \
         + 46. / (25515. * torch.pow(n, 2.)) \
         + 131. / (1148175. * torch.pow(n, 3.)) \
         - 2194697. / (30690717750. * torch.pow(n, 4.))

    a, b = r_eff, (1 - ellip_bulge) * r_eff
    cos_theta, sin_theta = torch.cos(rad_theta_bulge), torch.sin(rad_theta_bulge)
    x_maj = (x_bulge - dx) * cos_theta + (y_bulge - dy) * sin_theta
    x_min = -(x_bulge - dx) * sin_theta + (y_bulge - dy) * cos_theta
    z = torch.sqrt((x_maj / a) ** 2 + (x_min / b) ** 2)

    bulge = amplitude_bulge * torch.exp(-bn * (z ** (1 / n) - 1))

    a, b = scale_height, (1 - ellip_disk) * scale_height
    cos_theta, sin_theta = torch.cos(rad_theta_disk), torch.sin(rad_theta_disk)
    x_maj = (x_disk - dx) * cos_theta + (y_disk - dy) * sin_theta
    x_min = -(x_disk - dx) * sin_theta + (y_disk - dy) * cos_theta
    z = torch.sqrt((x_maj / a) ** 2 + (x_min / b) ** 2)

    disk = amplitude_disk * torch.exp(- z / scale_height)

    galaxy = bulge + disk

    return galaxy


galaxy_model = torch.vmap(_galaxy_model,
                          in_dims=0, out_dims=0)


def make_synthetic_data(tau=1.e-3, sigma=3.0e-3):
    from astropy.io import fits

    dx = torch.zeros(N_IMAGES) + torch.normal(0. * torch.ones(N_IMAGES), 1.)
    dy = torch.zeros(N_IMAGES) + torch.normal(0. * torch.ones(N_IMAGES), 1.)
    dx[0] = 0.
    dy[0] = 0.

    print(dx)
    print(dy)
    print(tau)


    models = torch.zeros_like(disk_models)

    for i in range(8):
        model = _galaxy_model(dx=dx[i], dy=dy[i])
        models[i, :] = model

    models = models * torch.exp(- tau * disk_models)
    models = models + torch.normal(torch.zeros_like(models), sigma)

    hdu = fits.PrimaryHDU(models.squeeze().cpu().numpy())

    hdu.writeto(
        '/home/lwelzel/Documents/git/debris-disk-photometry/data/raw_data/hd107146_synthetic/synthetic_cube.fits',
        overwrite=True,
    )


if __name__ == "__main__":

    make_synthetic_data()


    raise NotImplementedError
    import matplotlib.pyplot as plt

    # main interface
    dxs = torch.linspace(-7., 7., 9)

    galaxy = galaxy_model(dxs)

    fig, axes = plt.subplots(3, 3, constrained_layout=True,
                             sharex=True, sharey=True)
    axes = np.array(axes).flatten()
    for i in range(len(dxs)):
        im = axes[i].contourf(galaxy[i].cpu().numpy(), origin="lower")
        plt.colorbar(im, ax=axes[i])

    plt.show()

    # sub routines

    h = 24
    w = 24
    y, x = torch.meshgrid([torch.linspace(-h // 2, h // 2, h),
                           torch.linspace(-w // 2, w // 2, w)],
                          indexing='ij')

    dx, dy = -1.4, -1.4

    bulge = galaxy_bulge_model(x, y, dx=dx, dy=dy)
    disk = galaxy_disk_model(x, y, dx=dx, dy=dy)

    galaxy = bulge + disk

    im = plt.contourf(bulge.cpu().numpy(), origin="lower")
    plt.colorbar(im)

    plt.show()

    im = plt.contourf(disk.cpu().numpy(), origin="lower")
    plt.colorbar(im)

    plt.show()

    im = plt.contourf(galaxy.cpu().numpy(), origin="lower")
    plt.colorbar(im)

    plt.show()


    # with pyro.plate("data", len(x)):
    # __, y_bulge, x_bulge = torch.meshgrid([torch.ones(8),
    #                                        torch.linspace(-h // 2, h // 2, h),
    #                                        torch.linspace(-w // 2, w // 2, w)],
    #                                       indexing='ij')
    #
    # __, y_disk, x_disk = torch.meshgrid([torch.ones(8),
    #                                      torch.linspace(-h // 2, h // 2, h),
    #                                      torch.linspace(-w // 2, w // 2, w)],
    #                                     indexing='ij')
    #
    # rad_theta_bulge = torch.deg2rad(theta_bulge)
    # rad_theta_disk = torch.deg2rad(theta_disk)
    #
    # bn = 2. * n \
    #      - 1. / 3. \
    #      + 4. / (405. * n) \
    #      + 46. / (25515. * torch.pow(n, 2.)) \
    #      + 131. / (1148175. * torch.pow(n, 3.)) \
    #      - 2194697. / (30690717750. * torch.pow(n, 4.))
    #
    # a, b = r_eff, (1 - ellip_bulge) * r_eff
    # cos_theta, sin_theta = torch.cos(rad_theta_bulge), torch.sin(rad_theta_bulge)
    # x_maj = (x_bulge - dx) * cos_theta + (y_bulge - dy) * sin_theta
    # x_min = -(x_bulge - dx) * sin_theta + (y_bulge - dy) * cos_theta
    # z = torch.sqrt((x_maj / a) ** 2 + (x_min / b) ** 2)
    #
    # bulge = amplitude_bulge * torch.exp(-bn * (z ** (1 / n) - 1))
    #
    # a, b = scale_height, (1 - ellip_disk) * scale_height
    # cos_theta, sin_theta = torch.cos(rad_theta_disk), torch.sin(rad_theta_disk)
    # x_maj = (x_disk - dx) * cos_theta + (y_disk - dy) * sin_theta
    # x_min = -(x_disk - dx) * sin_theta + (y_disk - dy) * cos_theta
    # z = torch.sqrt((x_maj / a) ** 2 + (x_min / b) ** 2)
    #
    # disk = amplitude_disk * torch.exp(- z / scale_height)
    #
    # galaxy = bulge + disk
