import numpy as np
import torch
from torch import tensor
import pyro
from pyro import distributions as dist
from pyro.infer import Predictive, MCMC, NUTS
from kornia.geometry.transform import Affine

from dataloader import load_and_center_data, make_ellipse_mask


torch.set_default_device("cuda:0")
torch.set_default_tensor_type(torch.FloatTensor)  # float32 set explicitly

rad = 10. * np.pi / 180.
PARAMETER, LOWER, UPPER = zip(*[
    ["dx",      -5.,    5.],  # x translation in px
    ["dy",      -5.,    5.],  # y translation in px
    ["dtheta", -rad,   rad],  # rotation in radians
    ["tau",      0.,    1.],  # opacity
])

observations, disk_models, galaxy_positions = load_and_center_data()
observations, disk_models = observations.unsqueeze(1), disk_models.unsqueeze(1)
disk_subtracted_models = observations - disk_models
ellipse_mask = make_ellipse_mask(9, 5, 35)

N_IMAGES = len(observations)


def model(x, y=None, yerr=tensor([3.0e-3])):
    # define inference parameters
    dx = pyro.sample("dx",
                     dist.Uniform(
                         -5.0 * torch.ones(N_IMAGES - 1),
                          5.0 * torch.ones(N_IMAGES - 1)
                     ))
    dy = pyro.sample("dy",
                     dist.Uniform(
                         -5.0 * torch.ones(N_IMAGES - 1),
                          5.0 * torch.ones(N_IMAGES - 1)
                     ))
    # dt = pyro.sample("dt",
    #                  dist.Uniform(
    #                      -10.0 * torch.pi / 180. * torch.ones(N_IMAGES - 1),
    #                       10.0 * torch.pi / 180. * torch.ones(N_IMAGES - 1)
    #                  ))
    log_tau = pyro.sample("log_tau", dist.Uniform(-20, 2.))

    # prepare parameters for transforms
    # translation = pyro.deterministic("translation", torch.stack((dx, dy), 1))
    # angle = pyro.deterministic("angle", torch.zeros(N_IMAGES - 1))
    translation = torch.stack((dx, dy), 1)
    angle = torch.zeros(N_IMAGES - 1)
    tau = pyro.deterministic("tau", torch.pow(10, log_tau))

    # correct images using disk attenuation
    # correction = pyro.deterministic("correction", torch.exp(- tau * disk_models))
    # x_corr = pyro.deterministic("x_corr", x / correction[1:])
    # y_corr = pyro.deterministic("y_corr", (y / correction[0]).unsqueeze(1).expand(len(x), -1, -1, -1))
    correction = torch.exp(- tau * disk_models)
    x_corr = x / correction[1:]
    y_corr = (y / correction[0]).unsqueeze(1).expand(len(x), -1, -1, -1)

    # transform the images
    transform = Affine(
        angle=angle,
        translation=translation,
        # mode="bilinear",
        # padding_mode="zero",
    )
    # translated_x_corr = pyro.deterministic(
    #     "translated_x_corr",
    #     transform(x_corr)
    # )
    translated_x_corr = transform(x_corr)

    with pyro.plate("data"):  # , len(x)) as i:

        #rss = torch.square(translated_galaxies - y_corr)

        return pyro.sample("obs", dist.Normal(translated_x_corr, yerr), obs=y_corr)

def guide(x, y=None, yerr=tensor([3.0e-3])):




    # define inference parameters
    dx = pyro.sample("dx",
                     dist.Uniform(
                         -5.0 * torch.ones(N_IMAGES - 1),
                          5.0 * torch.ones(N_IMAGES - 1)
                     ))
    dy = pyro.sample("dy",
                     dist.Uniform(
                         -5.0 * torch.ones(N_IMAGES - 1),
                          5.0 * torch.ones(N_IMAGES - 1)
                     ))
    # dt = pyro.sample("dt",
    #                  dist.Uniform(
    #                      -10.0 * torch.pi / 180. * torch.ones(N_IMAGES - 1),
    #                       10.0 * torch.pi / 180. * torch.ones(N_IMAGES - 1)
    #                  ))
    log_tau = pyro.sample("log_tau", dist.Uniform(-20, 2.))

    # prepare parameters for transforms
    translation = torch.stack((dx, dy), 1)
    angle = torch.zeros(N_IMAGES - 1)
    tau = pyro.deterministic("tau", torch.pow(10, log_tau))

    # correct images using disk attenuation
    correction = torch.exp(- tau * disk_models)
    x_corr = x / correction[1:]
    y_corr = (y / correction[0]).unsqueeze(1).expand(len(x), -1, -1, -1)

    # transform the images
    transform = Affine(
        angle=angle,
        translation=translation,
        # mode="bilinear",
        # padding_mode="zero",
    )

    translated_x_corr = transform(x_corr)

    with pyro.plate("data"):  # , len(x)) as i:

        #rss = torch.square(translated_galaxies - y_corr)

        return pyro.sample("obs", dist.Normal(translated_x_corr, yerr), obs=y_corr)


if __name__ == "__main__":
    # print(observations.shape)
    # print(disk_subtracted_models.shape)
    # print(disk_models.shape)
    # print(ellipse_mask.shape)

    pyro.render_model(model, model_args=(disk_subtracted_models[1:], disk_subtracted_models[0]),
                      render_distributions=True, filename="model.pdf")

    kernel = NUTS(model,
                  target_accept_prob=0.9,
                  jit_compile=True,
                  # ignore_jit_warnings=True,
                  )
    mcmc = MCMC(kernel,
                warmup_steps=500,
                num_samples=2000,
                mp_context="spawn",
                )

    mcmc.run(x=disk_subtracted_models[1:],
             y=disk_subtracted_models[0],
             )

    mcmc.summary()

    mc_samples = mcmc.get_samples()

    import matplotlib.pyplot as plt
    from corner import corner

    corner(mcmc)

    plt.show()

    # predictive = Predictive(model, mc_samples)
    #
    # predictions = predictive(x=disk_subtracted_models[1:], y=disk_subtracted_models[0])['obs']
    #
    # print(predictions.shape)

    # sampler = infer.MCMC(
    #     infer.NUTS(model),
    #     num_warmup=2000,
    #     num_samples=2000,
    #     num_chains=2,
    #     progress_bar=True,
    # )