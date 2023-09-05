import numpy as np
import torch
from torch import tensor
from torch.distributions import constraints
import pyro
from pyro import distributions as dist
from pyro import poutine
from pyro.infer import Predictive, MCMC, NUTS
import matplotlib.pyplot as plt
import random
import arviz as az
from corner import corner

from dataloader import load_and_center_data, make_ellipse_mask
from galaxy_models import galaxy_model, GalaxyModel, GalaxyImageLoss

torch.set_default_device("cuda:0")
torch.set_default_tensor_type(torch.FloatTensor)  # float32 set explicitly
random.seed(42)
torch.manual_seed(42)

N_IMAGES = 1

# MODEL PARAMETERS
dxs = 4. * (torch.rand(1) - 0.5)
dys = 4. * (torch.rand(1) - 0.5)
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
ellipse_mask = make_ellipse_mask(9, 5, 35)

# MODEL AND LOG_PROB (LOSS)
nn_galaxy_model = GalaxyModel()
nn_galaxy_loss = GalaxyImageLoss(mask=ellipse_mask, truths=true_galaxy_images)

class GalaxyLogProb(dist.TorchDistribution):
    support = constraints.real
    has_rsample = False  # sampling this dist doesnt make sense

    def __init__(self, value):
        super().__init__()

        self.model_value = value

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        return tensor([1.])

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        return tensor([1.])

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        return - 0.5 * nn_galaxy_loss(self.model_value)


# nn_galaxy_log_prob = GalaxyLogProb()


def pyro_galaxy_model(n_images=1):
    # define inference parameters
    dx = pyro.sample("dx", dist.Uniform(-5.0 * torch.ones(n_images),
                                         5.0 * torch.ones(n_images))).view(1, 1, 1)

    dy = pyro.sample("dy", dist.Uniform(-5.0 * torch.ones(n_images),
                                         5.0 * torch.ones(n_images))).view(1, 1, 1)

    amplitude_bulge = pyro.sample("amplitude_bulge", dist.Uniform(0.0, 1.0)).view(1, 1, 1)

    r_eff = pyro.sample("r_eff", dist.Uniform(0.1, 20.0)).view(1, 1, 1)

    n = pyro.sample("n", dist.Uniform(0.36, 8.0)).view(1, 1, 1)

    ellip_bulge = pyro.sample("ellip_bulge", dist.Uniform(0., 1.0)).view(1, 1, 1)

    theta_bulge = pyro.sample("theta_bulge", dist.Uniform(5.0, 25.0)).view(1, 1, 1)

    amplitude_disk = pyro.sample("amplitude_disk", dist.Uniform(0.0, 1.0)).view(1, 1, 1)

    scale_height = pyro.sample("scale_height", dist.Uniform(0.0, 50.0)).view(1, 1, 1)

    ellip_disk = pyro.sample("ellip_disk", dist.Uniform(0.0, 1.0)).view(1, 1, 1)

    theta_disk = pyro.sample("theta_disk", dist.Uniform(15.0, 35.0)).view(1, 1, 1)

    galaxy = pyro.deterministic("galaxy_model",
                                nn_galaxy_model(dx, dy,
                                                amplitude_bulge, r_eff, n, ellip_bulge, theta_bulge,
                                                amplitude_disk, scale_height, ellip_disk, theta_disk)
                                )

    return pyro.sample("obs", GalaxyLogProb(galaxy))


unconditioned_pyro_galaxy_model = poutine.uncondition(pyro_galaxy_model)


def show_image_grid(images, dxs, dys, vmin=None, vmax=None):
    fig, axes = plt.subplots(3, 3, constrained_layout=True, sharex=True, sharey=True)
    axes = np.array(axes).flatten()
    for i, ax in enumerate(axes[:-1]):
        ax.imshow(images[i].cpu().numpy(),
                  origin="lower",
                  vmin=vmin, vmax=vmax)
        ax.scatter(dxs.cpu().numpy()[i] + 24 // 2,
                   dys.cpu().numpy()[i] + 24 // 2,
                   marker="+", c="white", s=50, linewidths=0.75)
    axes[-1].axis("off")
    plt.show()

if __name__ == "__main__":
    # extinction = torch.exp(-disk_models.squeeze() * tau)
    # extincted_galaxy_images = true_galaxy_images * extinction
    # noisy_extincted_galaxy_images = torch.normal(extincted_galaxy_images, yerr)
    # residuals = true_galaxy_images - noisy_extincted_galaxy_images

    # pyro.render_model(unconditioned_pyro_galaxy_model,
    #                   render_distributions=True, filename="model.pdf")

    kernel = NUTS(unconditioned_pyro_galaxy_model,
                  target_accept_prob=0.9,
                  jit_compile=True,
                  # ignore_jit_warnings=True,
                  )
    mcmc = MCMC(kernel,
                warmup_steps=50,
                num_samples=100,
                mp_context="spawn",
                )

    mcmc.run(# x=true_galaxy_images,
             # y=true_galaxy_images
    )


    print(dxs)
    print(dys)
    mcmc.summary()

    if len(dxs)>1:
        for i, (ddx, ddy) in enumerate(zip(dxs, dys)):
            truths[f"dx{i}"] = ddx.cpu().numpy()
            truths[f"dy{i}"] = ddy.cpu().numpy()
    else:
        truths[f"dx"] = dxs.item()
        truths[f"dy"] = dys.item()

    _mc_samples = mcmc.get_samples()
    mc_samples = {k: v.cpu().numpy().flatten() for k, v in _mc_samples.items()}

    print(mc_samples)
    print(type(mc_samples))
    if isinstance(mc_samples, dict):
        print(mc_samples.keys())

    from chainconsumer import ChainConsumer
    c = ChainConsumer()
    c.add_chain(mc_samples)

    fig = c.plotter.plot_distributions(truth=truths)
    plt.show()

    fig = c.plotter.plot_walks(truth=truths)
    plt.show()


    # arviz_samples = {k: v.detach().cpu().numpy() for k, v in mc_samples.items()}
    #
    # pyro_data = az.from_pyro(arviz_samples)
    #
    # az.plot_trace(arviz_samples)
    # plt.show()
    #
    # az.plot_autocorr(arviz_samples)
    # plt.show()

    # corner(mcmc)
    # plt.show()


    # print(true_galaxy_images.shape)
    # print(extinction.shape)
    # print(extincted_galaxy_images.shape)
    # print(noisy_extincted_galaxy_images.shape)
    #
    # # TRUE IMAGES
    # show_image_grid(true_galaxy_images, dxs, dys, vmin=0., vmax=0.4)
    #
    # # EXTINCTION
    # # show_image_grid(1 / extinction, dxs, dys,)  # vmin=5.e-06, vmax=1. - 1.e-07)
    #
    # # EXTINCTED IMAGES
    # show_image_grid(extincted_galaxy_images, dxs, dys, vmin=0., vmax=0.4)
    #
    # # NOISY EXTINCTED IMAGES
    # show_image_grid(noisy_extincted_galaxy_images, dxs, dys, vmin=0., vmax=0.4)
    #
    # # RESIDUAL IMAGES
    # show_image_grid(residuals, dxs, dys, vmin=-0.0110, vmax=0.011)