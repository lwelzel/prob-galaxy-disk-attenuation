import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import Module, Parameter, Sequential
import pyro
from pyro.nn import PyroModule, PyroParam, PyroSample
from pyro import distributions as dist
from pyro.distributions import constraints
import pyro.optim as optim
from pyro.infer import Predictive, MCMC, NUTS, SVI, Trace_ELBO
from pyro.infer.autoguide import AutoNormal
import kornia
from kornia.geometry.transform import Affine, Translate
from kornia.geometry.transform import translate

from dataloader import load_and_center_data, make_ellipse_mask
from astropy.io import fits


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
observations, disk_models = observations, disk_models
disk_subtracted_models = observations - disk_models
ellipse_mask = make_ellipse_mask(9, 5, 35)

hdul = fits.open(
    "/home/lwelzel/Documents/git/debris-disk-photometry/data/raw_data/hd107146_synthetic/synthetic_cube.fits")
synthetic_obs = torch.tensor(hdul[0].data.astype(np.float32))


N_IMAGES = len(observations)

def pair_indices(n):
    # Create a tensor of indices
    indices = torch.arange(n)

    # Generate a meshgrid of indices
    i, j = torch.meshgrid(indices, indices, indexing="ij")

    # Filter out redundant pairs and diagonal entries (we don't want images paired with themselves)
    mask = i < j

    return i[mask], j[mask]


index_obs, index_models = pair_indices(N_IMAGES)

class PyroInvAttenuate(PyroModule):
    def __init__(self, log_tau_loc=-torch.ones(1), log_tau_scale=1.e-5 * torch.ones(1), disk_model=disk_models):
        super(PyroInvAttenuate, self).__init__()

        self.log_tau_loc = PyroParam(log_tau_loc)
        self.log_tau_scale = PyroParam(log_tau_scale, constraint=constraints.positive)

        self.log_tau = PyroSample(
            dist.Normal(self.log_tau_loc, self.log_tau_scale).to_event(1))

        self.tau = pyro.deterministic("tau", torch.pow(10, self.log_tau))

        self.disk_models = disk_model


    def forward(self, input):

        correction = torch.exp(- self.tau * self.disk_models[1:].view(-1, 1, 24, 24))

        corrected_input = input / correction

        return corrected_input


class PyroTranslation(PyroModule):
    def __init__(self, translation_loc=torch.zeros(14), translation_scale=1.e-5 * torch.ones(14)):
        # https://github.com/kornia/kornia/issues/682
        super(PyroTranslation, self).__init__()
        self.translation_loc = PyroParam(translation_loc)
        self.translation_scale = PyroParam(translation_scale, constraint=constraints.positive)
        self.translation = PyroSample(
            dist.Normal(self.translation_loc, self.translation_scale).to_event(1))

    def forward(self, input):
        out = translate(input, self.translation.view(7, 2))
        return out

class PyroLogLikelihood(PyroModule):
    def __init__(self, mask=None, yerr=3.0e-3):
        super().__init__()

        # since all images should be roughly centered on the frame center all masks can be the same
        if mask is None:
            mask = ellipse_mask
        self.mask = mask.bool()
        self.yerr = yerr

        self.transform = PyroModule[Sequential](
            PyroInvAttenuate(),
            PyroTranslation(),
        )

    def forward(self, input, observed=None):
        transformed_input = self.transform(input.view(-1, 1, 24, 24)).view(-1, 24, 24)
        transformed_input = transformed_input \
                            * torch.exp(- torch.pow(10, self.transform[0].log_tau)
                                        * self.transform[0].disk_models[0].view(1, 24, 24))

        pyro.sample("obs", dist.Delta(transformed_input).to_event(3), obs=observed)

        return transformed_input  # - observed # pyro.sample("obs", dist.Delta(transformed_input).to_event(3), obs=observed)


if __name__ == "__main__":
    fig, axes = plt.subplots(3, 3, constrained_layout=True,
                             sharex=True, sharey=True,
                             figsize=(12, 12))
    axes = np.array(axes).flatten()

    for i in range(len(synthetic_obs)):
        im = axes[i].imshow(synthetic_obs[i].cpu().numpy(), origin="lower", vmin=0., vmax=0.45)
        plt.colorbar(im, ax=axes[i])

    plt.show()

    model = PyroLogLikelihood()
    guide = AutoNormal(model)

    observed = synthetic_obs[1:]
    model_basis = torch.ones_like(synthetic_obs[1:]) * synthetic_obs[0]

    adam = pyro.optim.Adam({"lr": 0.03})
    svi = SVI(model, guide, adam, loss=Trace_ELBO())

    pyro.render_model(model, model_args=(model_basis, observed),
                      render_distributions=True, render_params=True, filename="model.pdf")


    with pyro.poutine.trace() as tr:
        loss = model(model_basis, observed)
        print(loss.shape)

        fig, axes = plt.subplots(3, 3, constrained_layout=True,
                                 sharex=True, sharey=True,
                                 figsize=(12, 12))
        axes = np.array(axes).flatten()

        for i in range(len(loss)):
            im = axes[i].imshow(loss[i].detach().cpu().numpy(), origin="lower", vmin=0., vmax=0.45)
            plt.colorbar(im, ax=axes[i])

        plt.show()

    print("1", type(model).__name__)
    print("2", list(tr.trace.nodes.keys()))
    print("3", list(pyro.get_param_store().keys()))
    print("4", "params after:", [name for name, _ in model.named_parameters()])

    num_iterations = 500
    pyro.clear_param_store()
    for j in range(num_iterations):
        # calculate the loss and take a gradient step
        loss = svi.step(model_basis, observed)
        if j % 100 == 0:
            print("[iteration %04d] loss: %.4f" % (j + 1, loss))

    raise NotImplementedError


    with pyro.poutine.trace() as tr:
        loss = pyro_model(model_basis, observed)
        print(loss.shape)

        fig, axes = plt.subplots(3, 3, constrained_layout=True,
                                 sharex=True, sharey=True,
                                 figsize=(12, 12))
        axes = np.array(axes).flatten()

        for i in range(len(loss)):
            im = axes[i].imshow(loss[i].detach().cpu().numpy(), origin="lower", vmin=0., vmax=0.45)
            plt.colorbar(im, ax=axes[i])

        plt.show()

    print("1", type(pyro_model).__name__)
    print("2", list(tr.trace.nodes.keys()))
    print("3", list(pyro.get_param_store().keys()))
    print("4", "params after:", [name for name, _ in pyro_model.named_parameters()])


    pyro.render_model(pyro_model, model_args=(model_basis, observed),
                      render_distributions=True, filename="model.pdf")


    # fig, axes = plt.subplots(3, 3, constrained_layout=True,
    #                          sharex=True, sharey=True,
    #                          figsize=(12, 12))
    # axes = np.array(axes).flatten()
    #
    # for i in range(len(translated_obs)):
    #     im = axes[i].imshow(translated_obs[i].detach().cpu().numpy(), origin="lower", vmin=0.)
    #     plt.colorbar(im, ax=axes[i])
    #
    # plt.show()