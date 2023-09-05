import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import Module
import pyro
from pyro import distributions as dist
from pyro.distributions import constraints
import pyro.optim as optim
from pyro.infer import Predictive, MCMC, NUTS, SVI, Trace_ELBO
from kornia.geometry.transform import Affine, translate

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
observations, disk_models = observations, disk_models  # observations.unsqueeze(1), disk_models.unsqueeze(1)
disk_subtracted_models = observations - disk_models
ellipse_mask = make_ellipse_mask(9, 5, 35)


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



class GalaxyImageResiduals(Module):
    def __init__(self, mask, yerr=3.0e-3):
        super().__init__()

        # only exposed to paired images/masks
        self.mask = mask.bool()
        self.yerr = yerr
        self.forward = self.chi_squared

    # def chi_squared(self, x, y):
    #     chi_sqrd = torch.square(torch.masked_select(y, self.mask) - torch.masked_select(x, self.mask)) / self.yerr ** 2
    #     return chi_sqrd

    def chi_squared(self, x, y):

        x = pyro.sample("obs_values", dist.Delta(torch.masked_select(x, self.mask)).to_event(1))
        y = pyro.sample("model_values", dist.Delta(torch.masked_select(y, self.mask)).to_event(1))

        return x, y

    def chi_squared_error_map(self, x, y):
        chi_sqrd = torch.square(y - x) / self.yerr ** 2
        return chi_sqrd



class GalaxyLogLikelihood(Module):
    def __init__(self, mask=None, yerr=3.0e-3):
        super().__init__()

        # since all images should be roughly centered on the frame center all masks can be the same
        if mask is None:
            mask = ellipse_mask
        self.mask = mask.bool()
        self.yerr = yerr

        self.get_chi_squared = GalaxyImageResiduals(mask, yerr=3.0e-3)
        self.forward = self.log_likelihood

    def log_likelihood(self, translation, tau, obs):
        chi_squared = self.chi_squared_residuals(translation, tau, obs)
        return chi_squared # torch.sum(chi_squared)

    def chi_squared_residuals(self, translation, tau, obs):
        # translation, tau = theta
        translation = pyro.sample("expanded_translation",
                                  dist.Delta(
                                      dependent_translations(translation)
                                  ).to_event(2))

        correction = torch.exp(- tau * disk_models)

        corr_obs = obs / correction

        corr_trans_model = pyro.sample("bs_corr_trans_model",
                                       dist.Delta(
                                           corr_obs[index_models] * torch.sum(torch.abs(translation))
                                       ).to_event(3))

        # corr_trans_model = pyro.sample("corr_trans_model",
        #                           dist.Delta(
        #                               translate(corr_obs[index_models].view(28, 1, 24, 24),
        #                                         translation,
        #                                         mode='bilinear',
        #                                         align_corners=True).view(28, 24, 24)
        #                                      ).to_event(3))

        return self.get_chi_squared(corr_obs[index_obs], corr_trans_model)

    def chi_squared_residual_map(self, translation, tau, obs):
        # translation, tau = theta
        translation = dependent_translations(translation)

        correction = torch.exp(- tau * disk_models)

        corr_obs = obs / correction

        corr_trans_model = translate(corr_obs[index_models].view(28, 1, 24, 24),
                                     translation,
                                     mode='bilinear',
                                     align_corners=True).view(28, 24, 24)

        return self.get_chi_squared.chi_squared_error_map(corr_obs[index_obs], corr_trans_model)



def dependent_translations(translations):
    """
    Compute all translations given the translations with respect to the first image.

    Parameters:
    - translations (Tensor): A tensor of shape (N-1, 2) containing the known translations with respect to image 0.

    Returns:
    - Tensor: A tensor containing translations for all pairs, of shape ((N-1)*N/2, 2).
    """

    # Compute all differences between known translations
    diffs = translations.unsqueeze(0) - translations.unsqueeze(1)

    # We need to extract the upper triangle (excluding the diagonal) from 'diffs'.
    # We use a mask for this.
    mask = torch.triu(torch.ones(diffs.shape[0], diffs.shape[0]), diagonal=1)
    t_ij = diffs[mask.bool()].reshape(-1, 2)

    # Combining the known translations with the computed translations
    all_translations = torch.cat([translations, t_ij], dim=0)

    return all_translations

log_likelihood_fun = GalaxyLogLikelihood()


def model(obs):
    # define inference parameters
    scale_scale = 1.0e-2
    # delta_loc = pyro.param("delta_loc", torch.zeros(N_IMAGES - 1, 2))
    # delta_scale = pyro.param("delta_scale", scale_scale * torch.ones(N_IMAGES - 1, 2), constraint=constraints.positive)
    # translation = pyro.sample("translation", dist.Normal(delta_loc, delta_scale).to_event(2))

    delta_loc = pyro.param("delta_loc", torch.zeros((N_IMAGES - 1) * 2))
    delta_scale = pyro.param("delta_scale", scale_scale * torch.ones((N_IMAGES - 1) * 2), constraint=constraints.positive)
    translation = pyro.sample("translation", dist.Normal(delta_loc, delta_scale).to_event(1)).view(7, 2)

    # Variational parameters for log_tau
    tau_df = pyro.param("tau_df", torch.tensor(0.7), constraint=constraints.positive)
    tau = pyro.sample("tau", dist.Chi2(tau_df))

    # theta = [translation, tau]

    # Likelihood
    # likelihood = log_likelihood_fun(theta, obs)
    _obs, _model = log_likelihood_fun(translation, tau, obs)
    # using Delta atm, consider looking into empirical distributions
    # when using empirical potential to weight instead of mask
    return pyro.sample("obs", dist.Normal(_model, 3.0e-3).to_event(1), obs=_obs) # .to_event(1) # torch.sum(s) # s # pyro.sample("obs", dist.Delta(_model).to_event(1), obs=_obs)


def guide(obs):
    scale_scale = 1.0e-2
    # for some reason clamping doest work:
    # i.e log_tau = pyro.sample("log_tau", dist.Normal(log_tau_loc, log_tau_scale)).clamp(-10., 5.)
    # can still produce values outside the support: Error:
    # Expected value argument (Tensor of shape ()) to be within the support (Interval(lower_bound=-100, upper_bound=50.0))
    # of the distribution Uniform(low: -100.0, high: 50.0), but found invalid values: 50.434879302978516

    # Variational parameters for dx
    # dx_loc = pyro.param("dx_loc", torch.zeros(N_IMAGES - 1))
    # dx_scale = pyro.param("dx_scale", scale_scale * torch.ones(N_IMAGES - 1), constraint=constraints.positive)
    # dx = pyro.sample("dx", dist.Normal(dx_loc, dx_scale).to_event(1))  # , constraints=constraints.interval(-5., 5.))
    #
    # # Variational parameters for dy
    # dy_loc = pyro.param("dy_loc", torch.zeros(N_IMAGES - 1))
    # dy_scale = pyro.param("dy_scale", scale_scale * torch.ones(N_IMAGES - 1), constraint=constraints.positive)
    # dy = pyro.sample("dy", dist.Normal(dy_loc, dy_scale).to_event(1))  # , constraints=constraints.interval(-5., 5.))

    delta_loc = pyro.param("delta_loc", torch.zeros((N_IMAGES - 1) * 2))
    delta_scale = pyro.param("delta_scale", scale_scale * torch.ones((N_IMAGES - 1) * 2), constraint=constraints.positive)
    translation = pyro.sample("translation", dist.Normal(delta_loc, delta_scale).to_event(1)).view(7, 2)

    # Variational parameters for log_tau
    tau_df = pyro.param("tau_df", torch.tensor(0.7), constraint=constraints.positive)
    tau = pyro.sample("tau", dist.Chi2(tau_df))  # , constraints=constraints.interval(-10., 1.))

    # How can I implement a turncated gaussian with constraints in pyro?
    # log_tau = pyro.sample("log_tau", dist.Normal(log_tau_loc, log_tau_scale, constraints=constraints.interval(-10., 1.)))

if __name__ == "__main__":
    from astropy.io import fits
    print(observations.shape)
    print(disk_subtracted_models.shape)
    print(disk_models.shape)
    print(ellipse_mask.shape)

    # hdu = fits.PrimaryHDU(disk_subtracted_models.squeeze().cpu().numpy())
    #
    # hdu.writeto(
    #     '/home/lwelzel/Documents/git/debris-disk-photometry/data/raw_data/hd107146_synthetic/reduced_cube.fits',
    #     overwrite=True,
    # )

    hdul = fits.open("/home/lwelzel/Documents/git/debris-disk-photometry/data/raw_data/hd107146_synthetic/synthetic_cube.fits")
    synthetic_obs = torch.tensor(hdul[0].data.astype(np.float32))

    # fig, axes = plt.subplots(3, 3, constrained_layout=True,
    #                          sharex=True, sharey=True,
    #                          figsize=(12, 12))
    # axes = np.array(axes).flatten()
    #
    # for i in range(len(synthetic_obs)):
    #     im = axes[i].contourf(synthetic_obs[i].cpu().numpy(), origin="lower")
    #     plt.colorbar(im, ax=axes[i])
    #
    # plt.show()
    #
    # dx = torch.zeros(N_IMAGES - 1) + torch.normal(0.1 * torch.ones(N_IMAGES - 1), 0.2)
    # dy = torch.zeros(N_IMAGES - 1) + torch.normal(0.1 * torch.ones(N_IMAGES - 1), 0.2)
    # tau = 1.
    # translation = torch.stack((dx, dy), 1)
    # theta = [translation, tau]
    #
    # res = log_likelihood_fun.chi_squared_residual_map(theta, synthetic_obs)
    # print("res", res.shape)
    #
    # fig, axes = plt.subplots(6, 6, constrained_layout=True,
    #                          sharex=True, sharey=True,
    #                          figsize=(18, 18))
    # axes = np.array(axes).flatten()
    #
    # for i in range(len(res)):
    #     im = axes[i].contourf(res[i].cpu().numpy(), origin="lower")
    #     plt.colorbar(im, ax=axes[i])
    #
    # plt.show()

    # Set up the optimizer
    adam = optim.ClippedAdam({
        "lr": 5.0e-2,
        "lrd": 0.95,
    })

    # Set up the inference algorithm
    svi = SVI(model, guide, adam, loss=Trace_ELBO())

    from tqdm import tqdm

    pyro.render_model(model, model_args=(synthetic_obs, ),
                      render_distributions=True, filename="model.pdf")

    num_steps = 500
    losses = torch.zeros(num_steps)

    def train(num_iterations, data):
        pyro.clear_param_store()
        pbar = tqdm(range(num_iterations), total=num_iterations)
        _mean_loss = 0
        for j in pbar:
            # calculate the loss and take a gradient step
            loss = svi.step(data)
            _mean_loss += loss
            if j % 50 == 0:
                pbar.set_description(f"Mean loss: {_mean_loss / 50.:.1f}")
                _mean_loss = 0
    train(num_steps, synthetic_obs)

    delta_loc = pyro.param("delta_loc").detach()
    tau_df = pyro.param("tau_df").detach()

    print("Learned deltas:", delta_loc)
    print("Learned tau_df:", tau_df)

    import arviz as az

    predictive = Predictive(model, guide=guide, num_samples=500)
    preds = predictive(synthetic_obs)
    sanitized_preds = {k: v.unsqueeze(0).detach().cpu().numpy() for k, v in preds.items() if k in ['translation', 'tau']}

    import corner

    corner.corner(sanitized_preds, show_titles=True)

    plt.show()

    raise NotImplementedError

    pyro_data = az.convert_to_inference_data(sanitized_preds)
    az.plot_trace(pyro_data, compact=True)

    plt.show()

    ax = az.plot_pair(
        pyro_data,
        kind=["scatter", "kde"],
        kde_kwargs={"fill_last": False},
        marginals=True,
        point_estimate="median",
        figsize=(12, 12),
    )

    plt.show()


    # paired_obs = disk_subtracted_models[index_obs]
    # paired_models = disk_subtracted_models[index_models]
    #
    # paired_disk_obs = disk_models[index_obs]
    # paired_disk_models = disk_models[index_models]
    #
    # paired_masks = ellipse_mask[index_obs]

    # log_likelihood = GalaxyLogLikelihood(mask=ellipse_mask[0].view(1, 24, 24))
    #
    # theta = [torch.tensor([[0.01, 0.01]]), 0.]
    # chi_squared = log_likelihood(theta, disk_subtracted_models[0].view(1, 24, 24))
    #
    # print(chi_squared)
    #
    # residual = log_likelihood.residuals(theta, disk_subtracted_models[0].view(1, 24, 24))
    #
    # im = plt.hist(residual.cpu().numpy())
    # plt.show()

    # o1, o2 = pair_indices(4)
    #
    # print(len(o1))
    #
    # print(o1)
    # print(o2)
    #
    # trans = torch.tensor([[-1., -1],
    #                       [2, -2],
    #                       [1, 1]])
    #
    # print(trans)
    # print()
    # print(dependent_translations(trans))
    #print(len(dependent_translations(trans)))

    # tau = 1.e-1
    # trans = torch.tensor([[-0.01, -0.1]])
    #
    # correction = torch.exp(- tau * disk_models[0])
    # fake_obs = disk_subtracted_models[0].view(1, 24, 24) # / correction
    # fake_obs = translate(fake_obs,
    #                      trans,
    #                      mode='bilinear',
    #                      align_corners=True)

    # fake_obs = fake_obs * corr


    # pyro.render_model(model, model_args=(disk_subtracted_models[1:], disk_subtracted_models[0]),
    #                   render_distributions=True, filename="model.pdf")
    #
