import numpy as np
import torch
from torch import tensor
from torch.nn import Module
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

class Decoder(GalaxyModel):
    def __init__(self, h=24, w=24, n_images=1):
        super().__init__(h=h, w=w, n_images=n_images)



class Encoder(Module):
    pass