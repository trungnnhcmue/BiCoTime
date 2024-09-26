import logging
import os
import random
import numpy as np
import torch

def complex_mul(a, b):
    assert a.size(-1) == b.size(-1)
    dim = a.size(-1) // 2
    a_1, a_2 = torch.split(a, dim, dim=-1)
    b_1, b_2 = torch.split(b, dim, dim=-1)

    A = a_1 * b_1 - a_2 * b_2
    B = a_1 * b_2 + a_2 * b_1 

    return torch.cat([A,B], dim=-1)

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    logging.info(f"Random seed set as {seed}")

def save_model(save_path, args, model: torch.nn.Module, optim: torch.optim.Optimizer, result: dict, epoch: int):
    """
    Function to save a model. It saves the model parameters, best validation scores,
    best epoch corresponding to best validation, state of the optimizer and all arguments for the run.
    Parameters
    ----------
    save_path: path where the model is saved

    Returns
    -------
    """
    state = {
        'state_dict'	: model.state_dict(),
        'optimizer'	: optim.state_dict(),
        'args'		: vars(args),
        'current_epoch': epoch,
        'result': result
    }
    torch.save(state, save_path)

def load_model(load_path, model: torch.nn.Module, optim: torch.optim.Optimizer):
    """
    Function to load a saved model
    Parameters
    ----------
    load_path: path to the saved model

    Returns
    -------
    """
    state = torch.load(load_path)
    state_dict = state['state_dict']

    model.load_state_dict(state_dict)
    optim.load_state_dict(state['optimizer'])
    return model, optim, state['result'], state['current_epoch']

def get_param(shape):
    param = torch.nn.Parameter(torch.Tensor(*shape));
    torch.nn.init.xavier_normal_(param.data)
    return param