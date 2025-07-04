# ---------------------------------------------------------*\
# Title: Optimizer Loader
# Author: TM
# ---------------------------------------------------------*/

import torch
import dadaptation
from optimizers.adam_clara import Adam_CLARA
from optimizers.sgd_clara import SGD_CLARA


# -------------------- Optimizer Factory --------------------


def get_optimizer(optimizer_name, model_parameters, learning_rate=0.001, damping=1e-3):  # TODO: Add damping in SGD_CLARA and Adam_CLARA
    """Returns the requested optimizer initialized with model parameters."""

    optimizer_name = optimizer_name.lower()

    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model_parameters, lr=learning_rate)
    elif optimizer_name == "sgdmomentum":
        optimizer = torch.optim.SGD(model_parameters, lr=learning_rate, momentum=0.9)
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(model_parameters, lr=learning_rate)
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model_parameters, lr=learning_rate)
    elif optimizer_name == "rmsprop":
        optimizer = torch.optim.RMSprop(model_parameters, lr=learning_rate)
    elif optimizer_name == "d-adaptation":
        optimizer = dadaptation.DAdaptAdam(model_parameters, d0=learning_rate)  # To emulate initial lr value comparable to the rest of optimizers'
    elif optimizer_name == "sgd_clara":
        return SGD_CLARA(model_parameters, d=damping, lr=learning_rate, unit_step_direction=False)
    elif optimizer_name == "sgd_clara_us":  # CLARA with unit (normalized) steps
        return SGD_CLARA(model_parameters, d=damping, lr=learning_rate, unit_step_direction=True)
    elif optimizer_name == "adam_clara":
        return Adam_CLARA(model_parameters, d=damping, lr=learning_rate, unit_step_direction=False)
    elif optimizer_name == "adam_clara_us":
        return Adam_CLARA(model_parameters, d=damping, lr=learning_rate, unit_step_direction=True)
    else:
        raise ValueError(f"Optimizer {optimizer_name} is not supported.")

    return optimizer


# -------------------------Notes-----------------------------------------------*\
#
# -----------------------------------------------------------------------------*\
