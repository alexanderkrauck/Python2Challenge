
"""
Utility classes/functions for training models
"""

__author__ = "Alexander Krauck"
__email__ = "alexander.krauck@gmail.com"
__date__ = "17-08-2021"

import itertools
from time import time
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
import tqdm

from torch.utils.tensorboard import SummaryWriter


def train(model, optimizer, loader, epoch: int, logger: SummaryWriter, device = "cpu", use_tqdm = False):

    model.train()

    n_minibatches = len(loader)

    if use_tqdm:
        iterate = tqdm.tqdm(loader)
    else:
        iterate = loader

    batch_nr = 0
    for x, input, (left_margin, left_margin_size, top_margin, top_margin_size) in iterate:

        x, input = x.unsqueeze(1).float().to(device), input.unsqueeze(1).float().to(device)

        out = model(input)


        losses = F.mse_loss(x, out, reduction="none")
        losses[input != -1] *= 0.05 #questionable
        loss = torch.mean(losses)

        optimizer.zero_grad() 
        loss.backward()
        optimizer.step()

        global_step = loader.batch_size * (n_minibatches * (epoch - 1) + batch_nr + 1)#sample dependent
        logger.add_scalar(f"MSE-TRAIN", loss.detach().cpu().numpy(), global_step=global_step)

        batch_nr += 1

    logger.flush()

        

def test(model, loader, epoch:int, logger: SummaryWriter, run_type:str = "test", device:str = "cpu", use_tqdm = False):
    model.eval()

    if use_tqdm:
        iterate = tqdm.tqdm(loader)
    else:
        iterate = loader

    means = []
    for x, input, (left_margin, left_margin_size, top_margin, top_margin_size) in iterate:
        x, input = x.unsqueeze(1).float().to(device), input.unsqueeze(1).float().to(device)

        out = model(input)
        losses = F.mse_loss(x, out, reduction="none")
        losses[input != -1] *= 0
        loss = torch.mean(losses, dim=[1,2,3])
        means.extend(loss.detach().cpu().numpy())

    logger.add_scalar(f"MSE-{run_type.upper()}", np.mean(means), global_step=epoch)
    
    logger.flush()

def train_config(
    model_class,
    data_module,
    logger: SummaryWriter,
    lr = 1e-2,
    weight_decay = 1e-8,
    batch_size = 64,
    epochs = 100, 
    device = "cpu",
    **kwargs
    ):

    model = model_class(
        data_module = data_module,
        **kwargs
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


    train_loader = data_module.make_train_loader(batch_size = batch_size)
    val_loader = data_module.make_val_loader(batch_size = batch_size)

    test(model, val_loader, 0, logger, run_type="validation", device = device)
    test(model, train_loader, 0, logger, run_type="train", device = device)

    for epoch in range(1, epochs + 1):
    
        train(model, optimizer, train_loader, epoch, logger, device = device)
        test(model, val_loader, epoch, logger, run_type="validation", device = device)

def dict_product(dicts):

    return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))

def search_configs(model_class, data_module, search_grid, randomly_try_n = -1, logdir = "runs", device = "cpu"):

    configurations = [config for config in dict_product(search_grid)]
    print(f"Total number of Grid-Search configurations: {len(configurations)}")

    if randomly_try_n == -1:
        do_indices = range(len(configurations))
    else:
        do_indices = np.random.choice(len(configurations), size=randomly_try_n)
    
    print(f"Number of configurations now being trained {len(do_indices)}")
    print("--------------------------------------------------------------------------------------------\n")
    
    for trial_nr, idx in enumerate(do_indices):
        
        config = configurations[idx]


        config_str = str(config).replace("'","").replace(":", "-").replace(" ", "").replace("}", "").replace("_","").replace(",", "_").replace("{","_")

        print(f"Training config {config_str} ... ", end="")
        dt = time()
    

        logger = SummaryWriter(log_dir = logdir + "/" + config_str, comment = config_str)
        #logger.add_hparams(config,  ,run_name= f"run{trial_nr}")

        train_config(
            model_class = model_class,
            data_module = data_module,
            logger = logger,
            n_hidden_channels = config["n_hidden_channels"], 
            depthness = config["depthness"],
            weight_decay= config["weight_decay"],
            lr =  config["lr"], 
            batch_size = config["batch_size"],
            device = device
            )
            
        print(f"done (took {time() - dt:.2f}s)")
