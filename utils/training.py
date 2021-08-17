
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
    for input_x, other, full_x in iterate:
        x, full_x = x.float().to(device), full_x.float().to(device)

        out = model(x)


        loss = F.mse_loss(full_x, out)

        optimizer.zero_grad() 
        loss.backward()
        optimizer.step()

        global_step = loader.batch_size * (n_minibatches * (epoch - 1) + batch_nr + 1)#sample dependent
        logger.add_scalar(f"MSE", loss.detach().cpu().numpy(), global_step=global_step)

        batch_nr += 1

    logger.flush()

        

def test(model, loader, epoch:int, logger: SummaryWriter, run_type = "test", device = "cpu", use_tqdm = False):
    model.eval()

    if use_tqdm:
        iterate = tqdm.tqdm(loader)
    else:
        iterate = loader

    for input_x, other, full_x in iterate:
        x, full_x = x.float().to(device), full_x.float().to(device)

        out = model(x)

        #TODO: calculate changed approx (where the mid part is changed)
    
    logger.flush()

def train_config(
    model_class,
    data_module,
    logger: SummaryWriter,
    hidden_channels = 128,
    depthness = 2, 
    lr = 1e-2,
    weight_decay = 1e-8,
    batch_size = 64,
    epochs = 100, 
    device = "cpu"
    ):

    model = model_class(
        data_module = data_module,
        n_hidden_channels = hidden_channels,
        depthness = depthness
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


    train_loader = data_module.make_train_loader(batch_size = batch_size)
    val_loader = data_module.make_val_loader()

    test(model, train_loader, 0, logger, run_type="train", device = device)
    test(model, val_loader, 0, logger, run_type="validation", device = device)

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
            hidden_channels = config["hidden_channels"], 
            head_depth = config["depthness"],
            weight_decay= config["weight_decay"],
            lr =  config["lr"], 
            batch_size = config["batch_size"],
            device = device
            )
            
        print(f"done (took {time() - dt:.2f}s)")
