#!/usr/bin/python

from datetime import datetime
from argparse import ArgumentParser

import torch

import utils.data as data
import utils.training as training

from utils.baselines import Autoencoder, Unet

search_grid = {
    "n_hidden_channels": [64, 256, 1024],
    "depthness": [2],
    "lr": [1e-2, 1e-3],
    "weight_decay": [1e-8, 1e-5],
    "batch_size": [64, 16]
}



def main(
    name:str = "*time*", #*time* is replaced by the datetime
    logdir:str = "runs",
    configs:int = 5, 
    architecture:str = "autoencoder",
    device:str = "cpu"):

    #TODO: Add device check
    if torch.cuda.is_available():
        if device.isdigit():
            device_n = int(device)
            if device_n < torch.cuda.device_count():
                device = "cuda:" + device
            else:
                device = "cpu"
            
    name = name.replace("*time*", datetime.now().strftime("%d-%m-%Y_%H-%M-%S"))

    data_module = data.DataModule()


    ldir = logdir + "/" + name
    
    if architecture == "autoencoder":
        model_class = Autoencoder
    elif architecture == "unet":
        model_class = Unet
    else:
        print(f"Architecture \"{architecture}\" unknown. Stopping!")
        return

    training.search_configs(
        model_class,
        data_module, 
        search_grid, 
        randomly_try_n = configs, 
        logdir = ldir,
        device = device
        )



if __name__ == '__main__':


    parser = ArgumentParser()
    parser.add_argument("-n", "--name", help="Name of the experiment", default=f"*time*")
    parser.add_argument("-l", "--logdir", help="Directories where logs are stored", default=f"runs")
    parser.add_argument("-c", "--configs", help="Number of configs to try", default=5)
    parser.add_argument("-a", "--architecture", help="The architecture of choice", default="unet")
    parser.add_argument("-d", "--device", help="The device of choice", default="cuda")

    args = parser.parse_args()

    logdir = str(args.logdir)
    name = str(args.name)
    configs = int(args.configs)
    architecture = str(args.architecture).lower()
    device = str(args.device)

    main(name, logdir, configs, architecture, device)
