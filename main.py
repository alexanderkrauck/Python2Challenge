#!/usr/bin/python

from datetime import datetime
from argparse import ArgumentParser

import torch

import utils.data as data
import utils.training as training

from utils.baselines import Autoencoder, Unet

search_grid = {
    "lr": [1e-3],
    "weight_decay": [1e-8],
    "batch_size": [16],
    "use_sigmoid_out": [True],
    "use_lr_scheduler": [True],
    "epochs": [50],
}


def main(
    name: str = "*time*",  # *time* is replaced by the datetime
    logdir: str = "runs",
    configs: int = 5,
    architecture: str = "autoencoder",
    device: str = "cpu",
    tqdm: bool = False,
):

    # TODO: Add device check
    if torch.cuda.is_available():
        if device.isdigit():
            device_n = int(device)
            if device_n < torch.cuda.device_count():
                device = "cuda:" + device
            else:
                device = "cpu"

    name = name.replace("*time*", datetime.now().strftime("%d-%m-%Y_%H-%M-%S"))

    data_module = data.DataModule(do_train_image_augmentation=False)

    ldir = logdir + "/" + name

    if architecture == "autoencoder":
        model_class = Autoencoder
    elif architecture == "unet":
        model_class = Unet
    else:
        print(f'Architecture "{architecture}" unknown. Stopping!')
        return

    training.search_configs(
        model_class,
        data_module,
        search_grid,
        randomly_try_n=configs,
        logdir=ldir,
        device=device,
        tqdm=tqdm,
    )


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "-n", "--name", help="Name of the experiment", default=f"augmentation_try"
    )
    parser.add_argument(
        "-l", "--logdir", help="Directories where logs are stored", default=f"runs"
    )
    parser.add_argument("-c", "--configs", help="Number of configs to try", default=-1)
    parser.add_argument(
        "-a", "--architecture", help="The architecture of choice", default="unet"
    )
    parser.add_argument("-d", "--device", help="The device of choice", default="cuda")
    parser.add_argument("-t", "--tqdm", help="If tqdm should be used", default="True")

    args = parser.parse_args()

    logdir = str(args.logdir)
    name = str(args.name)
    configs = int(args.configs)
    architecture = str(args.architecture).lower()
    device = str(args.device)
    tqdm = bool(args.tqdm)

    main(name, logdir, configs, architecture, device, tqdm)
