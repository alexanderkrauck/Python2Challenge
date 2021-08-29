"""
Utility classes/functions for training models
"""

__author__ = "Alexander Krauck"
__email__ = "alexander.krauck@gmail.com"
__date__ = "17-08-2021"

import itertools
import os
import pickle
from time import time
from utils.data import DataModule
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
import tqdm

from torch.utils.tensorboard import SummaryWriter


def train(
    model,
    optimizer,
    loader,
    epoch: int,
    logger: SummaryWriter,
    device="cpu",
    use_tqdm=False,
):

    model.train()

    n_minibatches = len(loader)

    if use_tqdm:
        iterate = tqdm.tqdm(loader, desc=f"Ep{epoch:02}")
    else:
        iterate = loader

    batch_nr = 0
    for (
        x,
        input,
        (left_margin, left_margin_size, top_margin, top_margin_size),
    ) in iterate:

        x, input = (
            x.unsqueeze(1).float().to(device),
            input.unsqueeze(1).float().to(device),
        )

        out = model(input)

        losses = F.mse_loss(out, x, reduction="none")
        not_target = input != -1
        full_weight = (~not_target).sum() + not_target.sum() * 0.05
        losses[not_target] *= 0.05  # questionable, maybe make it 0
        loss = torch.sum(losses) / full_weight

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        global_step = loader.batch_size * (
            n_minibatches * (epoch - 1) + batch_nr + 1
        )  # sample dependent
        logger.add_scalar(
            f"MSE-TRAIN", loss.detach().cpu().numpy(), global_step=global_step
        )

        batch_nr += 1

    logger.flush()


def validate(
    model,
    loader,
    epoch: int,
    logger: SummaryWriter,
    run_type: str = "test",
    device: str = "cpu",
    use_tqdm=False
):
    model.eval()

    if use_tqdm:
        iterate = tqdm.tqdm(loader, desc=f"Ep{epoch:02}")
    else:
        iterate = loader

    means = []
    for (
        x,
        input,
        (left_margin, left_margin_size, top_margin, top_margin_size),
    ) in iterate:
        x, input = (
            x.unsqueeze(1).float().to(device),
            input.unsqueeze(1).float().to(device),
        )

        out = model(input)

        # This should be the same format as in the challenge servers
        losses = -F.mse_loss(
            torch.max(out, torch.zeros_like(out)) * 255, x * 255, reduction="none"
        )
        losses[input != -1] *= 0
        normalization = torch.sum(input == -1, dim=[1, 2, 3])
        loss = torch.sum(losses, dim=[1, 2, 3]) / normalization

        means.extend(loss.detach().cpu().numpy())

    neg_MSE = np.mean(means)
    logger.add_scalar(f"MSE-SUBMISSION-{run_type.upper()}", neg_MSE, global_step=epoch)

    logger.flush()

    return neg_MSE


def test(model, loader, write_to, device: str = "cpu"):

    outs = []
    knowns = []
    ids = []
    for input_arrays, known_arrays, sample_ids in loader:

        out = model(input_arrays.unsqueeze(1).to(device)).squeeze(1)
        out = torch.round(out * 255)

        outs.extend(out.detach().cpu().numpy())
        knowns.extend(known_arrays.detach().cpu().numpy())
        ids.extend(list(sample_ids))

    res_list = []
    for out, known, id in zip(outs, knowns, ids):
        assert int(id) == len(res_list)

        res = out[~known]
        res_list.append(res.astype(np.uint8))

    with open(write_to, mode="wb") as file:
        pickle.dump(res_list, file)


def train_config(
    model_class,
    data_module: DataModule,
    logger: SummaryWriter,
    lr=1e-2,
    weight_decay=1e-8,
    batch_size=64,
    epochs=10,
    device="cpu",
    tqdm=False,
    use_lr_scheduler: bool = False,
    optimizer_state_dict = None,
    **kwargs,
):
    logdir = logger.get_logdir()
    if isinstance(model_class, type):
        model = model_class(data_module=data_module, **kwargs).to(device)
    else:
        model = model_class

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
    if use_lr_scheduler:
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=2, cooldown=1, verbose=True, threshold=2
        )

    train_loader = data_module.make_train_loader(batch_size=batch_size)
    val_loader = data_module.make_val_loader(batch_size=batch_size)
    test_loader = data_module.make_test_loader(batch_size=batch_size)

    best_val_score = validate(
        model,
        val_loader,
        0,
        logger,
        run_type="validation",
        device=device,
        use_tqdm=tqdm,
    )

    for epoch in range(1, epochs + 1):

        train(
            model, optimizer, train_loader, epoch, logger, device=device, use_tqdm=tqdm
        )
        val_score = validate(
            model=model,
            loader=val_loader,
            epoch=epoch,
            logger=logger,
            run_type="validation",
            device=device,
            use_tqdm=tqdm,
        )

        if use_lr_scheduler:
            lr_scheduler.step(val_score)

        if val_score > best_val_score:
            best_val_score = val_score
            test(
                model,
                test_loader,
                write_to=os.path.join(logdir, f"TestResEp{epoch}.pkl"),
                device=device,
            )

            save_dict = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "kwargs": kwargs,
            }
            torch.save(save_dict, os.path.join(logdir, f"ModelEp{epoch}.pkl"))


def dict_product(dicts):

    return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))


def search_configs(
    model_class,
    data_module,
    search_grid,
    randomly_try_n=-1,
    logdir="runs",
    device="cpu",
    tqdm=False,
):

    configurations = [config for config in dict_product(search_grid)]
    print(f"Total number of Grid-Search configurations: {len(configurations)}")

    if randomly_try_n == -1:
        do_indices = range(len(configurations))
    else:
        do_indices = np.random.choice(len(configurations), size=randomly_try_n)

    print(f"Number of configurations now being trained {len(do_indices)}")
    print(
        "--------------------------------------------------------------------------------------------\n"
    )

    for trial_nr, idx in enumerate(do_indices):

        config = configurations[idx]

        config_str = f"{trial_nr:02}" + str(config).replace("'", "").replace(
            ":", "-"
        ).replace(" ", "").replace("}", "").replace("_", "").replace(",", "_").replace(
            "{", "_"
        )

        print(f"Training config {trial_nr:02}:\n”{config}")
        dt = time()

        logger = SummaryWriter(log_dir=logdir + "/" + config_str, comment=config_str)

        train_config(
            model_class=model_class,
            data_module=data_module,
            logger=logger,
            device=device,
            tqdm=tqdm,
            **config,
        )

        print(f"Done (took {time() - dt:.2f}s)")


def load_model(model_class, location, data_module):

    model_dict = torch.load(location)
    model = model_class(data_module, **model_dict["kwargs"])
    model.load_state_dict(model_dict["model"])

    return model
