import argparse
import json
import logging
import random
import os

import tqdm
import numpy as np
import core

import matplotlib
matplotlib.use('Agg')


def plot_loss(data, use_mean=True):
    import matplotlib.pyplot as plt

    keys = data.keys()
    x = data.pop("epoch")[:, 0]
    ys = np.vstack([data[k] for k in keys]).T

    fig, ax = plt.subplots(1, len(keys), figsize=(5 * len(keys), 5))
    maxs = max([d.max() for k, d in data.items() if "loss" in k])

    for i, k in enumerate(keys):
        if use_mean:
            ax[i].plot(x, np.mean(data[k], axis=1))
        else:
            ax[i].plot(x, data[k])

        # ax[i].legend([k])

        if "loss" in k:
            ax[i].set_ylim([0, maxs])
        ax[i].set_title(k)
        ax[i].set_xlabel("Epoch")
        ax[i].set_ylabel("Loss")
        # ax[i].set_yscale('log')

    fig.tight_layout()
    return fig


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(description="")
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory which includes specifications and saved model "
        + "files to use for reconstruction",
    )
    arg_parser.add_argument(
        "--mean",
        default=False,
        action="store_true",
    )

    # >>> end update
    core.add_common_args(arg_parser)
    args = arg_parser.parse_args()
    core.configure_logging(args)

    # Get specs
    logging.info("Loading specs from: {}".format(args.experiment_directory))
    specs = json.load(open(core.find_specs(args.experiment_directory)))

    # Create and load the dataset
    try:
        sdf_dataset = core.data.SamplesDataset(
            split=core.quick_load_path(
                specs["TestSplit"].replace("$DATADIR", os.environ["DATADIR"])
            ),
            root=specs["DataSource"].replace("$DATADIR", os.environ["DATADIR"]),
        )
        logging.info("Quickload failed, reverting to default load")
    except FileNotFoundError:
        sdf_dataset = core.data.SamplesDataset(
            split=specs["TestSplit"].replace("$DATADIR", os.environ["DATADIR"]),
            root=specs["DataSource"].replace("$DATADIR", os.environ["DATADIR"]),
        )

    # Load handler
    reconstruction_handler = core.handler.quick_load_handler(
        args.experiment_directory, json=True
    )

    reconstruct_list = range(len(sdf_dataset))

    loss_dict = None
    # Load the individual losses
    for r in reconstruct_list:
        loss_path = core.lossify_log_path(reconstruction_handler.path_code(r)) + ".npy"
        if not os.path.exists(loss_path):
            logging.debug("Couldnt find code: {}".format(loss_path))
            continue

        losses = core.loader(loss_path).item()
        if loss_dict is None:
            loss_dict = {k: list() for k in losses.keys()}

        # Accumulate the losses
        for key in loss_dict:
            loss_dict[key].append(losses[key])

    # Turn them into a 2d numpy array
    for key in loss_dict:
        loss_dict[key] = np.array(loss_dict[key])

    # Now take the mean
    for key in loss_dict:
        # loss_dict[key] = loss_dict[key].mean(axis=0)
        loss_dict[key] = loss_dict[key].T

    plot_loss(
        data=loss_dict,
        use_mean=args.mean,
    ).savefig("losses_{}.png".format(core.get_dataset(args.experiment_directory)))
