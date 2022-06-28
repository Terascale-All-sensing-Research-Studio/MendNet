import torch
import torch.utils.data as data_utils
import signal
import sys
import os
import logging
import math
import time

import socket
import tqdm
import neptune
import numpy as np
from sklearn.metrics import accuracy_score
import core

import core.workspace as ws


class LearningRateSchedule:
    def get_learning_rate(self, epoch):
        pass


class ConstantLearningRateSchedule(LearningRateSchedule):
    def __init__(self, value):
        self.value = value

    def get_learning_rate(self, epoch):
        return self.value


class StepLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, interval, factor):
        self.initial = initial
        self.interval = interval
        self.factor = factor

    def get_learning_rate(self, epoch):

        return self.initial * (self.factor ** (epoch // self.interval))


class WarmupLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, warmed_up, length):
        self.initial = initial
        self.warmed_up = warmed_up
        self.length = length

    def get_learning_rate(self, epoch):
        if epoch > self.length:
            return self.warmed_up
        return self.initial + (self.warmed_up - self.initial) * epoch / self.length


def get_learning_rate_schedules(specs):

    schedule_specs = specs["LearningRateSchedule"]

    schedules = []

    for schedule_specs in schedule_specs:

        if schedule_specs["Type"] == "Step":
            schedules.append(
                StepLearningRateSchedule(
                    schedule_specs["Initial"],
                    schedule_specs["Interval"],
                    schedule_specs["Factor"],
                )
            )
        elif schedule_specs["Type"] == "Warmup":
            schedules.append(
                WarmupLearningRateSchedule(
                    schedule_specs["Initial"],
                    schedule_specs["Final"],
                    schedule_specs["Length"],
                )
            )
        elif schedule_specs["Type"] == "Constant":
            schedules.append(ConstantLearningRateSchedule(schedule_specs["Value"]))

        else:
            raise Exception(
                'no known learning rate schedule of type "{}"'.format(
                    schedule_specs["Type"]
                )
            )

    return schedules


def save_model(experiment_directory, filename, decoder, epoch):

    model_params_dir = ws.get_model_params_dir(experiment_directory, True)

    torch.save(
        {"epoch": epoch, "model_state_dict": decoder.state_dict()},
        os.path.join(model_params_dir, filename),
    )


def save_optimizer(experiment_directory, filename, optimizer, epoch):

    optimizer_params_dir = ws.get_optimizer_params_dir(experiment_directory, True)

    torch.save(
        {"epoch": epoch, "optimizer_state_dict": optimizer.state_dict()},
        os.path.join(optimizer_params_dir, filename),
    )


def load_optimizer(experiment_directory, filename, optimizer):

    full_filename = os.path.join(
        ws.get_optimizer_params_dir(experiment_directory), filename
    )

    if not os.path.isfile(full_filename):
        raise Exception(
            'optimizer state dict "{}" does not exist'.format(full_filename)
        )

    data = torch.load(full_filename)

    optimizer.load_state_dict(data["optimizer_state_dict"])

    return data["epoch"]


def save_latent_vectors(experiment_directory, filename, latent_vec, epoch):

    latent_codes_dir = ws.get_latent_codes_dir(experiment_directory, True)

    all_latents = latent_vec.state_dict()

    torch.save(
        {"epoch": epoch, "latent_codes": all_latents},
        os.path.join(latent_codes_dir, filename),
    )


# TODO: duplicated in workspace
def load_latent_vectors(experiment_directory, filename, lat_vecs):

    full_filename = os.path.join(
        ws.get_latent_codes_dir(experiment_directory), filename
    )

    if not os.path.isfile(full_filename):
        raise Exception('latent state file "{}" does not exist'.format(full_filename))

    data = torch.load(full_filename)

    if isinstance(data["latent_codes"], torch.Tensor):

        # for backwards compatibility
        if not lat_vecs.num_embeddings == data["latent_codes"].size()[0]:
            raise Exception(
                "num latent codes mismatched: {} vs {}".format(
                    lat_vecs.num_embeddings, data["latent_codes"].size()[0]
                )
            )

        if not lat_vecs.embedding_dim == data["latent_codes"].size()[2]:
            raise Exception("latent code dimensionality mismatch")

        for i, lat_vec in enumerate(data["latent_codes"]):
            lat_vecs.weight.data[i, :] = lat_vec

    else:
        lat_vecs.load_state_dict(data["latent_codes"])

    return data["epoch"]


def save_logs(
    experiment_directory,
    loss_log,
    lr_log,
    timing_log,
    lat_mag_log,
    param_mag_log,
    epoch,
):

    torch.save(
        {
            "epoch": epoch,
            "loss": loss_log,
            "learning_rate": lr_log,
            "timing": timing_log,
            "latent_magnitude": lat_mag_log,
            "param_magnitude": param_mag_log,
        },
        os.path.join(experiment_directory, ws.logs_filename),
    )


def load_logs(experiment_directory):

    full_filename = os.path.join(experiment_directory, ws.logs_filename)

    if not os.path.isfile(full_filename):
        raise Exception('log file "{}" does not exist'.format(full_filename))

    data = torch.load(full_filename)

    return (
        data["loss"],
        data["learning_rate"],
        data["timing"],
        data["latent_magnitude"],
        data["param_magnitude"],
        data["epoch"],
    )


def clip_logs(loss_log, lr_log, timing_log, lat_mag_log, param_mag_log, epoch):

    iters_per_epoch = len(loss_log) // len(lr_log)

    loss_log = loss_log[: (iters_per_epoch * epoch)]
    lr_log = lr_log[:epoch]
    timing_log = timing_log[:epoch]
    lat_mag_log = lat_mag_log[:epoch]
    for n in param_mag_log:
        param_mag_log[n] = param_mag_log[n][:epoch]

    return (loss_log, lr_log, timing_log, lat_mag_log, param_mag_log)


def get_spec_with_default(specs, key, default):
    try:
        return specs[key]
    except KeyError:
        return default


def get_mean_latent_vector_magnitude(latent_vectors):
    return torch.mean(torch.norm(latent_vectors.weight.data.detach(), dim=1))


# >>> begin update: get mean magnitude from a list of lat vecs
def get_mean_latent_vector_magnitude_list(latent_vector_list):
    mag_list = [torch.mean(torch.norm(lv, dim=1)) for lv in latent_vector_list]
    return np.array(mag_list).mean()


# >>> end update


def append_parameter_magnitudes(param_mag_log, model):
    for name, param in model.named_parameters():
        if len(name) > 7 and name[:7] == "module.":
            name = name[7:]
        if name not in param_mag_log.keys():
            param_mag_log[name] = []
        param_mag_log[name].append(param.data.norm().item())


def main_function(experiment_directory, continue_from, batch_split):

    logging.debug("running " + experiment_directory)

    specs = ws.load_experiment_specifications(experiment_directory)

    logging.info("Experiment description: \n" + specs["Description"])

    data_source = specs["DataSource"]
    train_split_file = specs["TrainSplit"]

    arch = __import__("networks." + specs["NetworkArch"], fromlist=["Decoder"])

    logging.debug(specs["NetworkSpecs"])

    latent_size = specs["CodeLength"]

    checkpoints = list(
        range(
            specs["SnapshotFrequency"],
            specs["NumEpochs"] + 1,
            specs["SnapshotFrequency"],
        )
    )

    for checkpoint in specs["AdditionalSnapshots"]:
        checkpoints.append(checkpoint)
    checkpoints.sort()

    lr_schedules = get_learning_rate_schedules(specs)

    grad_clip = get_spec_with_default(specs, "GradientClipNorm", None)
    if grad_clip is not None:
        logging.debug("clipping gradients to max norm {}".format(grad_clip))

    def save_latest(epoch):

        save_model(experiment_directory, "latest.pth", decoder, epoch)
        save_optimizer(experiment_directory, "latest.pth", optimizer_all, epoch)
        save_latent_vectors(experiment_directory, "latest.pth", lat_vecs, epoch)

    def save_checkpoints(epoch):

        save_model(experiment_directory, str(epoch) + ".pth", decoder, epoch)
        save_optimizer(experiment_directory, str(epoch) + ".pth", optimizer_all, epoch)
        save_latent_vectors(experiment_directory, str(epoch) + ".pth", lat_vecs, epoch)

    def signal_handler(sig, frame):
        logging.info("Stopping early...")
        sys.exit(0)

    def adjust_learning_rate(lr_schedules, optimizer, epoch):

        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedules[i].get_learning_rate(epoch)

    def empirical_stat(latent_vecs, indices):
        lat_mat = torch.zeros(0).cuda()
        for ind in indices:
            lat_mat = torch.cat([lat_mat, latent_vecs[ind]], 0)
        mean = torch.mean(lat_mat, 0)
        var = torch.var(lat_mat, 0)
        return mean, var

    signal.signal(signal.SIGINT, signal_handler)

    num_samp_per_scene = specs["SamplesPerScene"]
    scene_per_batch = specs["ScenesPerBatch"]
    clamp_dist = specs["ClampingDistance"]
    minT = -clamp_dist
    maxT = clamp_dist
    enforce_minmax = True

    do_code_regularization = get_spec_with_default(specs, "CodeRegularization", True)
    code_reg_lambda = get_spec_with_default(specs, "CodeRegularizationLambda", 1e-4)

    code_bound = get_spec_with_default(specs, "CodeBound", None)

    # >>> begin update: added a few passable arguments
    assert specs["NetworkArch"] in [
        "decoder_z_leaky",
        "decoder_zr_leaky",
        "decoder_z_h_leaky",
    ], "wrong arch"
    eq_loss_active = get_spec_with_default(specs, "EqualityLoss", False)
    eq_loss_lambda = get_spec_with_default(specs, "EqualityLossLambda", 1.0)
    eq_loss_warmup = get_spec_with_default(specs, "EqualityLossWarmup", 100)
    reg_loss_warmup = get_spec_with_default(specs, "CodeRegularizationWarmup", 100)
    # >>> end update

    # >>> begin update: we need to pass a few more things to the network
    decoder = arch.Decoder(
        latent_size,
        num_dims=3,
        do_code_regularization=do_code_regularization,
        **specs["NetworkSpecs"],
        **specs["SubnetSpecs"]
    ).cuda()
    # >>> end update

    logging.info("training with {} GPU(s)".format(torch.cuda.device_count()))

    # if torch.cuda.device_count() > 1:
    decoder = torch.nn.DataParallel(decoder)

    num_epochs = specs["NumEpochs"]
    log_frequency = get_spec_with_default(specs, "LogFrequency", 10)

    # >>> begin update: using our dataloader
    # Get the data directory from environment variable
    train_split_file = train_split_file.replace("$DATADIR", os.environ["DATADIR"])

    # Create and load the dataset
    sdf_dataset = core.data.SamplesDataset(
        train_split_file,
        subsample=num_samp_per_scene,
        uniform_ratio=specs["UniformRatio"],
        use_occ=specs["UseOccupancy"],
        clamp_dist=clamp_dist,
    )
    # >>> end update

    num_data_loader_threads = get_spec_with_default(specs, "DataLoaderThreads", 1)
    logging.debug("loading data with {} threads".format(num_data_loader_threads))

    sdf_loader = data_utils.DataLoader(
        sdf_dataset,
        batch_size=scene_per_batch,
        shuffle=True,
        num_workers=num_data_loader_threads,
        drop_last=True,
    )

    logging.debug("torch num_threads: {}".format(torch.get_num_threads()))

    num_scenes = len(sdf_dataset)

    logging.info("There are {} scenes".format(num_scenes))

    logging.debug(decoder)

    lat_vecs = torch.nn.Embedding(num_scenes, latent_size, max_norm=code_bound)
    torch.nn.init.normal_(
        lat_vecs.weight.data,
        0.0,
        get_spec_with_default(specs, "CodeInitStdDev", 1.0) / math.sqrt(latent_size),
    )

    logging.debug(
        "initialized with mean magnitude {}".format(
            get_mean_latent_vector_magnitude(lat_vecs)
        )
    )

    # >>> begin update: using the bceloss
    loss_bce = torch.nn.BCEWithLogitsLoss(reduction="sum")
    loss_bce_no_logits = torch.nn.BCELoss(reduction="sum")
    ones = torch.ones((num_samp_per_scene * scene_per_batch, 1))
    ones.requires_grad = False
    ones = ones.cuda()
    # >>> end update

    optimizer_all = torch.optim.Adam(
        [
            {
                "params": decoder.parameters(),
                "lr": lr_schedules[0].get_learning_rate(0),
            },
            {
                "params": lat_vecs.parameters(),
                "lr": lr_schedules[1].get_learning_rate(0),
            },
        ]
    )

    loss_log = []
    lr_log = []
    lat_mag_log = []
    timing_log = []
    param_mag_log = {}

    start_epoch = 1

    # >>> begin update: added network backup
    backup_location = specs.setdefault("NetBackupLocation", None)
    if backup_location is not None:
        try:
            backup_location = backup_location.replace(
                "$MENDNETBACKUP", os.environ["MENDNETBACKUP"]
            )
        except KeyError:
            pass
        core.train.network_backup(experiment_directory, backup_location)
    # >>> end update

    # >>> begin update: neptune logging
    neptune_name = specs.setdefault("NeptuneName", None)
    stop_netptune_after = get_spec_with_default(specs, "StopNeptuneAfter", 200)
    if neptune_name is not None:
        logging.info("Logging to neptune project: {}".format(neptune_name))
        neptune.init(
            project_qualified_name=neptune_name,
            api_token=os.getenv("NEPTUNE_API_TOKEN"),
        )
        params = specs
        params.update(
            {
                "hostname": str(socket.gethostname()),
                "experiment_dir": os.path.basename(experiment_directory),
                "device count": str(int(torch.cuda.device_count())),
                "loader threads": str(int(num_data_loader_threads)),
                "torch threads": str(int(torch.get_num_threads())),
            }
        )
        neptune.create_experiment(params=params)
    # >>> end update

    if continue_from is not None:

        logging.info('continuing from "{}"'.format(continue_from))

        lat_epoch = load_latent_vectors(
            experiment_directory, continue_from + ".pth", lat_vecs
        )

        model_epoch = ws.load_model_parameters(
            experiment_directory, continue_from, decoder
        )

        optimizer_epoch = load_optimizer(
            experiment_directory, continue_from + ".pth", optimizer_all
        )

        loss_log, lr_log, timing_log, lat_mag_log, param_mag_log, log_epoch = load_logs(
            experiment_directory
        )

        if not log_epoch == model_epoch:
            loss_log, lr_log, timing_log, lat_mag_log, param_mag_log = clip_logs(
                loss_log, lr_log, timing_log, lat_mag_log, param_mag_log, model_epoch
            )

        if not (model_epoch == optimizer_epoch and model_epoch == lat_epoch):
            raise RuntimeError(
                "epoch mismatch: {} vs {} vs {} vs {}".format(
                    model_epoch, optimizer_epoch, lat_epoch, log_epoch
                )
            )

        start_epoch = model_epoch + 1

        logging.debug("loaded")

    logging.info("starting from epoch {}".format(start_epoch))

    logging.info(
        "Number of decoder parameters: {}".format(
            sum(p.data.nelement() for p in decoder.parameters())
        )
    )
    logging.info(
        "Number of shape code parameters: {} (# codes {}, code dim {})".format(
            lat_vecs.num_embeddings * lat_vecs.embedding_dim,
            lat_vecs.num_embeddings,
            lat_vecs.embedding_dim,
        )
    )

    # >>> being update: added tqdm indicator
    for epoch in tqdm.tqdm(
        range(start_epoch, num_epochs + 1), initial=start_epoch, total=num_epochs
    ):

        start = time.time()

        # logging.info("epoch {}...".format(epoch))

        if (stop_netptune_after is not False) and (epoch > stop_netptune_after):
            if neptune_name is not None:
                neptune.stop()
            neptune_name = None
        # >>> end update

        decoder.train()

        adjust_learning_rate(lr_schedules, optimizer_all, epoch)

        # >>> begin update: record a list of lat vecs
        g_lat_vec_list, h_lat_vec_list = [], []
        # >>> end update

        # >>> being update: data is in a slightly different format
        for data, indices in sdf_loader:
            # returns ((pts, cgt, bgt, rgt), (indices))

            for d in range(len(data)):
                data[d] = data[d].reshape(-1, data[d].shape[-1])
            num_sdf_samples = data[0].shape[0]

            # Disambiguate pts
            pts = data[0]
            pts.requires_grad = False

            # Disambiguate gt complete, broken, restoration
            gts = data[1:]
            for d in range(len(gts)):
                gts[d].requires_grad = False

            # Disambiguate indices
            indices = indices[0]

            # Chunk points
            xyz = pts.type(torch.float)
            xyz = torch.chunk(xyz, batch_split)

            # Chunk occ
            for d in range(len(gts)):
                gts[d] = gts[d].type(torch.float)
                gts[d] = torch.chunk(gts[d], batch_split)

            # Chunk indices
            indices = torch.chunk(
                indices.flatten(),
                batch_split,
            )
            # >>> end update

            batch_loss = 0.0

            optimizer_all.zero_grad()

            for i in range(batch_split):

                batch_vecs = lat_vecs(indices[i])

                input = torch.cat([batch_vecs, xyz[i]], dim=1)

                # >>> begin update: different outputs
                # NN optimization
                c_x, b_x, r_x, g_code, h_code = decoder(input.cuda())
                # >>> end update

                # >>> begin update: different loss
                c_gt, b_gt, r_gt = [g[i].cuda() for g in gts]
                chunk_loss = (
                    loss_bce(c_x, c_gt) + loss_bce(b_x, b_gt) + loss_bce(r_x, r_gt)
                ) / num_sdf_samples
                # >>> end update

                # >>> begin update: neptune logging
                if neptune_name is not None:
                    neptune.log_metric("data loss", chunk_loss.item())
                    neptune.log_metric(
                        "c accuracy",
                        accuracy_score(
                            torch.sigmoid(c_x).cpu().detach().round().numpy(),
                            gts[0][i].numpy(),
                        ),
                    )
                    neptune.log_metric(
                        "b accuracy",
                        accuracy_score(
                            torch.sigmoid(b_x).cpu().detach().round().numpy(),
                            gts[1][i].numpy(),
                        ),
                    )
                    neptune.log_metric(
                        "r accuracy",
                        accuracy_score(
                            torch.sigmoid(r_x).cpu().detach().round().numpy(),
                            gts[2][i].numpy(),
                        ),
                    )
                # >>> end update

                # >>> begin update: equality loss
                if eq_loss_active:
                    l1_equality_loss = (
                        eq_loss_lambda
                        * min(1, epoch / eq_loss_warmup)
                        * loss_bce(
                            (
                                (ones - torch.sigmoid(c_x))
                                * (ones - torch.sigmoid(b_x))
                                * (ones - torch.sigmoid(r_x))
                                + torch.sigmoid(c_x)
                                * (ones - torch.sigmoid(b_x))
                                * torch.sigmoid(r_x)
                                + torch.sigmoid(c_x)
                                * torch.sigmoid(b_x)
                                * (ones - torch.sigmoid(r_x))
                            ),
                            ones,
                        )
                    ) / num_sdf_samples
                    chunk_loss = chunk_loss + l1_equality_loss.cuda()
                    if neptune_name is not None:
                        neptune.log_metric("equality loss", l1_equality_loss.item())
                # >>> end update

                if do_code_regularization:
                    # >>> begin update: apply regularization to all vectors, renamed to l1_loss
                    l1_size_loss = torch.sum(torch.norm(batch_vecs, dim=1))
                    if g_code is not None:
                        l1_size_loss += torch.sum(torch.norm(g_code, dim=1)).cpu()
                    if h_code is not None:
                        l1_size_loss += torch.sum(torch.norm(h_code, dim=1)).cpu()
                    # >>> end update

                    # >>> begin update: added passable argument warmup, renamed to l1_loss
                    reg_loss = (
                        code_reg_lambda * min(1, epoch / reg_loss_warmup) * l1_size_loss
                    ) / num_sdf_samples
                    # >>> end update

                    # >>> begin update: neptune logging
                    if neptune_name is not None:
                        if g_code is not None:
                            g_lat_vec_list.append(g_code.cpu().detach())
                        if h_code is not None:
                            h_lat_vec_list.append(h_code.cpu().detach())
                        neptune.log_metric("reg loss", reg_loss.item())
                    # >>> end update

                    chunk_loss = chunk_loss + reg_loss.cuda()

                chunk_loss.backward()

                batch_loss += chunk_loss.item()

            logging.debug("loss = {}".format(batch_loss))

            loss_log.append(batch_loss)

            if grad_clip is not None:

                torch.nn.utils.clip_grad_norm_(decoder.parameters(), grad_clip)

            optimizer_all.step()

        end = time.time()

        seconds_elapsed = end - start
        timing_log.append(seconds_elapsed)

        lr_log.append([schedule.get_learning_rate(epoch) for schedule in lr_schedules])

        lat_mag_log.append(get_mean_latent_vector_magnitude(lat_vecs))

        append_parameter_magnitudes(param_mag_log, decoder)

        # >>> begin update: neptune logging
        if neptune_name is not None:
            neptune.log_metric("z mag", get_mean_latent_vector_magnitude(lat_vecs))
            neptune.log_metric(
                "g mag", get_mean_latent_vector_magnitude_list(g_lat_vec_list)
            )
            neptune.log_metric(
                "h mag", get_mean_latent_vector_magnitude_list(h_lat_vec_list)
            )
            neptune.log_metric("time", seconds_elapsed)
        # >>> end update

        if epoch in checkpoints:
            save_checkpoints(epoch)

            # >>> begin update: added network backup
            if backup_location is not None:
                core.train.network_backup(experiment_directory, backup_location)
            # >>> end update

        if epoch % log_frequency == 0:

            save_latest(epoch)
            save_logs(
                experiment_directory,
                loss_log,
                lr_log,
                timing_log,
                lat_mag_log,
                param_mag_log,
                epoch,
            )


if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser(description="Train a DeepSDF autodecoder")
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory. This directory should include "
        + "experiment specifications in 'specs.json', and logging will be "
        + "done in this directory as well.",
    )
    arg_parser.add_argument(
        "--continue",
        "-c",
        dest="continue_from",
        help="A snapshot to continue from. This can be 'latest' to continue"
        + "from the latest running snapshot, or an integer corresponding to "
        + "an epochal snapshot.",
    )
    arg_parser.add_argument(
        "--batch_split",
        dest="batch_split",
        default=1,
        help="This splits the batch into separate subbatches which are "
        + "processed separately, with gradients accumulated across all "
        + "subbatches. This allows for training with large effective batch "
        + "sizes in memory constrained environments.",
    )
    core.add_common_args(arg_parser)
    args = arg_parser.parse_args()
    core.configure_logging(args)

    main_function(args.experiment_directory, args.continue_from, int(args.batch_split))
