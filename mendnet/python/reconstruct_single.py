import argparse
import json
import logging
import os
import random
import math
import multiprocessing

import trimesh
import torch
import numpy as np

import core

STATUS_INDICATOR = None


def reconstruct(
    decoder,
    num_iterations,
    latent_size,
    test_sdf,
    stat,
    clamp_dist,
    num_samples=30000,
    lr=5e-4,
    l2reg=False,
    code_reg_lambda=1e-4,
    lambda_eq=0.0,
    lambda_sup=0.0,
    eq_warmup=1,
    sup_cooldown=0,
    iter_path=None,
    ncol_method=None,
):

    def adjust_learning_rate(
        initial_lr, optimizer, num_iterations, decreased_by, adjust_lr_every
    ):
        lr = initial_lr * ((1 / decreased_by) ** (num_iterations // adjust_lr_every))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    decreased_by = 10
    adjust_lr_every = int(num_iterations / 2)

    if type(stat) == type(0.1):
        latent = torch.ones(1, latent_size).normal_(mean=0, std=stat).cuda()
    else:
        latent = torch.normal(stat[0].detach(), stat[1].detach()).cuda()

    latent.requires_grad = True

    optimizer = torch.optim.Adam([latent], lr=lr)

    loss_bce_with_logits = torch.nn.BCEWithLogitsLoss()
    loss_bce = torch.nn.BCEWithLogitsLoss()
    ones = torch.ones((num_samples, 1))
    ones.requires_grad = False
    ones = ones.cuda()
    loss_dict = {
        "epoch": [],
        "data_loss": [],
        "eq_loss": [],
        "sup_loss": [],
        "reg_loss": [],
        "mag": [],
        "g_mag": [],
        "h_mag": [],
    }

    for e in range(num_iterations):

        decoder.eval()

        pts, data_sdf = test_sdf
        pts, sdf_gt = core.data.select_samples(
            pts, data_sdf, num_samples, uniform_ratio=0.2
        )

        # Visualize
        # core.vis.plot_samples((pts, sdf_gt), n_plots=16).savefig("test.png")

        # Convert to tensors
        xyz = torch.from_numpy(pts).type(torch.float).cuda()
        sdf_gt = torch.from_numpy(sdf_gt).type(torch.float).cuda()


        adjust_learning_rate(lr, optimizer, e, decreased_by, adjust_lr_every)

        optimizer.zero_grad()

        latent_inputs = latent.expand(num_samples, -1)

        inputs = torch.cat([latent_inputs, xyz], 1).cuda()

        c_x, b_x, r_x, g_code, h_code = decoder(inputs)

        # TODO: why is this needed?
        if e == 0:
            c_x, b_x, r_x, g_code, h_code = decoder(inputs)


        data_loss = loss_bce_with_logits(b_x, sdf_gt)
        loss = data_loss

        # Equality loss
        if lambda_eq != 0.0:
            eq_loss = (
                lambda_eq
                * min(1, e / eq_warmup)
                * (
                    loss_bce(
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
                )
            )
            loss = loss + eq_loss

        if lambda_sup != 0.0:
            if ncol_method is None:
                sup_loss = (
                    lambda_sup
                    * max(0, (sup_cooldown - e) / sup_cooldown)
                    * (
                        loss_bce(
                            torch.sigmoid(r_x),
                            ones,
                        )
                    )
                )
            elif ncol_method == "r=c":
                sup_loss = (
                    lambda_sup
                    * max(0, (sup_cooldown - e) / sup_cooldown)
                    * (
                        loss_bce(
                            torch.sigmoid(r_x),
                            torch.sigmoid(c_x),
                        )
                    )
                )
            elif ncol_method == "batch":
                if (torch.sigmoid(r_x).sum() / num_samples) < 0.1:
                    sup_loss = (
                        lambda_sup
                        * max(0, (sup_cooldown - e) / sup_cooldown)
                        * (
                            loss_bce(
                                torch.sigmoid(r_x),
                                torch.sigmoid(c_x),
                            )
                        )
                    )
            else:
                raise RuntimeError("Unknown ncol method: {}".format(ncol_method))
            loss = loss + sup_loss

        # Regularization loss
        if l2reg:
            reg_loss = torch.mean(latent.pow(2))
            if g_code is not None:
                reg_loss += torch.mean(g_code.pow(2))
            if h_code is not None:
                reg_loss += torch.mean(h_code.pow(2))
            reg_loss = reg_loss * code_reg_lambda
            loss = loss + reg_loss

        if e % 10 == 0:
            loss_dict["epoch"].append(e)
            loss_dict["data_loss"].append(data_loss.item())
            if "eq_loss" in locals():
                loss_dict["eq_loss"].append(eq_loss.item())
            else:
                loss_dict["eq_loss"].append(0)
            if "sup_loss" in locals():
                loss_dict["sup_loss"].append(sup_loss.item())
            else:
                loss_dict["sup_loss"].append(0)
            loss_dict["reg_loss"].append(reg_loss.item())
            loss_dict["mag"].append(torch.norm(latent).item())
            if g_code is not None:
                loss_dict["g_mag"].append(torch.norm(g_code).item())
            if h_code is not None:
                loss_dict["h_mag"].append(torch.norm(h_code).item())
            logging.debug(
                "epoch: {:4d} | data_loss: {} eq_loss: {} sup_loss: {} reg_loss: {}".format(
                    loss_dict["epoch"][-1],
                    loss_dict["data_loss"][-1],
                    loss_dict["eq_loss"][-1],
                    loss_dict["sup_loss"][-1],
                    loss_dict["reg_loss"][-1],
                )
            )

        loss.backward()
        optimizer.step()

    return loss_dict, latent.cpu()


def callback():
    global STATUS_INDICATOR
    STATUS_INDICATOR.increment()


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(
        description="Use a trained DeepSDF decoder to reconstruct a shape given SDF "
        + "samples."
    )
    arg_parser.add_argument(
        "--input_mesh",
        required=True,
        help="gt fractured object, used only for rendering.",
    )
    arg_parser.add_argument(
        "--input_points",
        required=True,
        help="Sample points, specified as a .npz file.",
    )
    arg_parser.add_argument(
        "--input_sdf",
        required=True,
        help="Sample sdf, specified as a .npz file.",
    )
    arg_parser.add_argument(
        "--output_meshes",
        required=True,
        help="Path template to save the meshes to.",
    )
    arg_parser.add_argument(
        "--output_code",
        required=True,
        help="Path template to save the code to.",
    )
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory which includes specifications and saved model "
        + "files to use for reconstruction",
    )
    arg_parser.add_argument(
        "--checkpoint",
        "-c",
        dest="checkpoint",
        default="latest",
        help="The checkpoint weights to use. This can be a number indicated an epoch "
        + "or 'latest' for the latest weights (this is the default)",
    )
    arg_parser.add_argument(
        "--threads",
        default=6,
        type=int,
        help="Number of threads to use for reconstruction.",
    )
    arg_parser.add_argument(
        "--render_threads",
        default=6,
        type=int,
        help="Number of threads to use for rendering.",
    )
    arg_parser.add_argument(
        "--num_iters",
        default=800,
        type=int,
        help="The number of iterations of latent code optimization to perform.",
    )
    arg_parser.add_argument(
        "--num_samples",
        default=8000,
        type=int,
        help="Number of samples to use.",
    )
    arg_parser.add_argument(
        "--eq_warmup",
        default=1,
        type=float,
        help="Equality lambda value.",
    )
    arg_parser.add_argument(
        "--sup_cooldown",
        default=800,
        type=float,
        help="Equality lambda value.",
    )
    arg_parser.add_argument(
        "--lambda_eq",
        default=0.0001,
        type=float,
        help="Equality lambda value.",
    )
    arg_parser.add_argument(
        "--lambda_sup",
        default=0.0001,
        type=float,
        help="Equality lambda value.",
    )
    arg_parser.add_argument(
        "--lambda_reg",
        default=1e-4,
        type=float,
        help="Regularization lambda value.",
    )
    arg_parser.add_argument(
        "--learning_rate",
        default=5e-3,
        type=float,
        help="Regularization lambda value.",
    )
    arg_parser.add_argument(
        "--uniform_ratio",
        default=None,
        type=float,
        help="Uniform Ratio.",
    )
    arg_parser.add_argument(
        "--ncol_method",
        default=None,
        type=str,
        help="Method by which to apply noncollapse constraint.",
    )
    arg_parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite code.",
    )
    arg_parser.add_argument(
        "--gif",
        action="store_true",
        default=False,
        help="",
    )
    core.add_common_args(arg_parser)
    args = arg_parser.parse_args()
    core.configure_logging(args)

    specs_filename = core.find_specs(args.experiment_directory)
    specs = json.load(open(specs_filename))
    args.experiment_directory = os.path.dirname(specs_filename)

    arch = __import__("networks." + specs["NetworkArch"], fromlist=["Decoder"])

    latent_size = specs["CodeLength"]

    num_samples = args.num_samples
    num_iterations = args.num_iters
    lambda_eq = args.lambda_eq
    lambda_sup = args.lambda_sup
    code_reg_lambda = args.lambda_reg
    lr = args.learning_rate
    eq_warmup = args.eq_warmup
    sup_cooldown = args.sup_cooldown
    uniform_ratio = args.uniform_ratio
    ncol_method = args.ncol_method
    if uniform_ratio is None:
        uniform_ratio = specs["UniformRatio"]

    clamp_dist = specs["ClampingDistance"]

    network_outputs = (0, 1, 2)
    do_code_regularization = True
    isosurface_level = 0.5
    use_sigmoid = True

    assert specs["NetworkArch"] in [
        "decoder_z_leaky",
        "decoder_zr_leaky",
        "decoder_z_h_leaky",
    ], "wrong arch"
    # assert lambda_eq != 0.0, "Must use equality loss"

    network_kwargs = dict(
        decoder_kwargs=dict(
            latent_size=latent_size,
            num_dims=3,
            do_code_regularization=do_code_regularization,
            **specs["NetworkSpecs"],
            **specs["SubnetSpecs"],
        ),
        decoder_constructor=arch.Decoder,
        experiment_directory=args.experiment_directory,
        checkpoint=args.checkpoint,
    )
    reconstruction_kwargs = dict(
        num_iterations=num_iterations,
        latent_size=latent_size,
        stat=0.01,  # [emp_mean,emp_var],
        clamp_dist=clamp_dist,
        num_samples=num_samples,
        lr=lr,
        l2reg=do_code_regularization,
        code_reg_lambda=code_reg_lambda,
        lambda_eq=lambda_eq,
        lambda_sup=lambda_sup,
        eq_warmup=eq_warmup,
        sup_cooldown=sup_cooldown,
        ncol_method=ncol_method,
    )
    mesh_kwargs = dict(
        dims=[256, 256, 256],
        sigmoid=use_sigmoid,
        level=isosurface_level,
        gradient_direction="descent",
        batch_size=2 ** 14,
    )

    assert os.path.splitext(args.output_code)[-1] == ".pth"
    assert os.path.splitext(args.output_meshes)[-1] == ".obj"

    # Load the data
    xyz = np.load(args.input_points)["xyz"]
    sdf = np.load(args.input_sdf)["sdf"]
    sdf = core.sdf_to_occ(np.expand_dims(sdf, axis=1))
    assert len(xyz.shape) == 2 and len(sdf.shape) == 2

    # Load the network
    decoder = core.load_network(**network_kwargs)
    decoder.eval()

    # Reconstruct the code
    if not os.path.exists(args.output_code) or args.overwrite:
        losses, code = reconstruct(
            test_sdf=[xyz, sdf],
            decoder=decoder,
            **reconstruction_kwargs,
        )
        core.saver(args.output_code, code)
    else:
        code = core.loader(args.output_code)

    mesh_path_list = [
        os.path.splitext(args.output_meshes)[0]
        + str(shape_idx)
        + os.path.splitext(args.output_meshes)[-1]
        for shape_idx in range(3)
    ]

    # Reconstruct the meshes
    mesh_list = []
    for shape_idx, path in enumerate(mesh_path_list):
        if not os.path.exists(path) or args.overwrite:
            try:
                mesh = core.reconstruct.create_mesh(
                    vec=code,
                    decoder=decoder,
                    use_net=shape_idx,
                    **mesh_kwargs,
                )
                mesh.export(path)
            except core.errors.IsosurfaceExtractionError:
                logging.info(
                    "Isosurface extraction error for shape: {}".format(shape_idx)
                )
                mesh = None
        else:
            mesh = core.loader(path)
        mesh_list.append(mesh)

    # Create a render of the the restoration object with gt fractured mesh

    DURATION = 10  # in seconds
    FRAME_RATE = 30
    RESOLUTION = (600, 600)
    ZOOM = 2.0
    num_renders = DURATION * FRAME_RATE

    if mesh_list[2] is not None:
        gt_mesh = core.loader(args.input_mesh)
        gt_mesh.fix_normals()
        if args.gif:
            core.saver(
                f_out=os.path.splitext(args.output_meshes)[0] + "_f_r.gif",
                data=core.create_gif_rot(
                    [
                        core.colorize_mesh_from_index(gt_mesh, 1),
                        core.colorize_mesh_from_index(mesh_list[2], 2),
                    ],
                    num_renders=num_renders,
                    resolution=RESOLUTION,
                    zoom=ZOOM,
                    bg_color=0,
                ),
                loop=0,
                duration=(1 / num_renders) * DURATION * 1000,
            )
        else:
            core.saver(
                f_out=os.path.splitext(args.output_meshes)[0] + "_f_r.png",
                data=core.render_mesh(
                    [
                        core.colorize_mesh_from_index(gt_mesh, 1),
                        core.colorize_mesh_from_index(mesh_list[2], 2),
                    ],
                    resolution=RESOLUTION,
                    ztrans=ZOOM,
                    bg_color=0,
                )
            )
