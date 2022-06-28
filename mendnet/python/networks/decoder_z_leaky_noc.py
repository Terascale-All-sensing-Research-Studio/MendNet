import torch
import torch.nn as nn
import torch.nn.functional as F


def update_args(specs, **kwargs):
    specs_args = specs["SubnetSpecs"]
    specs_args.update(kwargs)
    return specs_args


class Decoder(nn.Module):
    def __init__(
        self,
        latent_size,
        dims,
        num_dims=2,
        dropout=None,
        dropout_prob=0.0,
        norm_layers=(),
        latent_in=(),
        latent_in_inflate=False,
        weight_norm=False,
        xyz_in_all=False,
        use_tanh=False,
        latent_dropout=False,
        do_code_regularization=False,
        use_occ=True,
        subnet_dims=None,
        subnet_dropout=None,
        subnet_norm=None,
        subnet_xyz=False,
    ):
        super(Decoder, self).__init__()

        dims = [latent_size + num_dims] + dims + [1]
        self.num_dims = num_dims
        self.dropout = dropout
        self.dropout_prob = dropout_prob
        self.norm_layers = norm_layers
        self.latent_in = latent_in
        self.weight_norm = weight_norm
        self.pts_in_all = xyz_in_all
        self.use_tanh = use_tanh
        self.do_code_regularization = do_code_regularization
        self.latent_dropout = latent_dropout
        self.subnet_xyz = subnet_xyz

        self.num_layers = len(dims)
        subnet_pts_in = 0
        if subnet_xyz:
            subnet_pts_in = self.num_dims

        # Build gnet
        gnet_dims = [latent_size + subnet_pts_in] + subnet_dims + [latent_size]
        self.gnet_dropout = subnet_dropout
        self.gnet_norm_layers = subnet_norm
        self.gnet_num_layers = len(gnet_dims)
        for layer in range(0, self.gnet_num_layers - 1):
            out_dim = gnet_dims[layer + 1]

            if weight_norm and (layer in self.gnet_norm_layers):
                setattr(
                    self,
                    "gnet_lin" + str(layer),
                    nn.utils.weight_norm(nn.Linear(gnet_dims[layer], out_dim)),
                )
            else:
                setattr(
                    self, "gnet_lin" + str(layer), nn.Linear(gnet_dims[layer], out_dim)
                )

            if (
                (not weight_norm)
                and (self.gnet_norm_layers is not None)
                and (layer in self.gnet_norm_layers)
            ):
                setattr(self, "gnet_bn" + str(layer), nn.LayerNorm(out_dim))

        # Build fnet
        for layer in range(0, self.num_layers - 1):
            if latent_in_inflate:
                out_dim = dims[layer + 1]
                if layer in latent_in:
                    in_dim = dims[layer] + dims[0]
                else:
                    in_dim = dims[layer]
            else:
                in_dim = dims[layer]
                if layer + 1 in latent_in:
                    out_dim = dims[layer + 1] - dims[0]
                else:
                    out_dim = dims[layer + 1]
            if (self.pts_in_all) and (layer != self.num_layers - 2):
                out_dim -= self.num_dims

            if weight_norm and (layer in norm_layers):
                setattr(
                    self,
                    "lin" + str(layer),
                    nn.utils.weight_norm(nn.Linear(in_dim, out_dim)),
                )
            else:
                setattr(self, "lin" + str(layer), nn.Linear(in_dim, out_dim))

            if (
                (not weight_norm)
                and (norm_layers is not None)
                and (layer in norm_layers)
            ):
                setattr(self, "bn" + str(layer), nn.LayerNorm(out_dim))

        if use_tanh:
            self.tanh = nn.Tanh()
        self.relu = nn.LeakyReLU()

        if not use_occ:
            self.th = nn.Tanh()

        setattr(
            self,
            "secondary_net_parameters",
            [p for n, p in self.named_parameters() if ("gnet" in n) or ("hnet" in n)],
        )
        setattr(
            self,
            "primary_net_parameters",
            [
                p
                for n, p in self.named_parameters()
                if ("gnet" not in n) and ("hnet" not in n)
            ],
        )

    def disable_f_grad(self):
        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(layer))
            lin.requires_grad = False

            if layer < self.num_layers - 2:
                if (
                    (self.norm_layers is not None)
                    and (layer in self.norm_layers)
                    and (not self.weight_norm)
                ):
                    bn = getattr(self, "bn" + str(layer))
                    bn.requires_grad = False

    def forward(self, net_input, use_net=None):
        # Input is N x (|z|+num_dims)
        pts = net_input[:, -self.num_dims :]

        # Apply dropout to the latent vector directly
        if net_input.shape[1] > 2 and self.latent_dropout:
            latent_vecs = net_input[:, : -self.num_dims]
            latent_vecs = F.dropout(latent_vecs, p=0.2, training=self.training)
            net_input = torch.cat([latent_vecs, pts], 1)
        lat_vecs_input = net_input[:, : -self.num_dims]
        if self.subnet_xyz:
            subet_input = net_input
        else:
            subet_input = lat_vecs_input

        def ff_fnet(x):
            # Store the latent vector for skip connections
            x_input = x

            # Feed forward Bnet
            for layer in range(0, self.num_layers - 1):
                # Latent vector appended to input
                if layer in self.latent_in:
                    x = torch.cat([x, x_input], 1)

                # pts appended to input
                elif layer != 0 and self.pts_in_all:
                    x = torch.cat([x, pts], 1)

                # Feed forward
                lin = getattr(self, "lin" + str(layer))
                x = lin(x)

                # Apply tanh layer (second to last)
                if (layer == self.num_layers - 2) and self.use_tanh:
                    x = self.tanh(x)

                if layer < self.num_layers - 2:
                    # Apply weight normalization
                    if (
                        (self.norm_layers is not None)
                        and (layer in self.norm_layers)
                        and (not self.weight_norm)
                    ):
                        bn = getattr(self, "bn" + str(layer))
                        x = bn(x)

                    # Apply relu
                    x = self.relu(x)

                    # Apply dropout
                    if (self.dropout is not None) and (layer in self.dropout):
                        x = F.dropout(x, p=self.dropout_prob, training=self.training)

            # Apply final layer
            if hasattr(self, "th"):
                x = self.th(x)

            return x

        def ff_gnet(x):
            # Feed forward gnet
            for layer in range(0, self.gnet_num_layers - 1):
                # Feed forward
                lin = getattr(self, "gnet_lin" + str(layer))
                x = lin(x)

                if layer < self.gnet_num_layers - 1:
                    # Apply weight normalization
                    if (
                        (self.gnet_norm_layers is not None)
                        and (layer in self.gnet_norm_layers)
                        and (not self.weight_norm)
                    ):
                        bn = getattr(self, "gnet_bn" + str(layer))
                        x = bn(x)

                    # Apply relu
                    x = self.relu(x)

                    # Apply dropout
                    if (self.gnet_dropout is not None) and (layer in self.gnet_dropout):
                        x = F.dropout(x, p=self.dropout_prob, training=self.training)

            return x

        if use_net is None:
            # Feed forward gnet
            x = ff_gnet(subet_input)
            r_code = x
            r_x = ff_fnet(torch.cat([x, pts], 1))

            # Finally, feed forward Bnet
            b_x = ff_fnet(net_input)

            if self.do_code_regularization:
                return (None, b_x, r_x, r_code, None)
            return (None, b_x, r_x)
        elif use_net == 1:
            return ff_fnet(net_input)
        elif use_net == 2:
            r_code = ff_gnet(subet_input)
            return ff_fnet(torch.cat([r_code, pts], 1))
        else:
            raise RuntimeError(
                "Requested output from non-existent network: {}".format(use_net)
            )
