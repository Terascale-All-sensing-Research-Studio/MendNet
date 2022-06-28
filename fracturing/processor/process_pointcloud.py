import os
import logging

import trimesh
import numpy as np


def process_pointnet(
    f_in, 
    fc_in,
    f_out,
    sample=8192,
    overwrite=False,
):  

    mesh_b = trimesh.load(f_in)
    mesh_c = trimesh.load(fc_in)

    vb, inds = mesh_b.sample(count=sample, return_index=True)
    nb = mesh_b.face_normals[inds, :]
    vc, inds = mesh_c.sample(count=sample, return_index=True)
    nc = mesh_c.face_normals[inds, :]

    if overwrite or not os.path.exists(f_out):
        logging.debug("Saving to: {}".format(f_out))
        np.save(
            f_out,
            [vb, vc, nb, nc]
        )
