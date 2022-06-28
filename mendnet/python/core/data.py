import os
import logging
import pickle

import torch
import torch.utils.data
import trimesh
import numpy as np
from scipy.spatial import cKDTree as KDTree

import core
import core.errors as errors


def select_samples(pts, values, num_samples=None, uniform_ratio=0.5):
    """
    Return a point value pair such that each shape will have at least
    num_samples / (len(shapes) * 2) interior and exterior points.

    Args:
        pts: Points in n dimensional space.
        values: Value at each point. The number of columns gives the number
            of shapes.
        num_samples: Number of samples to return.
        uniform_ratio: Ratio between uniform and surface sampled points.
            Set to 0.5 to disable.
    """
    if uniform_ratio is not None:
        pts, values = select_uniform_ratio_samples(pts, values, uniform_ratio)

    num_shapes = values.shape[1]
    if num_samples is None:
        num_samples = values.shape[0]

    # We want to sample each object about equally
    if num_shapes == 1:
        num_samples = [num_samples]
    else:
        num_samples = [(num_samples // num_shapes) for _ in range(num_shapes - 1)] + [
            num_samples - ((num_shapes - 1) * (num_samples // num_shapes))
        ]

    # Pick the samples fairly
    idx_accumulator = []
    for d, s in zip(values.T, num_samples):
        d = np.expand_dims(d, axis=1)
        pos_inds = np.where(d[:, 0] > 0)[0]
        neg_inds = np.where(d[:, 0] <= 0)[0]
        pos_num, neg_num = (s // 2), (s - (s // 2))

        # Pick negative samples
        if len(neg_inds) > neg_num:
            start_ind = np.random.randint(len(neg_inds) - neg_num)
            neg_picked = neg_inds[start_ind : start_ind + neg_num]
        else:
            try:
                neg_picked = np.random.choice(neg_inds, neg_num)
            except ValueError:
                raise errors.NoSamplesError

        # Pick positive samples
        if len(pos_inds) > pos_num:
            start_ind = np.random.randint(len(pos_inds) - pos_num)
            pos_picked = pos_inds[start_ind : start_ind + pos_num]
        else:
            try:
                pos_picked = np.random.choice(pos_inds, pos_num)
            except ValueError:
                raise errors.NoSamplesError
        idx_accumulator.extend([pos_picked, neg_picked])
    idx_accumulator = np.concatenate(idx_accumulator)

    return pts[idx_accumulator, :], values[idx_accumulator, :]


def select_uniform_ratio_samples(pts, values, uniform_ratio=0.5, randomize=True):
    """
    Randomly pick samples with a specific uniform ratio.

    Args:
        pts: N-dimensional sample points.
        values: Sdf or occupancy values, same size as pts.
        uniform_ratio: Ratio between uniform and surface sampled points.
            Set to 0.5 to disable.
    """

    # Points will come in split half between uniform and surface
    def adjust_uniform_ratio(data, n_pts, bad_surface=0, bad_uniform=0):
        max_can_select = int(n_pts / 2) - max(bad_surface, bad_uniform)
        surface_ends_at = int(n_pts / 2) - bad_surface

        # We can balance the number of uniform and surface points here
        if uniform_ratio > 0.5:
            select_n_pts = int((max_can_select * (1 - uniform_ratio)) / uniform_ratio)
            selected_pts = np.random.choice(
                max_can_select, size=(select_n_pts), replace=False
            )
            data = np.vstack(
                (
                    data[selected_pts, :],
                    data[surface_ends_at : surface_ends_at + max_can_select, :],
                )
            )
        elif uniform_ratio < 0.5:
            select_n_pts = int((max_can_select * uniform_ratio) / (1 - uniform_ratio))
            selected_pts = (
                np.random.choice(max_can_select, size=(select_n_pts), replace=False)
                + surface_ends_at
            )
            data = np.vstack((data[:max_can_select, :], data[selected_pts, :]))
        else:
            data = np.vstack(
                (
                    data[:max_can_select, :],
                    data[surface_ends_at : surface_ends_at + max_can_select, :],
                )
            )
        return data

    num_dims = pts.shape[1]
    data = np.hstack((pts, values))

    # Adjust the ratio of points if necessary
    if uniform_ratio != 0.5:
        data = adjust_uniform_ratio(data, data.shape[0])

    # Shuffle
    if randomize:
        data = data[np.random.permutation(data.shape[0]), :]
    return data[:, :num_dims], data[:, num_dims:]


def sdf_to_occ_grid_threshold(data, thresh=0, flip=False):
    """
    Given a grid of sdf values, convert sdf values to occupancy values with a threshold.
    """
    data = data.copy()
    mask = data >= thresh
    if flip:
        data[mask] = 1
        data[~mask] = 0
    else:
        data[mask] = 0
        data[~mask] = 1
    return data.astype(int)


def sdf_to_occ(data, skip_cols=0):
    """
    Given a sample, convert sdf values to occupancy values.
    """
    if data.shape[1] == 0:
        data[data >= 0] = 0.0
        data[data < 0] = 1.0
    else:
        data[:, skip_cols:][data[:, skip_cols:] >= 0] = 0.0
        data[:, skip_cols:][data[:, skip_cols:] < 0] = 1.0
    return data


def clamp_samples(data, clamp_dist, skip_cols=0):
    """
    Given a sample, clamp that sample to +/- clamp_dist.
    """
    data[:, skip_cols:] = np.clip(data[:, skip_cols:], -clamp_dist, clamp_dist)
    return data


def get_uniform(data):
    """
    Given a sample, return uniform points.
    """
    return data[int(data.shape[0] / 2) :, :]


def get_surface(data):
    """
    Given a sample, return surface points.
    """
    return data[: int(data.shape[0] / 2), :]


def partial_sample_with_random_noise(complete_gt, broken_gt, pts, mask, percent=0.15):
    """
    Creates a partial sample mask by randomly selecting points in the fracture region to flip.
    """

    if percent == 1.0:
        mask[:] = True
        return mask
    elif percent == 0.0:
        return mask

    # Get the fracture verts (this will work because B comes from C)
    d, _ = KDTree(complete_gt.vertices).query(broken_gt.vertices)
    fracture_vert_mask = d > 0.001

    # How many are we flipping
    num_to_flip = int(fracture_vert_mask.sum() * percent)

    # Pick this many from the mask
    idxs_to_flip = np.random.choice(
        np.where(fracture_vert_mask)[0], num_to_flip, replace=False
    )
    fracture_vert_mask[idxs_to_flip] = False

    # Find the closest point on the broken object for all the query points
    _, ind = KDTree(broken_gt.vertices).query(pts)

    no_sample_zone_mask = np.logical_not(mask)
    eroded_no_sample_zone_mask = np.logical_and(
        no_sample_zone_mask, fracture_vert_mask[ind]
    )
    return np.logical_not(eroded_no_sample_zone_mask)


def partial_sample_with_classification_noise(
    complete_gt, broken_gt, pts, mask, percent=0.15
):
    """
    Creates a partial sample mask by adding to the boundary of the fracture region.
    """

    if percent == 1.0:
        mask[:] = True
        return mask
    elif percent == 0.0:
        return mask

    # Get the fracture verts (this will work because B comes from C)
    d, _ = KDTree(complete_gt.vertices).query(broken_gt.vertices)
    fracture_vertices_index = set(np.where(d > 0.001)[0])
    exterior_vertices_index = set(np.where(d < 0.001)[0])
    start_size = len(fracture_vertices_index)

    while True:
        # Find adjacencies of the exterior surface
        accumulator = set()
        for v1, v2 in broken_gt.edges:
            if v1 in exterior_vertices_index:
                accumulator.add(v2)
            elif v2 in exterior_vertices_index:
                accumulator.add(v1)

        # Subtract them
        fracture_vertices_index = fracture_vertices_index.difference(accumulator)
        exterior_vertices_index = exterior_vertices_index.union(accumulator)

        if len(fracture_vertices_index) < start_size * (1 - percent):
            break

    # Add some vertices back
    add_back = int(start_size * (1 - percent)) - len(fracture_vertices_index)
    fracture_vertices_index = fracture_vertices_index.union(
        set(list(accumulator)[: min(len(accumulator), add_back)])
    )

    # Update the mask
    fracture_vert_mask = np.zeros((broken_gt.vertices.shape[0])).astype(bool)
    fracture_vert_mask[np.array(list(fracture_vertices_index))] = True

    # Find the closest point on the broken object for all the query points
    _, ind = KDTree(broken_gt.vertices).query(pts)

    no_sample_zone_mask = np.logical_not(mask)
    eroded_no_sample_zone_mask = np.logical_and(
        no_sample_zone_mask, fracture_vert_mask[ind]
    )
    return np.logical_not(eroded_no_sample_zone_mask)


def compute_roughness_mask(mesh, threshold=0.001, lamb=0.5, iterations=10):
    """
    Compute a vertex mask that identifies rough vertices on the surface of the mesh.
    """
    # Perform laplacian smoothing
    smoothed_mesh = mesh.copy()
    trimesh.smoothing.filter_laplacian(
        smoothed_mesh, 
        lamb=lamb, 
        iterations=iterations
    )

    # Get indices where roughness is above the threshold
    roughness = np.linalg.norm(mesh.vertices - smoothed_mesh.vertices, axis=1)
    return roughness > threshold
    

def partial_sample_analytical(
    complete_gt, broken_gt, pts, mask, percent=0.15, threshold=0.01, lamb=0.5, iterations=10,
):
    """
    Creates a partial sample mask by using an analytical roughness measure.
    """

    assert percent == 0.0
    fracture_vert_mask = compute_roughness_mask(
        broken_gt, 
        threshold=threshold, 
        lamb=lamb, 
        iterations=iterations
    )
    not_fracture_vert_mask = np.logical_not(fracture_vert_mask)

    # Find the closest point on the broken object for all the query points, 
    # and return the mask at that value
    _, ind = KDTree(broken_gt.vertices).query(pts)
    return not_fracture_vert_mask[ind]


def select_partial_samples(pts, values, mask, num_samples=None, uniform_ratio=0.5):
    """
    Return a point value pair such that each shape will have at least
    num_samples / (len(shapes) * 2) interior and exterior points. Applies a
    mask first tho.

    Args:
        shapes: List of shapes [c, b, r].
        mask: Mask corresponding to those points to not sample from.
        num_samples: Number of samples to return.
        uniform_ratio: Ratio between uniform and surface sampled points.
            Set to 0.5 to disable.
    """

    # Add an indexing row to pts
    pts = np.hstack((np.expand_dims(np.arange(pts.shape[0]), axis=1), pts))

    # Get a random sampling of the data with the correct uniform ratio
    pts, values = select_uniform_ratio_samples(
        pts, values, uniform_ratio, randomize=False
    )

    # Get non-masked points
    _, intersection_inds, _ = np.intersect1d(
        pts[:, 0], np.where(mask)[0], return_indices=True
    )

    # Remove the indexing row
    pts = pts[:, 1:]

    # Discard non masked points
    pts = pts[intersection_inds, :]
    values = values[intersection_inds, :]

    # Now we need to re-randomize the order
    num_dims = pts.shape[1]
    data = np.hstack((pts, values))
    data = data[np.random.permutation(data.shape[0])]
    pts, values = data[:, :num_dims], data[:, num_dims:]

    # Return fairly selected samples
    return select_samples(pts, values, num_samples, uniform_ratio=None)


def quick_load_path(path):
    """Converts a normal path to a quickload path"""
    return os.path.splitext(path)[0] + "_quickload" + os.path.splitext(path)[1]


class SamplesDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        split,  # The dictionary object holding sample or shape data
        use_occ=True,  # Use occupancy samples
        subsample=None,  # How many samples to use during training
        learned_breaks=False,
        uniform_ratio=0.5,  # Ratio of uniform to surface samples
        root=None,  # Location of the top level data directory on disk
        validate_occ=True,  # Will perform validation on the occupancy values
        clamp_dist=None,  # If using sdf, clamp the values
        load_values=True,
    ):
        self._subsample = subsample
        self._use_occ = use_occ
        self._learned_breaks = learned_breaks
        self._uniform_ratio = uniform_ratio
        self._clamp_dist = clamp_dist
        self._num_dims = 3  # This is always constant for the time being
        self._num_components = None

        # Must pass a clamp distance if using sdf samples
        if not self._use_occ:
            assert (
                self._clamp_dist is not None
            ), "Must provide clamping distance for sdf values"
            logging.info("Using SDF samples")
        else:
            logging.info("Using OCC samples")

        logging.info("Using uniform ratio:  {}".format(self._uniform_ratio))
        logging.info("Using clamp distance: {}".format(self._clamp_dist))
        logging.info("Using point sampling: {}".format(self._subsample))

        # Data class supports passing a dictionary containing the data directly,
        # if desired. Note that if not passed a dictionary, can pass a string
        # containing DATADIR that will be autoreplaced.
        if not isinstance(split, dict):
            split = split.replace("$DATADIR", os.environ["DATADIR"])
            logging.info("Loading data from: {}".format(split))
            self._raw_data = pickle.load(open(split, "rb"))
        else:
            self._raw_data = split

        # The input data could be in OCC mode to save on disk space. If this is
        # the case, then we cannot convert back to sdf.
        input_data_in_occ = False
        if ("use_occ" in self._raw_data) and self._raw_data["use_occ"]:
            assert use_occ, "Cannot generate sdf samples"
            input_data_in_occ = True

        # Pick apart the input data
        train_mode = self._raw_data.get("train", False)
        self._indexer = self._raw_data["indexer"]
        self._complete_indexer = self._raw_data["complete_indexer"]
        self._objects = self._raw_data["objects"]

        # Record the number of shapes and the number of individual instances
        self._num_shapes = len(set([o[0] for o in self._indexer]))
        self._num_instances = len(self._indexer)
        logging.info(
            "Loaded {} shapes, {} instances".format(self.num_shapes, self.num_instances)
        )

        # Optionally update the stored root location. Note the double check so
        # that root is not set to the empty string. This could be considered
        # a bug, but I don't think it'll every be used.
        if (root is not None) and root:
            for idx in range(len(self._objects)):
                self._objects[idx]._root_dir = root

        # The data can also be loaded without sdf values. This dramatically
        # reduces disk size.
        try:
            self._data = self._raw_data["sdf"]
        except KeyError:
            return
        if not load_values:
            return
        
        if isinstance(self._data, str):
            logging.info("Loading raw data from: {}".format(self._data))
            self._data = np.load(self._data.replace("$DATADIR", os.environ["DATADIR"]))["sdf"]

        if self._use_occ:
            logging.debug("Converting sdf samples to occupancy samples ...")

            # Data is in test mode
            if isinstance(self._data[0], dict):
                if input_data_in_occ:
                    for idx in range(len(self._data)):
                        self._data[idx]["sdf"] = self._data[idx]["sdf"].astype(float)

                else:
                    for idx in range(len(self._data)):
                        self._data[idx]["sdf"] = sdf_to_occ(
                            self._data[idx]["sdf"].astype(float)
                        )

                # If this is a training split, data is of form [xyz + sdf]
                if train_mode:
                    for idx in range(len(self._data)):
                        self._data[idx] = np.hstack(
                            (self._data[idx]["xyz"], self._data[idx]["sdf"])
                        )

            # Data is in train mode
            else:
                for idx in range(len(self._data)):
                    self._data[idx] = sdf_to_occ(self._data[idx].astype(float), skip_cols=self._num_dims)

                # Do a quick check that the values are correct
                if validate_occ:
                    for idx, d in enumerate(self._data):
                        assert (
                            d[:, self._num_dims]
                            == d[:, self._num_dims + 1] + d[:, self._num_dims + 2]
                        ).all(), "Occupancy values are incorrect for object {}".format(
                            idx
                        )

        else:
            if self._clamp_dist is not None:
                logging.debug("Clamping sdf samples to +/-{} ...".format(self._clamp_dist))
                
                # Data is in test mode
                if isinstance(self._data[0], dict):
                    for idx in range(len(self._data)):
                        self._data[idx]["sdf"] = clamp_samples(
                            self._data[idx]["sdf"], self._clamp_dist
                        ).astype(float)

                # Data is in train mode
                else:
                    for idx in range(len(self._data)):
                        self._data[idx] = clamp_samples(self._data[idx], self._clamp_dist, skip_cols=self._num_dims)

    @property
    def num_dims(self):
        return self._num_dims

    @property
    def is_occ(self):
        return self._use_occ

    @property
    def num_shapes(self):
        """Number of complete objects"""
        return self._num_shapes

    @property
    def num_instances(self):
        """Number of training/testing instances"""
        return self._num_instances

    @property
    def objects(self):
        return self._objects.copy()

    @property
    def data(self):
        return self._data.copy()

    @property
    def indexer(self):
        return self._indexer.copy()

    @property
    def complete_indexer(self):
        return self._complete_indexer.copy()

    @property
    def num_components(self):
        """Return an array with the number of components associated with each restoration shape"""
        if self._num_components is None:
            self._num_components = np.array(
                [
                    core.metrics.connected_components(self.get_mesh(idx, 2))
                    for idx in range(len(self))
                ]
            )
        return self._num_components.copy()

    def get_complete_index(self, idx):
        """
        Return the means to access a specific object.

        obj index | break index | overall index
        0 : 0 : 0
        0 : 1 : 1
        1 : 0 : 2
        1 : 1 : 3
        """
        obj_idx, _ = self._indexer[idx]
        return obj_idx

    def get_broken_index(self, idx):
        """
        Return the means to access a specific object.

        obj index | break index | overall index
        0 : 0 : 0
        0 : 1 : 1
        1 : 0 : 2
        1 : 1 : 3
        """
        _, break_idx = self._indexer[idx]
        return break_idx

    def get_object(self, idx):
        """Return a specific object"""
        obj_idx, _ = self._indexer[idx]
        return self._objects[obj_idx]

    def get_sample(self, idx, shape_idx):
        """
        Return a complete, broken, or restoration samples. Each sample returned
        will be of the form: (pts, samples).

        Args:
            idx: index of the break to return.
            shape_idx: index of the shape to return in ["complete", "broken",
                "restoration"].
        """

        if self._learned_breaks:
            assert 0 <= shape_idx <= 1
        else:
            assert 0 <= shape_idx <= 4
        assert not isinstance(
            self._data[0], dict
        ), "Dataloader is in test mode, use 'get_broken_sample()' instead"

        return tuple(
            (
                self._data[idx][:, : self._num_dims].copy(),
                np.expand_dims(
                    self._data[idx][:, self._num_dims + shape_idx].copy(), axis=1
                ),
            )
        )

    def get_broken_sample(self, idx, return_mask=False):
        """
        Return broken samples (for testing). Each sample returned
        will be of the form: (pts, samples). Optionally return a mask for partial
        view synthesis.

        Args:
            idx: index of the break to return.
            return_mask: return a partial view mask.
        """
        assert isinstance(
            self._data[0], dict
        ), "Dataloader is in training mode, use 'get_sample()' instead"

        if return_mask:
            return (
                self._data[idx]["xyz"].copy(),
                self._data[idx]["sdf"].copy(),
                self._data[idx]["mask"].copy(),
            )
        return self._data[idx]["xyz"].copy(), self._data[idx]["sdf"].copy()

    def get_mesh(self, idx, shape_idx):
        """
        Return the mesh for a specific sample.
        """
        assert 0 <= shape_idx <= 3
        obj_idx, break_idx = self._indexer[idx]
        obj = self._objects[obj_idx]
        if shape_idx == 0:
            return obj.load(obj.path_c())
        elif shape_idx == 1:
            return obj.load(obj.path_b(break_idx))
        elif shape_idx == 2:
            return obj.load(obj.path_r(break_idx))
        elif shape_idx == 3:
            return obj.load(obj.path_tool(break_idx))

    def get_tool(self, idx):
        """
        Return the mesh for a specific sample.
        """
        obj_idx, break_idx = self._indexer[idx]
        obj = self._objects[obj_idx]
        return obj.load(obj.path_tool(break_idx))

    def get_render(self, idx, shape_idx, angle=0, resolution=(200, 200), save=True):
        """
        Return render for a specific sample.
        """
        assert 0 <= shape_idx <= 2
        obj_idx, break_idx = self._indexer[idx]
        obj = self._objects[obj_idx]

        try:
            if shape_idx == 0:
                path = obj.path_c_rendered(angle=angle, resolution=resolution)
            elif shape_idx == 1:
                path = obj.path_b_rendered(
                    idx=break_idx, angle=angle, resolution=resolution
                )
            elif shape_idx == 2:
                path = obj.path_r_rendered(
                    idx=break_idx, angle=angle, resolution=resolution
                )
            try:
                return obj.load(path)
            except FileNotFoundError:
                pass

            logging.debug(
                "Render ({}, {}, {}) with filename {} could not be found, generating".format(
                    idx,
                    shape_idx,
                    angle,
                    path,
                )
            )
        except (errors.PathAccessError, PermissionError):
            save = False  # Cannot access the path, so cannot load or save

            logging.debug(
                "Render ({}, {}, {}) could not be accessed, generating".format(
                    idx,
                    shape_idx,
                    angle,
                )
            )

        render = core.utils_3d.render_mesh(
            self.get_mesh(idx, shape_idx),
            yrot=angle,
            resolution=resolution,
            remove_texture=True,
        )

        if save:
            try:
                core.handler.saver(path, render)
            except PermissionError:
                pass
        return render

    def get_composite(
        self,
        idx,
        angle=0,
        resolution=(200, 200),
        self_color=(128, 64, 64, 255),
        gt_color=(64, 128, 64, 128),
        save=True,
    ):
        obj_idx, break_idx = self._indexer[idx]
        obj = self._objects[obj_idx]

        try:
            path = obj.path_composite(idx=break_idx, angle=angle, resolution=resolution)
            try:
                return obj.load(path)
            except FileNotFoundError:
                pass

            logging.debug(
                "Composite ({}, {}) with filename {} could not be found, generating".format(
                    idx,
                    angle,
                    path,
                )
            )
        except (errors.PathAccessError, PermissionError):
            save = False  # Cannot access the path, so cannot load or save

            logging.debug(
                "Composite ({}, {}) could not be accessed, generating".format(
                    idx,
                    angle,
                )
            )

        # Get the meshes
        r_mesh = self.get_mesh(idx, 2)
        b_mesh = self.get_mesh(idx, 1)

        # Update the colors
        r_mesh.visual = trimesh.visual.color.ColorVisuals(
            r_mesh, vertex_colors=self_color
        )
        b_mesh.visual = trimesh.visual.color.ColorVisuals(
            b_mesh, vertex_colors=gt_color
        )

        # Render
        render = core.utils_3d.render_mesh(
            [b_mesh, r_mesh],
            yrot=angle,
            resolution=resolution,
        )

        if save:
            try:
                core.handler.saver(path, render)
            except PermissionError:
                pass
        return render

    def get_classname(self, idx):
        """
        Return classname for a given shape.
        """
        return self._objects[self._indexer[idx][0]].class_id

    def __len__(self):
        """
        Return the number of discrete shapes in the dataset.
        """
        return self.num_instances

    def __getitem__(self, idx):
        """
        Return
            [[pts, c, b, r], [idx]]
            OR
            [[pts, c, t], [idx, b_idx]]
        """
        if self._learned_breaks:

            # Get the corresponding complete object index
            complete_idx = self.get_complete_index(idx)

            # Will return [pts], [c, b, r, t]
            pts, values = core.data.select_samples(
                self._data[idx][:, : self._num_dims].copy(),
                self._data[idx][:, self._num_dims :].copy(),
                self._subsample,
                self._uniform_ratio,
            )

            return tuple(
                [
                    torch.from_numpy(pts),
                    torch.from_numpy(values[:, 0]).unsqueeze(1),  # Complete
                    torch.from_numpy(values[:, 1]).unsqueeze(1),  # Tool
                ]
            ), tuple(
                [
                    torch.tensor(complete_idx)
                    .repeat(1, self._subsample)
                    .T,  # Complete idx
                    torch.tensor(idx).repeat(1, self._subsample).T,  # Break idx
                ]
            )

        else:

            # Will return [pts], [c, b, r]
            pts, values = core.data.select_samples(
                self._data[idx][:, : self._num_dims].copy(),
                self._data[idx][:, self._num_dims :].copy(),
                self._subsample,
                self._uniform_ratio,
            )

            return tuple(
                [torch.from_numpy(pts)]
                + [torch.from_numpy(v).unsqueeze(1) for v in values.T]
            ), tuple([torch.tensor(idx).repeat(1, self._subsample).T])
