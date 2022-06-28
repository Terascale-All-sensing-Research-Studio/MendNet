import logging
import trimesh
import numpy as np

from scipy.spatial import cKDTree as KDTree

try:
    from libmesh import check_mesh_contains
except ImportError:
    pass
try:
    import pymesh
except ImportError:
    pass

import core
import core.errors as errors


def chamfer(gt_shape, pred_shape, num_mesh_samples=30000):
    """
    Compute the chamfer distance for two 3D meshes.
    This function computes a symmetric chamfer distance, i.e. the mean chamfers.
    Based on the code provided by DeepSDF.

    Args:
        gt_shape (trimesh object or points): Ground truth shape.
        pred_shape (trimesh object): Predicted shape.
        num_mesh_samples (points): Number of points to sample from the predicted
            shape. Must be the same number of points as were computed for the
            ground truth shape.
    """

    if pred_shape.vertices.shape[0] == 0:
        raise core.errors.MeshEmptyError
    assert gt_shape.vertices.shape[0] != 0, "gt shape has no vertices"

    try:
        gt_pts = trimesh.sample.sample_surface(gt_shape, num_mesh_samples)[0]
    except AttributeError:
        gt_pts = gt_shape
        assert (
            gt_pts.shape[0] == num_mesh_samples
        ), "Wrong number of gt points, expected {} got {}".format(
            num_mesh_samples, gt_pts.shape[0]
        )
    pred_pts = trimesh.sample.sample_surface(pred_shape, num_mesh_samples)[0]

    # one direction
    one_distances, _ = KDTree(pred_pts).query(gt_pts)
    gt_to_pred_chamfer = np.mean(np.square(one_distances))

    # other direction
    two_distances, _ = KDTree(gt_pts).query(pred_pts)
    pred_to_gt_chamfer = np.mean(np.square(two_distances))

    return gt_to_pred_chamfer + pred_to_gt_chamfer


def exact_union(mesh1, mesh2):
    union = core.utils_3d.pymesh2trimesh(
        pymesh.boolean(
            core.utils_3d.trimesh2pymesh(mesh1),
            core.utils_3d.trimesh2pymesh(mesh2),
            "union",
        )
    )
    return union


def exact_iou(mesh1, mesh2):
    intersection = core.utils_3d.pymesh2trimesh(
        pymesh.boolean(
            core.utils_3d.trimesh2pymesh(mesh1),
            core.utils_3d.trimesh2pymesh(mesh2),
            "intersection",
        )
    )

    union = core.utils_3d.pymesh2trimesh(
        pymesh.boolean(
            core.utils_3d.trimesh2pymesh(mesh1),
            core.utils_3d.trimesh2pymesh(mesh2),
            "union",
        )
    )
    return intersection.volume / union.volume


def iou(gt_shape, pred_shape, n=128, padding=0.0):
    """
    Approximation of the intersection over union.
    Will fallback to cleanup and boolean operations if occupancies cannot be computed.
    """

    if not gt_shape.is_watertight:
        gt_shape = core.utils_3d.repair_watertight(gt_shape)
    if not pred_shape.is_watertight:
        pred_shape = core.utils_3d.repair_watertight(pred_shape)
    if not gt_shape.is_watertight or not pred_shape.is_watertight:
        logging.debug("Meshes are not watertight, falling back to pymesh")
        return exact_iou(gt_shape, pred_shape)

    # Get the query points
    grid_pts = np.meshgrid(
        *[np.linspace(0, 1.0 + (padding * 2), d) - (0.5 + padding) for d in (n, n, n)]
    )
    query_pts = np.vstack([p.flatten() for p in grid_pts]).T

    # Get occupancies
    try:
        gt_occ = check_mesh_contains(pred_shape, query_pts)
    except RuntimeError:
        logging.debug("IOU failed on predicted, falling back to pymesh")
        return exact_iou(gt_shape, pred_shape)
    try:
        pd_occ = check_mesh_contains(gt_shape, query_pts)
    except RuntimeError:
        logging.debug("IOU failed on gt, falling back to pymesh")
        return exact_iou(gt_shape, pred_shape)

    # Compute iou
    intersection = (np.logical_and(gt_occ, pd_occ)).sum()
    union = np.logical_or(gt_occ, pd_occ).sum()
    if union == 0:
        return 0
    return intersection / union


def union_score(pd_complete, pd_broken, pd_restoration, padding=0.0, n=128):

    if not pd_complete.is_watertight:
        pd_complete = core.utils_3d.repair_watertight(pd_complete)
    if not pd_broken.is_watertight:
        pd_broken = core.utils_3d.repair_watertight(pd_broken)
    if not pd_restoration.is_watertight:
        pd_restoration = core.utils_3d.repair_watertight(pd_restoration)

    if (
        not pd_complete.is_watertight
        or not pd_broken.is_watertight
        or not pd_restoration.is_watertight
    ):
        logging.debug("Meshes are not watertight, falling back to pymesh")
        return exact_iou(
            exact_union(pd_broken, pd_restoration),
            pd_complete,
        )

    # Get the query points
    grid_pts = np.meshgrid(
        *[np.linspace(0, 1.0 + (padding * 2), d) - (0.5 + padding) for d in (n, n, n)]
    )
    query_pts = np.vstack([p.flatten() for p in grid_pts]).T

    # Get occupancies
    try:
        c_occ = check_mesh_contains(pd_complete, query_pts)
    except RuntimeError:
        logging.debug("IOU failed, falling back to pymesh")
        return exact_iou(
            exact_union(pd_broken, pd_restoration),
            pd_complete,
        )
    try:
        b_occ = check_mesh_contains(pd_broken, query_pts)
    except RuntimeError:
        logging.debug("IOU failed, falling back to pymesh")
        return exact_iou(
            exact_union(pd_broken, pd_restoration),
            pd_complete,
        )
    try:
        r_occ = check_mesh_contains(pd_restoration, query_pts)
    except RuntimeError:
        logging.debug("IOU failed, falling back to pymesh")
        return exact_iou(
            exact_union(pd_broken, pd_restoration),
            pd_complete,
        )

    # Compute iou
    combined_b_r = np.logical_or(b_occ, r_occ)
    intersection = np.logical_and(c_occ, combined_b_r).sum()
    union = np.logical_or(c_occ, combined_b_r).sum()
    if union == 0:
        return 0
    return intersection / union


def intersection_score(broken, restoration):
    return iou(broken, restoration)


def connected_components(mesh):
    """
    Return number of connected components.
    """
    if mesh.vertices.shape[0] == 0:
        raise core.errors.MeshEmptyError
    return len(core.utils_3d.trimesh2vedo(mesh).splitByConnectivity())


def get_intersection_points(a, b, sig=5):
    """get mask of vertices in a occurring in both a and b, corresponding to a"""
    av = [frozenset(np.round(v, sig)) for v in a]
    bv = set([frozenset(np.round(v, sig)) for v in b])
    return np.asarray(list(map(lambda v: v in bv, av)))


def get_fracture_points(b, r):
    """ Get points on the fracture """
    try:
        vb, vr = b.vertices, r.vertices
    except AttributeError:
        vb, vr = b, r

    logging.debug(
        "Computing fracture points for meshes with size {} and {} ..."
        .format(vb.shape[0], vr.shape[0])
    )
    return get_intersection_points(vb, vr)
    

def connected_artifacts_score2(
    gt_complete, gt_broken, gt_restoration, pd_restoration, max_dist=0.02, num_mesh_samples=30000
):

    if pd_restoration.vertices.shape[0] == 0:
        raise core.errors.MeshEmptyError
    assert gt_complete.vertices.shape[0] != 0, "gt shape has no vertices"
    assert gt_broken.vertices.shape[0] != 0, "gt shape has no vertices"
    assert gt_restoration.vertices.shape[0] != 0, "gt shape has no vertices"

    # Get the fracture and exterior vertices
    exterior_verts = np.ones(gt_broken.vertices.shape[0]).astype(bool)
    exterior_verts[get_fracture_points(gt_broken, gt_restoration)] = False

    # Get the associated faces
    # fracture_faces = fracture_verts[gt_broken.faces].all(axis=1)
    exterior_faces = exterior_verts[gt_broken.faces].all(axis=1)

    # Sample the broken
    gt_broken_points, face_inds = trimesh.sample.sample_surface(gt_broken, num_mesh_samples)
    _, exterior_inds, _ = np.intersect1d(
        face_inds, np.where(exterior_faces)[0], return_indices=True
    )
    exterior_points = gt_broken_points[exterior_inds, :]

    # Sample the restorations
    gt_restoration_points = trimesh.sample.sample_surface(gt_restoration, num_mesh_samples)[0]
    pd_restoration_points = trimesh.sample.sample_surface(pd_restoration, num_mesh_samples)[0]

    # Throw out exterior points that have a close point in the gt restoration
    d = KDTree(gt_restoration_points).query(exterior_points)[0]
    exterior_points = exterior_points[d > max_dist, :]

    # What percentage of exterior points DO have close neighbors when they SHOULDNT?
    return (
        KDTree(pd_restoration_points).query(exterior_points)[0] < max_dist
    ).sum() / exterior_points.shape[0]


def complete_sameness(mesh, gt_complete, gt_broken):
    """
    Return boolean indicating if the predicted mesh is closer to the complete
    mesh than it is to the broken mesh.
    """
    return int(chamfer(mesh, gt_complete) < chamfer(mesh, gt_broken))


def difference(gt_shape, pred_shape, pow=2, rounded=True):
    """
    Return absolute value of the difference between two shapes, raised to
    a power. Shapes should have the same dimensions. This function should be
    used to compare the structured outputs of the network after grid
    sampling, ie voxels or image values.

    Args:
        gt_shape (grid of values): Ground truth shape.
        pred_shape (grid of values): Predicted shape.
    """
    assert gt_shape.shape == pred_shape.shape, (
        "Input shapes must have the "
        + "same dimensions. Shapes had dimensions {} (gt) and {} (pred)".format(
            gt_shape.shape, pred_shape.shape
        )
    )

    gt_shape = gt_shape.astype(np.float)
    pred_shape = pred_shape.astype(np.float)
    if rounded:
        return np.sum(np.abs(gt_shape.round() - pred_shape.round()) ** pow)
    else:
        return np.sum(np.abs(gt_shape - pred_shape) ** pow)


def estimate_break_percent(broken_samples, complete_samples):
    """
    Estimate the percent of the object broken off using the uniformally
    sampled occupancy points.
    """
    assert broken_samples.shape[0] == complete_samples.shape[0]
    num_samples = broken_samples.shape[0]

    # Extract only the uniform points
    broken_uniform_pts = broken_samples[int(num_samples / 2) :, :]
    complete_uniform_pts = complete_samples[int(num_samples / 2) :, :]

    # Compute the ratio
    assert complete_uniform_pts.sum() > broken_uniform_pts.sum()
    val = 1 - (broken_uniform_pts.sum() / complete_uniform_pts.sum())

    return val


def compute_break_percent(gt_restoration, gt_complete, method="volume"):
    """Compute the percent of an object removed by a break"""

    def num_intersecting(a, b, thresh=1e-8):
        """Return number of intersecting vertices"""
        d, _ = KDTree(a).query(b)
        return (d < thresh).sum()

    if method == "volume":
        return gt_restoration.volume / gt_complete.volume
    elif method == "surface_area":
        return (
            num_intersecting(gt_complete.vertices, gt_restoration.vertices)
            / gt_complete.vertices.shape[0]
        )
    else:
        raise RuntimeError("Unknown method {}".format(method))
