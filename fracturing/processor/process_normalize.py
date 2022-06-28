import argparse

import trimesh
import numpy as np

import processor.logger as logger
import processor.errors as errors


def smooth(f_in, f_out, lamb=0.5, iterations=10):
    """Perform laplacian smoothing on a mesh"""
    # Load mesh
    mesh = trimesh.load(f_in)
    if len(mesh.vertices) > 1000000:
        raise errors.MeshSizeError
    if not mesh.is_watertight:
        raise errors.MeshNotClosedError
    mesh = trimesh.smoothing.filter_laplacian(mesh, lamb, iterations)

    # mesh.vertex_normals
    mesh.export(f_out)


def normalize(f_in, f_out, skip_check):
    """Translate and rescale a mesh so that it is centered inside a unit cube"""
    # Load mesh
    mesh = trimesh.load(f_in)
    if not skip_check:
        if len(mesh.vertices) > 1000000:
            raise errors.MeshSizeError
    if not mesh.is_watertight:
        raise errors.MeshNotClosedError

    # Get the overall size of the object
    mesh_min, mesh_max = np.min(mesh.vertices, axis=0), np.max(mesh.vertices, axis=0)
    size = mesh_max - mesh_min

    # Center the object
    vertices = mesh.vertices - ((size / 2.0) + mesh_min)

    # Normalize scale of the object
    vertices = vertices * (1.0 / np.max(size))

    # Save
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=mesh.faces,
        vertex_colors=mesh.visual.vertex_colors,
    ).export(f_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apply laplacian smoothing or unit cube normalization. Will "
        + "fail if the mesh is not waterproof."
    )
    parser.add_argument(dest="input", type=str, help="Path to the input file.")
    parser.add_argument(dest="output", type=str, help="Path to the output file.")
    parser.add_argument(
        "--smooth",
        action="store_true",
        default=False,
        help="If passed, will smooth the mesh instead of performing unit cube "
        + "normalization.",
    )
    parser.add_argument(
        "--lamb", type=float, default=0.5, help="Lambda value for laplacian smoothing."
    )
    parser.add_argument(
        "--iterations", type=int, default=10, help="Iterations for laplacian smoothing."
    )
    parser.add_argument(
        "--skip_check", action="store_true", default=False, help="Skip size check."
    )
    logger.add_logger_args(parser)
    args = parser.parse_args()
    logger.configure_logging(args)

    # Process
    if args.smooth:
        smooth(
            f_in=args.input,
            f_out=args.output,
            lamb=args.lamb,
            iterations=args.iterations,
        )
    else:
        normalize(f_in=args.input, f_out=args.output, skip_check=args.skip_check)
