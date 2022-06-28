import logging

import numpy as np

import core.eval as evaluator
import core.errors as errors


def build_metric(
    reconstruct_list, reconstruction_handler, output_pairs, metric, stats=True
):
    """
    Gets a given metric
    """

    # Get the metric
    values = np.empty((len(reconstruct_list), len(output_pairs)))
    values[:] = np.nan

    for ii, idx in enumerate(reconstruct_list):
        for e, (gt_idx, pd_idx) in enumerate(output_pairs):
            try:
                # Try to load the value from disk
                value = np.load(
                    reconstruction_handler.path_eval(
                        idx,
                        pd_idx,
                        gt_shape_idx=gt_idx,
                        metric=metric,
                    )
                )
            except (FileNotFoundError, ValueError):
                logging.debug(
                    "Metric: {} could not be loaded for pair {}".format(
                        metric, (pd_idx, gt_idx)
                    )
                )
                continue
            except errors.IsosurfaceExtractionError:
                continue
            if np.isinf(value):
                continue
            values[ii, e] = value
    # Values will be size (num_objects, output_pars)

    logging.info(
        "Num generated {} {}".format(
            np.logical_not(np.isnan(values)).astype(int).sum(axis=0),
            metric,
        )
    )

    # Add some summary statistics
    if stats:
        values = np.vstack(
            (
                np.expand_dims(
                    np.nanmean(values, axis=0), axis=0
                ),  # Get column-wise mean
                np.expand_dims(
                    np.nanstd(values, axis=0), axis=0
                ),  # Get column-wise std
                np.expand_dims(
                    np.logical_not(np.isnan(values)).astype(int).sum(axis=0), axis=0
                ),  # Get column-wise number of non-nan values
                values,
            )
        )

    return values


def get_alias(name, aliases):
    """Given a metric name and a list of aliases, return the corresponding alias"""
    for (m_old, m_new) in aliases:
        if m_old == name:
            return m_new
    raise RuntimeError("Metric {} has no alias in {}".format(name, aliases))


def export_report(
    out_folder,
    out_metrics,
    reconstruction_handler,
    reconstruct_list,
    output_pairs,
    exclusion_list,
    metrics=["chamfer"],
    save_plot=True,
    save_csv=True,
    compare_to_baselines=False,
    shape_idx_aliases=["C", "B", "R"],
):
    """
    Creates the following files:
        - out_metrics: .npy file containing all of the data
        - out_folder:
    """

    assert len(output_pairs) >= 3
    logging.info("Output pairs: {}".format(output_pairs))
    logging.info("Column 0 will be interpreted as C")
    logging.info("Column 1 will be interpreted as B")
    logging.info("Column 2 will be interpreted as R")
    if len(output_pairs) > 3:
        logging.info(
            "More than 3 output pairs passed. Please ensure you know what you're doing."
        )

    num_shapes = len(output_pairs)
    assert num_shapes == len(
        shape_idx_aliases
    ), "Each output pair needs a name. Incorrect number of names given: {} for output pairs: {}".format(
        shape_idx_aliases, output_pairs
    )

    # Extract the metrics data
    metrics_dict = {
        m: build_metric(reconstruct_list, reconstruction_handler, output_pairs, m)
        for m in metrics
    }
    metrics_dict["reconstruct_list"] = reconstruct_list

    # Save the dictionary to disk
    np.save(out_metrics, metrics_dict)