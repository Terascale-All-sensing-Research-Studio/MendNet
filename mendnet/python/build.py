import os
import argparse
import logging
import json
import pickle

import tqdm
import zipfile
import numpy as np

import core
from processor.shapenet import ShapeNetObject


def main(
    root_dir,
    train_out,
    test_out,
    splits_file,
    num_breaks,
    cache_models,
    use_occ,
    use_pointer,
):
    logging.info("Loading saved data from splits file {}".format(splits_file))

    object_id_dict = json.load(open(splits_file, "r"))
    id_train_list = [
        ShapeNetObject(root_dir, o[0], o[1]) for o in object_id_dict["id_train_list"]
    ]
    id_test_list = [
        ShapeNetObject(root_dir, o[0], o[1]) for o in object_id_dict["id_test_list"]
    ]

    logging.info("Will save train data to: {}".format(train_out))
    if train_out is not None:
        sdf_list = []
        index_list = []
        complete_index_list = []
        class_list = []
        logging.info("Loading train list")
        for obj_idx, obj in tqdm.tqdm(enumerate(id_train_list)):

            breaks_loaded = []
            for break_idx in range(num_breaks):
                if (
                    os.path.exists(obj.path_sampled(break_idx))
                    and os.path.exists(obj.path_c_sdf(break_idx))
                    and os.path.exists(obj.path_b_sdf(break_idx))
                    and os.path.exists(obj.path_r_sdf(break_idx))
                ):
                    try:
                        sdf_sample = np.hstack(
                            (
                                obj.load(obj.path_sampled(break_idx), skip_cache=True)[
                                    "xyz"
                                ],
                                np.expand_dims(
                                    obj.load(
                                        obj.path_c_sdf(break_idx), skip_cache=True
                                    )["sdf"],
                                    axis=1,
                                ),
                                np.expand_dims(
                                    obj.load(
                                        obj.path_b_sdf(break_idx), skip_cache=True
                                    )["sdf"],
                                    axis=1,
                                ),
                                np.expand_dims(
                                    obj.load(
                                        obj.path_r_sdf(break_idx), skip_cache=True
                                    )["sdf"],
                                    axis=1,
                                ),
                            )
                        )
                    except (zipfile.BadZipFile):
                        logging.warning(
                            "Sample ({}, {}) is corrupted, skipping".format(
                                obj_idx, break_idx
                            )
                        )

                    occ_sample = core.data.sdf_to_occ(
                        sdf_sample.astype(float), skip_cols=3
                    )

                    class_list.append(obj.class_id)

                    # This is a sanity check
                    if not (
                        occ_sample[:, 3] == occ_sample[:, 4] + occ_sample[:, 5]
                    ).all():
                        logging.warning(
                            "Sample ({}, {}) is invalid, skipping.".format(
                                obj_idx, break_idx
                            )
                        )
                        continue

                    # Convert to occ
                    if use_occ:
                        sdf_sample = {
                            "xyz": sdf_sample[:, :3].astype(np.float16),
                            "sdf": occ_sample[:, 3:].astype(bool),
                        }

                    breaks_loaded.append(len(sdf_list))
                    sdf_list.append(sdf_sample)
                    index_list.append(
                        (
                            obj_idx,
                            break_idx,
                        )
                    )

            if len(breaks_loaded) > 0:
                complete_index_list.append(breaks_loaded)

            # Make sure the cache is empty
            obj._cache = {}


        class_set = set(class_list)
        print("Counts:")
        for c in class_set:
            print("{}: {}".format(c, 
                len([c1 for c1 in class_list if c1 == c])
            ))
        data_dict = {
            "indexer": index_list,
            "complete_indexer": complete_index_list,
            "objects": id_train_list,
            "use_occ": use_occ,
            "train": True,
        }
        tr_out, _ = os.path.splitext(train_out)

        if use_pointer:
            sdf_path = tr_out + "_sdf.npz"
            logging.info("Saving sdf values to: {}".format(sdf_path))

            data_dict["sdf"] = sdf_path
            np.savez(sdf_path, sdf=sdf_list)
        else:
            data_dict["sdf"] = sdf_list

        logging.info("num samples loaded: {}".format(len(sdf_list)))
        logging.info("Saving data ...")
        pickle.dump(
            data_dict,
            open(train_out, "wb"),
            pickle.HIGHEST_PROTOCOL,
        )

    logging.info("Will save test data to: {}".format(test_out))
    if test_out is not None:
        sdf_list = []
        index_list = []
        complete_index_list = []
        logging.info("Loading test list")
        for obj_idx, obj in tqdm.tqdm(enumerate(id_test_list)):

            breaks_loaded = []
            for break_idx in range(num_breaks):

                # if break_idx < 6:
                #     continue

                if os.path.exists(obj.path_b_partial_sdf(break_idx)):
                    try:
                        sdf_sample = obj.load(
                            obj.path_b_partial_sdf(break_idx), skip_cache=True
                        )
                    except (zipfile.BadZipFile):
                        logging.warning(
                            "Sample ({}, {}) is corrupted, skipping".format(
                                obj_idx, break_idx
                            )
                        )

                    # # Convert to occ
                    # if use_occ:
                    #     sdf_sample["sdf"] = core.data.sdf_to_occ(sdf_sample["sdf"]).astype(bool)

                    breaks_loaded.append(len(sdf_list))
                    sdf_list.append(sdf_sample)
                    index_list.append(
                        (
                            obj_idx,
                            break_idx,
                        )
                    )

                    if cache_models:
                        obj.load(obj.path_b(break_idx))
                        obj.load(obj.path_r(break_idx))

            if len(breaks_loaded) > 0:
                complete_index_list.append(breaks_loaded)
                if cache_models:
                    obj.load(obj.path_c())

        logging.info("num samples loaded: {}".format(len(sdf_list)))
        logging.info("Saving data ...")
        pickle.dump(
            {
                "sdf": sdf_list,
                "indexer": index_list,
                "complete_indexer": complete_index_list,
                "objects": id_test_list,
                "train": False,
                # "use_occ": use_occ,
            },
            open(test_out, "wb"),
            pickle.HIGHEST_PROTOCOL,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build a train and test pkl file from a splits file. The "
        + "pkl files will only contain valid samples. Optionally preload "
        + "upsampled points and models to accelerate evaluation."
    )
    parser.add_argument(
        dest="input",
        type=str,
        help="Location of the database. Pass the top level directory. For "
        + "ShapeNet this would be ShapeNet.v2",
    )
    parser.add_argument(
        dest="splits",
        type=str,
        help=".json file path, this file will be created and will store the "
        + "ids of all objects in the training and testing split.",
    )
    parser.add_argument(
        "--train_out",
        default=None,
        type=str,
        help="Where to save the resulting train database file. Should be a .pkl",
    )
    parser.add_argument(
        "--test_out",
        default=None,
        type=str,
        help="Where to save the resulting test database file. Should be a .pkl",
    )
    parser.add_argument(
        "--breaks",
        "-b",
        type=int,
        default=1,
        help="Number of breaks to generate for each object. This will only be "
        + "used if BREAK is passed.",
    )
    parser.add_argument(
        "--load_models",
        action="store_true",
        default=False,
        help="If passed, will preload object models.",
    )
    parser.add_argument(
        "--use_occ",
        action="store_true",
        default=False,
        help="If passed, will compress the data using occupancy samples. Note that "
        + "this will not allow you to train sdf models. NOTE: THIS HAS BEEN DISABLED "
        + "FOR TESTING SPLITS.",
    )
    parser.add_argument(
        "--use_pointer",
        action="store_true",
        default=False,
        help="If passed, will save a pointer to the file."
    )
    core.add_common_args(parser)
    args = parser.parse_args()
    core.configure_logging(args)

    main(
        args.input,
        args.train_out,
        args.test_out,
        args.splits,
        args.breaks,
        args.load_models,
        args.use_occ,
        args.use_pointer,
    )
