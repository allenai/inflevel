import argparse
import copy
from typing import Sequence, Tuple, Literal

import compress_pickle
import numpy as np
import pandas as pd


def split_datasets_using_key(
    datasets: Sequence[np.ndarray],
    metadata_df: pd.DataFrame,
    groupby_keys: Sequence[str],
    meta_key: str,
    expected_skips: int = 0,
):
    to_join = sum([[metadata_df[k], "__"] for k in groupby_keys], [])
    joined = list(sum((tj for tj in to_join[1:-1]), to_join[0]))

    key_to_type_and_index = {}
    for i, (key, trial_type) in enumerate(zip(joined, metadata_df[meta_key])):
        if key not in key_to_type_and_index:
            key_to_type_and_index[key] = []

        key_to_type_and_index[key].append((i, trial_type))

    num_splits = len(set(metadata_df[meta_key]))
    could_not_find_ids = []
    val_to_ordered_inds = {v: [] for v in list(sorted(set(metadata_df[meta_key])))}
    for trial_group in key_to_type_and_index.values():
        if len(trial_group) > num_splits:
            raise NotImplementedError
        elif len(trial_group) < num_splits:
            could_not_find_ids.append(trial_group)
        else:
            for index, trial_type in trial_group:
                val_to_ordered_inds[trial_type].append(index)

    if len(could_not_find_ids) == 0 or len(could_not_find_ids) != expected_skips:
        print(
            f"COULD NOT FIND {len(could_not_find_ids)} ids in row_id_to_index (expected to skip {expected_skips}), SKIPPED!"
        )

    split_datasets = tuple(
        {val: dataset[val_to_ordered_inds[val]] for val in val_to_ordered_inds}
        for dataset in datasets
    )
    metadata_df_dict = {
        val: metadata_df.iloc[val_to_ordered_inds[val]].reset_index() for val in val_to_ordered_inds
    }
    return split_datasets, metadata_df_dict


def _scores_to_metric(real0: np.ndarray, real1: np.ndarray, magic0: np.ndarray, magic1: np.ndarray):
    assert real0.shape == real1.shape == magic0.shape == magic1.shape
    assert len(real0.shape) == 1 and real0.shape[0] > 0
    count = 0
    for real_scores in [real0, real1]:
        for magic_scores in [magic0, magic1]:
            count += (real_scores > magic_scores).sum() + 0.5 * (real_scores == magic_scores).sum()
    return count / (4.0 * real0.shape[0])


def _permutation_test(
    real0: np.ndarray,
    real1: np.ndarray,
    magic0: np.ndarray,
    magic1: np.ndarray,
    min_expected_max: Tuple[float, float, float],
    reps: int = 100,
    seed=0,
):
    assert real0.shape == real1.shape == magic0.shape == magic1.shape
    assert len(real0.shape) == 1 and real0.shape[0] > 0

    v = _scores_to_metric(real0=real0, real1=real1, magic0=magic0, magic1=magic1)

    all_together = np.stack((real0, real1, magic0, magic1), axis=0)
    metrics = []

    rng = np.random.default_rng(seed)
    col_indexer = np.arange(0, real0.shape[0]).reshape(1, -1)
    for _ in list(range(reps)):
        permuted_rows_indexer = rng.random(all_together.shape).argsort(axis=0)
        permuted = all_together[permuted_rows_indexer, col_indexer]
        metrics.append(_scores_to_metric(*permuted))

    metrics = np.array(metrics)
    a, e, b = min_expected_max
    rescaled_metrics = copy.deepcopy(metrics)
    assert (metrics >= a).all() and (metrics <= b).all()
    rescaled_metrics -= e
    rescaled_metrics = rescaled_metrics / (
        (b - e) * (rescaled_metrics >= 0.0) + (e - a) * (rescaled_metrics < 0.0)
    )

    rescaled_v = (v - e) / ((b - e) if v >= e else (e - a))

    return {
        "accuracy": v,
        "pval_1sided": (metrics >= v).mean(),
        "pval": (
            1
            if rescaled_v == 0
            else (
                (rescaled_metrics >= abs(rescaled_v)).mean()
                + (rescaled_metrics <= -abs(rescaled_v)).mean()
            )
        ),
        "perm_vals": metrics,
        "min_expected_max": min_expected_max,
    }


def continuity_scores_to_metric(vv: np.ndarray, ii: np.ndarray, iv: np.ndarray, vi: np.ndarray):
    return _scores_to_metric(real0=vv, real1=ii, magic0=iv, magic1=vi)


def continuity_scores_permutation_test(
    vv: np.ndarray,
    ii: np.ndarray,
    iv: np.ndarray,
    vi: np.ndarray,
    min_expected_max: Tuple[float, float, float],
    reps: int = 100,
    seed=0,
):
    return _permutation_test(
        real0=vv,
        real1=ii,
        magic0=iv,
        magic1=vi,
        min_expected_max=min_expected_max,
        reps=reps,
        seed=seed,
    )


def solidity_scores_to_metric(ui: np.ndarray, cv: np.ndarray, uv: np.ndarray, ci: np.ndarray):
    return _scores_to_metric(real0=ui, real1=cv, magic0=uv, magic1=ci)


def solidity_scores_permutation_test(
    ui: np.ndarray,
    cv: np.ndarray,
    uv: np.ndarray,
    ci: np.ndarray,
    min_expected_max: Tuple[float, float, float],
    reps: int = 100,
    seed=0,
):
    return _permutation_test(
        real0=ui,
        real1=cv,
        magic0=uv,
        magic1=ci,
        min_expected_max=min_expected_max,
        reps=reps,
        seed=seed,
    )


def gravity_scores_to_metric(ui: np.ndarray, cv: np.ndarray, uv: np.ndarray, ci: np.ndarray):
    return _scores_to_metric(real0=ui, real1=cv, magic0=uv, magic1=ci)


def gravity_scores_permutation_test(
    ui: np.ndarray,
    cv: np.ndarray,
    uv: np.ndarray,
    ci: np.ndarray,
    min_expected_max: Tuple[float, float, float],
    reps: int = 100,
    seed=0,
):
    return _permutation_test(
        real0=ui,
        real1=cv,
        magic0=uv,
        magic1=ci,
        min_expected_max=min_expected_max,
        reps=reps,
        seed=seed,
    )


def main(
    scores_df_path: str,
    event_category: Literal["continuity", "solidity", "gravity"],
    reps: int,
    score_key: str,
):

    if scores_df_path.endswith(".tsv"):
        model_results_df = pd.read_csv(scores_df_path, sep="\t", header=0)
    elif scores_df_path.endswith(".csv"):
        model_results_df = pd.read_csv(scores_df_path, header=0)
    elif (
        scores_df_path.endswith(".pkl")
        or scores_df_path.endswith(".pickle")
        or scores_df_path.endswith(".pkl.gz")
        or scores_df_path.endswith(".pickle.gz")
    ):
        model_results_df = compress_pickle.load(scores_df_path)
    else:
        raise NotImplementedError("scores_df_path must be a tsv, csv, or pickle file.")

    if event_category == "continuity":
        permutation_test_func = continuity_scores_permutation_test
        expected_skips = 36
    elif event_category == "solidity":
        permutation_test_func = solidity_scores_permutation_test
        expected_skips = 6
    elif event_category == "gravity":
        permutation_test_func = gravity_scores_permutation_test
        expected_skips = 114
    else:
        raise NotImplementedError

    groupby_keys = [
        "camera_loc",
        "cover",
        "obj",
    ]
    if "dir" in model_results_df:
        groupby_keys.append("dir")

    _, metadata_df_dict = split_datasets_using_key(
        datasets=tuple(),
        metadata_df=model_results_df,
        groupby_keys=groupby_keys,
        meta_key="trial_type",
        expected_skips=expected_skips,
    )

    assert len(metadata_df_dict) == 4

    perm_test_result = permutation_test_func(
        reps=reps,
        seed=0,
        min_expected_max=(0.0, 0.5, 1.0),
        **{
            trial_type: np.array(subdf[score_key]) for trial_type, subdf in metadata_df_dict.items()
        },
    )

    print({k: v for k, v in perm_test_result.items() if k not in ["perm_vals", "min_expected_max"]})

    return perm_test_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This script computes metric values and p-values for the InfLevel benchmark.",
    )

    parser.add_argument(
        "--scores_df_path",
        dest="scores_df_path",
        type=str,
        required=True,
        help="The path tsv, csv, or pickle file containing the scores data (see this project's README.md for"
        "an example of how this data should be formatted).",
    )

    parser.add_argument(
        "--event_category",
        dest="event_category",
        type=str,
        choices=["continuity", "solidity", "gravity"],
        required=True,
        help="The event category to compute metrics for, e.g., continuity, solidity, or gravity."
        " This event category should match that used to produce the data in scores_path.",
    )

    parser.add_argument(
        "--reps",
        dest="reps",
        type=int,
        default=1000,
        help="The number of random iterations to use in the permutation test.",
    )

    parser.add_argument(
        "--score_key",
        dest="score_key",
        type=str,
        required=True,
        help="The column in the scores_df (recall `scores_df_path`) that corresponds to your models score."
        " This score should be LARGE if the model believes the video is physically plausible, and SMALL if the model believes"
        " the video is physically implausible.",
    )

    args = parser.parse_args()

    main(
        scores_df_path=args.scores_df_path,
        event_category=args.event_category,
        reps=args.reps,
        score_key=args.score_key,
    )
