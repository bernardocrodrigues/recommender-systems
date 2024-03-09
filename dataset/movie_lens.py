""" movie_lens.py

This module provides functions to download and load the MovieLens dataset.

Copyright 2023 Bernardo C. Rodrigues
See COPYING file for license details
"""

import zipfile
from shutil import copyfileobj
from urllib.request import urlopen
from pathlib import Path
from surprise import Dataset, Reader
from surprise.model_selection import PredefinedKFold, KFold


MOVIELENS_100K_DOWNLOAD_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
MOVIELENS_1M_DOWNLOAD_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
MOVIELENS_URLS = {"100k": MOVIELENS_100K_DOWNLOAD_URL, "1m": MOVIELENS_1M_DOWNLOAD_URL}

OUTPUT_DIR = Path("/tmp/movielens")
MOVIELENS_100K_PATH = OUTPUT_DIR / "ml-100k"
MOVIELENS_1M_PATH = OUTPUT_DIR / "ml-1m"


def download_movielens(destination_dir: Path, dataset: str = "100k"):
    """
    Download the MovieLens 100K or 1M dataset to the specified directory.

    Args:
        destination_dir: The directory where the dataset will be downloaded.
        dataset: The dataset to download. Supported datasets: 100k, 1m.

    Raises:
        AssertionError: If the dataset is not supported.
    """
    assert (
        dataset in MOVIELENS_URLS.keys()
    ), f"Dataset {dataset} not supported. Supported datasets: {MOVIELENS_URLS.keys()}"

    if destination_dir.exists():
        print("Already downloaded!. Nothing to do.")
        return

    destination_dir.mkdir(parents=True, exist_ok=True)

    movielens_zip_file = destination_dir / f"ml-{dataset}.zip"

    if not movielens_zip_file.exists():
        print(f"Downloading MovieLens {dataset} to {destination_dir}...")
        with urlopen(MOVIELENS_URLS[dataset]) as stream, open(movielens_zip_file, "wb") as out_file:
            copyfileobj(stream, out_file)
        print("Done!")

    print("Extracting...")
    with zipfile.ZipFile(movielens_zip_file, "r") as zip_file:
        zip_file.extractall(destination_dir.parent)

    movielens_zip_file.unlink()

    print("Done!")


def load_ml_100k_folds(predefined: bool = True):
    """
    Load the MovieLens 100K dataset and return the folds for cross-validation.

    Args:
        predefined: Whether to load the pre defined folds shipped with the dataset or to generate
            new folds using Surprise's KFold.

    """
    download_movielens(MOVIELENS_100K_PATH, "100k")

    if predefined:
        folds_files = [
            (MOVIELENS_100K_PATH / f"u{i}.base", MOVIELENS_100K_PATH / f"u{i}.test")
            for i in (1, 2, 3, 4, 5)
        ]
        data = Dataset.load_from_folds(folds_files, reader=Reader("ml-100k"))
        k_fold = PredefinedKFold()
    else:
        data = Dataset.load_from_file(str(MOVIELENS_100K_PATH / "u.data"), reader=Reader("ml-100k"))
        k_fold = KFold(n_splits=5)

    return data, k_fold


def load_ml_1m_folds():
    """
    Load the MovieLens 1M dataset and return the folds for cross-validation.

    """
    download_movielens(MOVIELENS_1M_PATH, "1m")

    data = Dataset.load_from_file(str(MOVIELENS_1M_PATH / "ratings.dat"), reader=Reader("ml-1m"))
    k_fold = KFold(n_splits=5)

    return data, k_fold


def resolve_folds(data, k_fold):
    return list((index, fold) for index, fold in enumerate(k_fold.split(data)))
