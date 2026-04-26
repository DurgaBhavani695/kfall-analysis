"""
data/loader.py
==============
Responsible for reading the raw KFall sensor and label files from disk
and assembling them into a single, flat :class:`pandas.DataFrame`.

The KFall dataset is organised as:

.. code-block:: text

    data/raw/
    ├── k_fall.csv                  # activity catalogue (task-code → label)
    ├── sensor_data/
    │   └── <SubjectId>/
    │       └── <SubjectId>T<TaskCode>R0<TrialId>.csv
    └── label_data/
        └── <SubjectId>_labels.xlsx

Usage
-----
>>> from kfall.data.loader import DataLoader
>>> loader = DataLoader()
>>> df = loader.load()          # returns merged DataFrame
>>> loader.save(df)             # persists to data/processed/data.csv
"""

from __future__ import annotations

import glob
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tqdm

from kfall.config import (
    FEATURE_COLUMNS,
    KFALL_CATALOGUE_CSV,
    META_COLUMNS,
    PROCESSED_CSV,
    RAW_DATA_DIR,
    LABEL_DATA_SUBDIR,
    SENSOR_DATA_SUBDIR,
)

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Loads and merges the KFall sensor and label data into a flat DataFrame.

    Parameters
    ----------
    raw_data_dir : Path, optional
        Root directory that contains ``k_fall.csv``, ``sensor_data/``, and
        ``label_data/``.  Defaults to ``data/raw``.
    catalogue_csv : Path, optional
        Path to the activity catalogue CSV.  Defaults to
        ``data/raw/k_fall.csv``.

    Attributes
    ----------
    classes : dict
        Mapping ``task_code -> (task_id, task_code, description, class_label)``
        built from the catalogue CSV.

    Examples
    --------
    >>> loader = DataLoader()
    >>> df = loader.load()
    >>> print(df.shape)
    (N, 14)
    """

    COLUMNS: List[str] = META_COLUMNS + FEATURE_COLUMNS

    def __init__(
        self,
        raw_data_dir: Path = RAW_DATA_DIR,
        catalogue_csv: Path = KFALL_CATALOGUE_CSV,
    ) -> None:
        self.raw_data_dir = Path(raw_data_dir)
        self.catalogue_csv = Path(catalogue_csv)
        self.classes: Dict[int, list] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> pd.DataFrame:
        """
        Parse all label Excel files and join them with the corresponding
        sensor CSVs to produce a flat DataFrame.

        Returns
        -------
        pd.DataFrame
            Columns: SubjectId, TaskId, TaskCode, Description, Class,
            AccX, AccY, AccZ, GyrX, GyrY, GyrZ, EulerX, EulerY, EulerZ
        """
        self._validate_paths()
        self.classes = self._build_class_map()

        label_dir = self.raw_data_dir / LABEL_DATA_SUBDIR
        sensor_dir = self.raw_data_dir / SENSOR_DATA_SUBDIR

        label_files = sorted(glob.glob(str(label_dir / "*.xlsx")))
        if not label_files:
            raise FileNotFoundError(
                f"No .xlsx label files found in {label_dir}. "
                "Please check your data/raw/label_data directory."
            )

        rows: List[list] = []
        for fpath in tqdm.tqdm(label_files, desc="[DataLoader] Loading files"):
            rows.extend(self._process_label_file(fpath, sensor_dir))

        df = pd.DataFrame(rows, columns=self.COLUMNS)
        logger.info("Loaded %d rows across %d files.", len(df), len(label_files))
        return df

    def save(self, df: pd.DataFrame, path: Path = PROCESSED_CSV) -> Path:
        """
        Persist ``df`` to disk as a CSV file.

        Parameters
        ----------
        df : pd.DataFrame
            The merged DataFrame returned by :meth:`load`.
        path : Path, optional
            Destination file path.  Defaults to ``data/processed/data.csv``.

        Returns
        -------
        Path
            Absolute path to the saved file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logger.info("Saved processed data → %s", path)
        return path

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_paths(self) -> None:
        """Raise informative errors if expected directories/files are missing."""
        if not self.raw_data_dir.exists():
            raise FileNotFoundError(
                f"Raw data directory not found: {self.raw_data_dir}\n"
                "Run the download instructions in data/README.md first."
            )
        if not self.catalogue_csv.exists():
            raise FileNotFoundError(
                f"KFall catalogue CSV not found: {self.catalogue_csv}\n"
                "Place k_fall.csv inside data/raw/."
            )

    def _build_class_map(self) -> Dict[int, list]:
        """
        Parse k_fall.csv and build a mapping::

            task_code -> [task_id, task_code, description, class_label]

        Only rows whose ``TaskCode`` column starts with ``'F'`` (fall tasks)
        are included so the model focuses on fall vs. ADL classification.
        """
        catalogue = pd.read_csv(self.catalogue_csv)
        classes: Dict[int, list] = {}
        for row in catalogue.values:
            if str(row[0]).startswith("F"):
                task_id: int = int(row[1])
                task_code: int = int(row[1])
                description: str = str(row[2])
                class_label: int = task_id - 20
                classes[task_id] = [task_id, task_code, description, class_label]
        return classes

    def _process_label_file(
        self, fpath: str, sensor_dir: Path
    ) -> List[list]:
        """
        For a single subject label Excel file, iterate over every trial and
        return the flattened sensor rows.

        Parameters
        ----------
        fpath : str
            Path to the ``<SubjectId>_labels.xlsx`` file.
        sensor_dir : Path
            Root directory containing per-subject sensor CSV sub-folders.

        Returns
        -------
        list of list
            Each inner list is one sensor sample with metadata prepended.
        """
        label_df = pd.read_excel(fpath)
        subject_id: str = os.path.basename(fpath).split("_")[0]
        subject_num: str = subject_id.replace("A", "")

        rows: List[list] = []
        task_code: int = 0
        description: str = ""

        for row in label_df.values:
            # Header rows encode the task code inside parentheses, e.g. "Fall (21)"
            if isinstance(row[0], str):
                task_code = int(row[0].split("(")[-1].strip()[:-1])
                description = str(row[1])
                continue

            if task_code not in self.classes:
                continue

            trial_id = row[2]
            onset = int(row[3])
            impact = int(row[4])

            sensor_path = (
                sensor_dir
                / subject_id
                / f"{subject_num}T{task_code}R0{trial_id}.csv"
            )
            if not sensor_path.exists():
                logger.warning("Sensor file not found, skipping: %s", sensor_path)
                continue

            sensor_df = pd.read_csv(sensor_path)
            window = sensor_df.iloc[onset : impact + 1].values[:, 2:]  # drop ts cols

            class_label = self.classes[task_code][3]
            for sample in window:
                rows.append([subject_num, task_code, task_code, description, class_label, *sample])

        return rows
