import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from gaarc.data.augmentation import DataAugmentationTransformation
from gaarc.data.preprocessing import padd_image


class ARCAutoencoderDataset(Dataset):
    """
    Class to load a group of jsons containing ARC samples as numpy arrays, along as holding their
    metadata information, such as the kind of role of each sample or subset it belongs to.

    Useful to build upon any data management functionality, as well as doing data exploration.

    Also acts as a toch Dataset, in which case it can use to iterate over the samples in a training
    process.

    In this effect, transformations such as padding are made available to treat the images.

    Lastly, the possibility to avoid data caches is to be able to use have a non-training oriented
    mode, so that if processing time is not of the essence because, for example, the class is
    being used for data exploration and it's desired not to exploit memory as much, so that this
    system can be removed. This flag should not be set to False if the class is being loaded for
    training, since it will slow down the training procedure by an order of magnitude.

    Ideally, it's expected to be initialized with either train or eval jsons or even both subsets
    at the same time.
    """

    def __init__(
        self,
        arc_task_jsons: list[Path],
        padding_shape: tuple[int, int] | None = None,
        augmentation_transformations: (
            list[DataAugmentationTransformation] | None
        ) = None,
        cache_data_views: bool = True,
    ):
        self._samples: list[np.ndarray] | None = None
        self._train_samples: list[np.ndarray] | None = None
        self._test_samples: list[np.ndarray] | None = None
        self._padding_height: int | None = None
        self._padding_width: int | None = None
        self._augment_data: bool = True
        self._augmentation_transformations: (
            list[DataAugmentationTransformation] | None
        ) = augmentation_transformations

        self._cache_data_views: bool = cache_data_views

        if padding_shape is not None:
            self._padding_height = padding_shape[0]
            self._padding_width = padding_shape[1]

        samples: list[np.ndarray] = []
        sample_roles: list[str] = []
        example_ids: list[int] = []
        task_names: list[str] = []
        task_subsets: list[str] = []

        for arc_task_json in arc_task_jsons:
            task_name: str = arc_task_json.stem

            with open(arc_task_json, "r", encoding="utf-8") as file_pointer:
                task_contents = json.loads(file_pointer.read())

                for task_subset in task_contents:
                    for example_id, examples in enumerate(task_contents[task_subset]):
                        for sample_role in examples:
                            sample = examples[sample_role]
                            sample = np.array(sample)

                            samples.append(sample)
                            sample_roles.append(sample_role)
                            example_ids.append(example_id)
                            task_names.append(task_name)
                            task_subsets.append(task_subset)

        self._contents = pd.DataFrame(
            {
                "sample": samples,
                "role": sample_roles,
                "task_name": task_names,
                "subset": task_subsets,
                "task_example_id": example_ids,
            }
        )

    @property
    def contents(self) -> pd.DataFrame:
        return self._contents

    @property
    def samples(self) -> list[np.ndarray]:
        if self._samples is not None:
            samples = self._samples

        else:
            samples = self._contents["sample"].to_list()

        if self._cache_data_views:
            self._samples = samples

        return samples

    @property
    def train_samples(self) -> list[np.ndarray]:
        if self._train_samples is not None:
            train_samples = self._train_samples

        else:
            train_samples = self._contents[self._contents["role"] == "train"][
                "sample"
            ].to_list()

        if self._cache_data_views:
            self._train_samples = train_samples

        return train_samples

    @property
    def test_samples(self) -> list[np.ndarray]:
        if self._test_samples is not None:
            test_samples = self._test_samples

        else:
            test_samples = self._contents[self._contents["role"] == "test"][
                "sample"
            ].to_list()

        if self._cache_data_views:
            self._train_samples = test_samples

        return test_samples

    def activate_data_augmentation(self) -> None:
        self._augment_data = True

    def deactivate_data_augmentation(self) -> None:
        self._augment_data = False

    def __len__(self):
        return len(self._contents)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        if (
            self._augmentation_transformations is not None
            and self._augment_data is True
        ):
            for agumentation_transformation in self._augmentation_transformations:
                sample = agumentation_transformation.transform(sample)

        padded_sample, padding = padd_image(
            sample, self._padding_height, self._padding_width, -1
        )

        padded_sample = torch.tensor(padded_sample, dtype=torch.float).unsqueeze(0)

        return padded_sample, torch.tensor(padding, dtype=torch.int)
