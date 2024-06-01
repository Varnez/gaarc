import json
from pathlib import Path

import numpy as np


class ARC_dataset:
    def __init__(self, arc_jsons: Path | list[Path]):
        self.tasks: list[dict[str, list]] = []
        self._samples: list | None = None
        self._input_samples: list | None = None
        self._output_samples: list | None = None

        if type(arc_jsons) is Path:
            arc_jsons: list[Path] = [arc_jsons]  # type: ignore[no-redef]

        for arc_json_file in arc_jsons:  # type: ignore[union-attr]
            with open(arc_json_file, "r", encoding="utf-8") as file_pointer:
                contents = json.loads(file_pointer.read())

            for task in contents:
                processed_task: dict[str, list] = {"input": [], "ouput": []}

                for input_sample in task["input"]:
                    processed_task["input"].append(np.array(input_sample))

                for output_sample in task["output"]:
                    processed_task["output"].append(np.array(output_sample))

            self.tasks.append(processed_task)

    @property
    def samples(self) -> list[np.ndarray]:
        if self._samples is None:
            samples = []

            for task in self.tasks:
                samples.extend(task["input"])
                samples.extend(task["output"])

            self._samples = samples

        return self._samples

    @property
    def input_samples(self) -> list[np.ndarray]:
        if self._input_samples is None:
            input_samples = []

            for task in self.tasks:
                input_samples.extend(task["input"])

            self._input_samples = input_samples

        return self._input_samples

    @property
    def output_samples(self) -> list[np.ndarray]:
        if self._output_samples is None:
            output_samples = []

            for task in self.tasks:
                output_samples.extend(task["output"])

            self._output_samples = output_samples

        return self._output_samples
