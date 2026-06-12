# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os

from rlinf.envs.metaworld.utils import load_prompt_from_json

_SUPPORTED_SUITES = [
    "metaworld_50",
    "metaworld_45_ind",
    "metaworld_45_ood",
    "metaworld_single",
]


class MetaWorldBenchmark:
    def __init__(self, task_suite_name, task_names=None, task_num_trials=None):
        assert task_suite_name in _SUPPORTED_SUITES, (
            f"Unsupported MetaWorld task suite {task_suite_name!r}. "
            f"Expected one of {_SUPPORTED_SUITES}."
        )
        self.task_suite_name = task_suite_name
        config_path = os.path.join(os.path.dirname(__file__), "metaworld_config.json")
        self.task_description_dict = load_prompt_from_json(
            config_path, "TASK_DESCRIPTIONS"
        )
        self.ML45_dict = load_prompt_from_json(config_path, "ML45")

        if self.task_suite_name == "metaworld_single":
            if not task_names:
                raise ValueError(
                    "metaworld_single requires task_names in env config, "
                    "e.g. task_names: ['push-v3']."
                )
            self._task_names = list(task_names)
            unknown = [
                name
                for name in self._task_names
                if name not in self.task_description_dict
            ]
            if unknown:
                raise ValueError(
                    f"Unknown MetaWorld task names: {unknown}. "
                    f"Valid names are keys in TASK_DESCRIPTIONS."
                )
            self._task_num_trials = 10 if task_num_trials is None else int(
                task_num_trials
            )
        else:
            self._task_names = None
            self._task_num_trials = None

    def get_num_tasks(self):
        if self.task_suite_name == "metaworld_50":
            return 50
        elif self.task_suite_name == "metaworld_45_ind":
            return 45
        elif self.task_suite_name == "metaworld_45_ood":
            return 5
        elif self.task_suite_name == "metaworld_single":
            return len(self._task_names)

    def get_task_num_trials(self):
        if self.task_suite_name == "metaworld_50":
            return 10
        elif self.task_suite_name == "metaworld_45_ind":
            return 10
        elif self.task_suite_name == "metaworld_45_ood":
            return 20
        elif self.task_suite_name == "metaworld_single":
            return self._task_num_trials

    def get_env_names(self):
        if self.task_suite_name == "metaworld_50":
            return list(self.task_description_dict.keys())
        elif self.task_suite_name == "metaworld_45_ind":
            return self.ML45_dict["train"]
        elif self.task_suite_name == "metaworld_45_ood":
            return self.ML45_dict["test"]
        elif self.task_suite_name == "metaworld_single":
            return list(self._task_names)

    def get_task_description(self):
        task_descriptions = []
        for env_name in self.get_env_names():
            task_descriptions.append(self.task_description_dict[env_name])
        return task_descriptions
