"""EMA utilities whose parameter ordering is stable across freezing changes."""

from __future__ import annotations

from collections.abc import Iterable

import torch


class AllParameterExponentialMovingAverage:
    """Track every supplied parameter, including frozen parameters.

    Filtering by ``requires_grad`` is unsafe when parameters are frozen after an
    EMA object has been constructed: the filtered positional order can change
    and silently associate a shadow value with the wrong parameter.
    """

    def __init__(
        self,
        parameters: Iterable[torch.nn.Parameter],
        decay: float,
        use_num_updates: bool = False,
    ) -> None:
        if not 0.0 <= decay <= 1.0:
            raise ValueError("decay must be between 0 and 1")
        parameter_list = list(parameters)
        self.decay = decay
        self.num_updates = 0 if use_num_updates else None
        self.shadow_params = [
            parameter.detach().clone() for parameter in parameter_list
        ]
        self.collected_params: list[torch.Tensor] = []

    def _validated_parameters(
        self,
        parameters: Iterable[torch.nn.Parameter],
    ) -> list[torch.nn.Parameter]:
        parameter_list = list(parameters)
        if len(parameter_list) != len(self.shadow_params):
            raise ValueError(
                "EMA parameter count changed: "
                f"expected {len(self.shadow_params)}, got {len(parameter_list)}"
            )
        for index, (shadow, parameter) in enumerate(
            zip(self.shadow_params, parameter_list)
        ):
            if shadow.shape != parameter.shape:
                raise ValueError(
                    "EMA parameter shape changed at position "
                    f"{index}: expected {tuple(shadow.shape)}, "
                    f"got {tuple(parameter.shape)}"
                )
        return parameter_list

    def move_shadow_params_to_device(self, device: torch.device | str) -> None:
        self.shadow_params = [parameter.to(device) for parameter in self.shadow_params]

    def update(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        parameter_list = self._validated_parameters(parameters)
        decay = self.decay
        if self.num_updates is not None:
            self.num_updates += 1
            decay = min(decay, (1 + self.num_updates) / (10 + self.num_updates))
        one_minus_decay = 1.0 - decay
        with torch.no_grad():
            for shadow, parameter in zip(self.shadow_params, parameter_list):
                shadow.sub_(
                    one_minus_decay
                    * (shadow - parameter.detach().to(device=shadow.device))
                )

    def copy_to(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        parameter_list = self._validated_parameters(parameters)
        with torch.no_grad():
            for shadow, parameter in zip(self.shadow_params, parameter_list):
                parameter.copy_(shadow.to(device=parameter.device))

    def store(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        parameter_list = self._validated_parameters(parameters)
        self.collected_params = [
            parameter.detach().clone() for parameter in parameter_list
        ]

    def restore(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        parameter_list = self._validated_parameters(parameters)
        if len(self.collected_params) != len(parameter_list):
            raise RuntimeError("EMA restore called before a matching store")
        with torch.no_grad():
            for collected, parameter in zip(
                self.collected_params, parameter_list
            ):
                parameter.copy_(collected.to(device=parameter.device))
        self.collected_params = []

    def state_dict(self) -> dict:
        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "shadow_params": self.shadow_params,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        shadow_params = list(state_dict["shadow_params"])
        if len(shadow_params) != len(self.shadow_params):
            raise ValueError(
                "EMA checkpoint parameter count does not match the decoder: "
                f"expected {len(self.shadow_params)}, got {len(shadow_params)}"
            )
        for index, (expected, loaded) in enumerate(
            zip(self.shadow_params, shadow_params)
        ):
            if expected.shape != loaded.shape:
                raise ValueError(
                    "EMA checkpoint shape mismatch at position "
                    f"{index}: expected {tuple(expected.shape)}, "
                    f"got {tuple(loaded.shape)}"
                )
        self.decay = float(state_dict["decay"])
        self.num_updates = state_dict["num_updates"]
        self.shadow_params = [
            loaded.detach().clone().to(device=expected.device)
            for expected, loaded in zip(self.shadow_params, shadow_params)
        ]
