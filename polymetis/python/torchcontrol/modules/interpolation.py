import torch

import torchcontrol as toco
from torchcontrol.planning.min_jerk import interpolate_quintic_min_jerk_zero_vT
from torchcontrol.types import TensorLike
from torchcontrol.utils.tensor_utils import to_tensor
from torchcontrol.utils.time_utils import timestamp_diff_seconds


class MinJerkInterpolation(toco.ControlModule):
    def __init__(
        self,
        pos_current: TensorLike,
        update_hz: float,
        slowdown_factor: float = 1.0,
    ):
        super().__init__()

        # Starting position from which to interpolate to desired position
        pos_current = to_tensor(pos_current)
        self.pos_init = torch.zeros_like(pos_current)
        self.vel_init = torch.zeros_like(pos_current)
        self.time_init = torch.zeros((2,), dtype=torch.int32)

        # duration of each waypoint
        self.T = torch.tensor(1 / update_hz)

        # increase the duration of each waypoint to add a margin of safety for
        # updates from the client, in case they don't match the nominal rate
        self.T *= slowdown_factor

    @torch.jit.export
    def reset(
        self,
        time_current: torch.Tensor,
        pos_current: torch.Tensor,
        vel_current: torch.Tensor,
    ) -> None:
        # Update initial conditions to current state
        self.pos_init[:] = pos_current
        self.vel_init[:] = vel_current

        # reset start time to current time
        self.time_init.copy_(time_current)

    def forward(
        self, now_timestamp: torch.Tensor, pos_desired: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.time_init.sum() == 0:
            # in this case, reset was never called, which means the output may
            # be nonsense or unsafe
            # just return the desired position unmodified and zero velocity
            return pos_desired, torch.zeros_like(pos_desired)

        time = timestamp_diff_seconds(now_timestamp, self.time_init)

        # clamp time to [0, T]
        time.clamp_max_(self.T)

        # compute desired joint positions and velocities by interpolation
        pos_desired, vel_desired = interpolate_quintic_min_jerk_zero_vT(
            time,
            self.pos_init,
            pos_desired,
            self.vel_init,
            self.T,
        )

        return pos_desired, vel_desired
