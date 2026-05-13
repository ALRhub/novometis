import torch

import torchcontrol as toco
from torchcontrol.planning.min_jerk import (
    interpolate_min_jerk,
    interpolate_min_jerk_zero_vfinal,
)
from torchcontrol.types import TensorLike
from torchcontrol.utils.tensor_utils import to_tensor
from torchcontrol.utils.time_utils import timestamp_diff_seconds, timestamp_subtract


class MinJerkInterpolation(toco.ControlModule):
    def __init__(
        self,
        pos_current: TensorLike,
        update_hz: float,
        slowdown_factor: float = 1.0,
    ):
        super().__init__()

        # starting position, velocity, and time from which to interpolate to
        # desired position
        self.pos_init = to_tensor(pos_current, ensure_copy=True)
        self.vel_init = torch.zeros_like(pos_current)
        self.time_init = torch.zeros((2,), dtype=torch.int32)

        # store the final position
        self.pos_final = to_tensor(pos_current, ensure_copy=True)

        # Store the final velocity of the unconstrained interpolation. This is
        # used as the initial velocity of the backup interpolation to zero
        # velocity.
        self.vel_final = torch.zeros_like(pos_current)

        # store each interpolated position so it can be used as the initial
        # position on reset
        self.last_pos = to_tensor(pos_current, ensure_copy=True)
        self.last_vel = torch.zeros_like(pos_current)

        # duration of each waypoint
        self.T = torch.tensor(1 / update_hz)

        # increase the duration of each waypoint to add a margin of safety for
        # updates from the client, in case they don't match the nominal rate
        self.T *= slowdown_factor

    @torch.jit.export
    def reset(
        self,
        pos_final: torch.Tensor,
        time_current: torch.Tensor,
    ) -> None:
        self.pos_final.copy_(pos_final)

        # reset start time to current time minus 1ms, which is the nominal
        # inference rate of the controller
        interval = torch.tensor([0, 1_000_000])  # 1ms = 1e6 ns
        reset_time = timestamp_subtract(time_current, interval)
        self.time_init.copy_(reset_time)

        # Update initial conditions to the last (interpolated) position and
        # velocity. This prevents discontinuities in the targets, since the
        # PD controller might never reach the targets in steady state.
        self.pos_init.copy_(self.last_pos)
        self.vel_init.copy_(self.last_vel)

        # Compute and store the velocity at the end of the interpolated
        # trajectory. This is used as the initial velocity of the backup
        # interpolation to zero velocity.
        _, vel_final, _ = interpolate_min_jerk(
            self.T,
            self.pos_init,
            self.pos_final,
            self.vel_init,
            self.T,
        )
        self.vel_final.copy_(vel_final)

    def forward(self, now_timestamp: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.time_init.sum() == 0:
            # in this case, reset was never called, which means the output may
            # be nonsense or unsafe
            # just return the desired position unmodified and zero velocity
            return self.pos_final, torch.zeros_like(self.pos_final)

        time = timestamp_diff_seconds(now_timestamp, self.time_init)

        # clamp time to [0, T*1.5]
        time = time.clamp(min=0.0, max=self.T * 1.5)

        if time <= self.T:
            # compute desired joint positions and velocities by interpolation
            pos, vel, _ = interpolate_min_jerk(
                time,
                self.pos_init,
                self.pos_final,
                self.vel_init,
                self.T,
            )
        else:
            # We should have received an updated final position by now, but we
            # haven't. As a backup, interpolate from the final position to the
            # final position with zero target velocity within 0.5 * T, to
            # guarantee a smooth stop.
            pos, vel, _ = interpolate_min_jerk_zero_vfinal(
                time - self.T,
                self.pos_final,
                self.pos_final,
                self.vel_final,
                self.T * 0.5,
            )

        self.last_pos.copy_(pos)
        self.last_vel.copy_(vel)

        return pos, vel
