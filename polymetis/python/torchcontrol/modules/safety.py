import torch

import torchcontrol as toco
from torchcontrol.models.torchscript_pinocchio import RobotModelPinocchio
from torchcontrol.transform import Rotation as R
from torchcontrol.types import TensorLike
from torchcontrol.utils.tensor_utils import to_tensor
from torchcontrol.utils.time_utils import timestamp_diff_seconds


class UniformScalingRateLimiter(toco.ControlModule):
    def __init__(
        self,
        joint_pos_current: TensorLike,
        robot_model: RobotModelPinocchio,
        joint_pos_rate_limit: TensorLike | float = float("inf"),
        joint_vel_rate_limit: TensorLike | float = float("inf"),
        ee_pos_rate_limit: float = float("inf"),
        ee_angle_rate_limit: float = float("inf"),
    ):
        super().__init__()

        self.robot_model = robot_model

        # rate limits
        self.joint_pos_rate_limit = to_tensor(joint_pos_rate_limit)
        self.joint_vel_rate_limit = to_tensor(joint_vel_rate_limit)
        self.ee_pos_rate_limit = ee_pos_rate_limit
        self.ee_angle_rate_limit = ee_angle_rate_limit

        self.joint_pos_desired_limited = to_tensor(joint_pos_current)
        self.joint_vel_desired_limited = torch.zeros_like(self.joint_pos_desired)
        self.last_timestamp = torch.zeros((2,), dtype=torch.int32)

    def forward(
        self,
        now_timestamp: torch.Tensor,
        joint_pos_desired: torch.Tensor,
        joint_vel_desired: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        secs_since_last = timestamp_diff_seconds(now_timestamp, self.last_timestamp)
        self.last_timestamp.copy_(now_timestamp)

        if secs_since_last > 0.1:
            # on first step after init, last_timestamp is still zeros
            # set secs_since_last to interval corresponding to nominal 1kHz control rate
            secs_since_last.fill_(0.001)

        pos_change_limit = secs_since_last * self.joint_pos_rate_limit
        vel_change_limit = secs_since_last * self.joint_vel_rate_limit

        target_delta_joint_pos = joint_pos_desired - self.joint_pos_desired_limited
        target_delta_joint_vel = joint_vel_desired - self.joint_vel_desired_limited

        # accumulate scale factors from all 4 constraints, then scale all joints
        # equally according to the strongest constraint
        safety_scale = torch.ones(1, dtype=torch.float32)

        # for joint position and velocity, compute a scale factor for each joint
        # and take the minimum
        safety_scale = torch.minimum(
            safety_scale,
            torch.min(pos_change_limit / (target_delta_joint_pos + 1e-9)),
        )
        safety_scale = torch.minimum(
            safety_scale,
            torch.min(vel_change_limit / (target_delta_joint_vel + 1e-9)),
        )

        ee_pos_change_limit = secs_since_last * self.ee_pos_rate_limit
        ee_angle_change_limit = secs_since_last * self.ee_angle_rate_limit

        ee_pos_desired, ee_quat_desired = self.robot_model.forward_kinematics(
            joint_pos_desired
        )

        ee_pos_desired_limited, ee_quat_desired_limited = (
            self.robot_model.forward_kinematics(self.joint_pos_desired_limited)
        )

        target_delta_ee_pos = ee_pos_desired - ee_pos_desired_limited
        target_delta_ee_angle = rel_quaternion_angle(
            ee_quat_desired_limited, ee_quat_desired
        )

        # for ee position and orientation, compute a norm over the delta, then
        # compute the scale factor from the norm
        safety_scale = torch.minimum(
            safety_scale,
            torch.min(
                ee_pos_change_limit
                / (torch.linalg.vector_norm(target_delta_ee_pos) + 1e-9)
            ),
        )
        safety_scale = torch.minimum(
            safety_scale,
            torch.min(
                ee_angle_change_limit
                / (torch.linalg.vector_norm(target_delta_ee_angle) + 1e-9)
            ),
        )

        self.joint_pos_desired_limited.add_(safety_scale * target_delta_joint_pos)
        self.joint_vel_desired_limited.add_(safety_scale * target_delta_joint_vel)

        # change of position target is also a form of velocity target.
        # E.g. if we are supposed to hold a constant velocity but then change the position
        # then we really should target a changed velocity
        target_joint_pos_rate = (
            safety_scale * target_delta_joint_pos
        ) / secs_since_last

        return (
            self.joint_pos_desired_limited,
            self.joint_vel_desired_limited,
            target_joint_pos_rate,
        )


def rel_quaternion_angle(current_q, target_q):
    current_q = R.functional.normalize_quaternion(current_q)
    target_q = R.functional.normalize_quaternion(target_q)
    # relative rotation: q_rel = target * inv(current)
    q_rel = R.functional.quaternion_multiply(
        target_q, R.functional.invert_quaternion(current_q)
    )
    q_rel = R.functional.normalize_quaternion(q_rel)
    angle = R.functional.quat2angle(q_rel)  # [...,], in radians, in [0, pi]
    # axis = R.functional.quat2axis(q_rel)  # [...,3]
    return angle
