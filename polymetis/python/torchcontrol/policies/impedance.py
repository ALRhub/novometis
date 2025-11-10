# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch

import torchcontrol as toco
from torchcontrol.models.torchscript_pinocchio import RobotModelPinocchio
from torchcontrol.transform import Rotation as R
from torchcontrol.transform import Transformation as T
from torchcontrol.types import TensorLike
from torchcontrol.utils.tensor_utils import to_tensor
from torchcontrol.utils.time_utils import timestamp_diff_seconds


class JointImpedanceControl(toco.PolicyModule):
    """
    Impedance control in joint space.
    """

    def __init__(
        self,
        joint_pos_current: TensorLike,
        Kp: TensorLike,
        Kd: TensorLike,
        robot_model: RobotModelPinocchio,
        ignore_gravity: bool = True,
    ):
        """
        Args:
            joint_pos_current: Current joint positions
            Kp: P gains in joint space
            Kd: D gains in joint space
            robot_model: A robot model from torchcontrol.models
            ignore_gravity: `True` if the robot is already gravity compensated, `False` otherwise
        """
        super().__init__()

        # Initialize modules
        self.robot_model = robot_model
        self.invdyn = toco.modules.feedforward.InverseDynamics(
            self.robot_model, ignore_gravity=ignore_gravity
        )
        self.joint_pd = toco.modules.feedback.JointSpacePD(Kp, Kd)

        # Reference pose
        self.joint_pos_desired = torch.nn.Parameter(to_tensor(joint_pos_current))
        self.joint_vel_desired = torch.zeros_like(self.joint_pos_desired)

    def forward(self, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Args:
            state_dict: A dictionary containing robot states

        Returns:
            A dictionary containing the controller output
        """
        # State extraction
        joint_pos_current = state_dict["joint_positions"]
        joint_vel_current = state_dict["joint_velocities"]

        # Control logic
        torque_feedback = self.joint_pd(
            joint_pos_current,
            joint_vel_current,
            self.joint_pos_desired,
            self.joint_vel_desired,
        )
        torque_feedforward = self.invdyn(
            joint_pos_current, joint_vel_current, torch.zeros_like(joint_pos_current)
        )  # coriolis
        torque_out = torque_feedback + torque_feedforward

        return {"joint_torques": torque_out}


class HybridJointImpedanceControl(toco.PolicyModule):
    """
    Impedance control in joint space, but with both fixed joint gains and adaptive operational space gains.
    """

    def __init__(
        self,
        joint_pos_current: TensorLike,
        Kq: TensorLike,
        Kqd: TensorLike,
        Kx: TensorLike,
        Kxd: TensorLike,
        robot_model: RobotModelPinocchio,
        ignore_gravity: bool = True,
    ):
        """
        Args:
            joint_pos_current: Current joint positions
            Kp: P gains in Cartesian space
            Kd: D gains in Cartesian space
            robot_model: A robot model from torchcontrol.models
            ignore_gravity: `True` if the robot is already gravity compensated, `False` otherwise
        """
        super().__init__()

        # Initialize modules
        self.robot_model = robot_model
        self.invdyn = toco.modules.feedforward.InverseDynamics(
            self.robot_model, ignore_gravity=ignore_gravity
        )
        self.joint_pd = toco.modules.feedback.HybridJointSpacePD(Kq, Kqd, Kx, Kxd)

        # Reference pose
        self.joint_pos_desired = torch.nn.Parameter(to_tensor(joint_pos_current))
        self.joint_vel_desired = torch.zeros_like(self.joint_pos_desired)

        self.ema_decay = 1
        self.joint_pos_desired_ema = torch.clone(self.joint_pos_desired)
        self.joint_vel_desired_ema = torch.clone(self.joint_vel_desired)

        self.max_desired_pos_rate_norm = 3.14 / 4  # float("inf")
        self.max_desired_vel_rate_norm = float("inf")
        self.max_desired_ee_pos_rate_norm = 0.05  # float("inf")
        self.max_desired_ee_ang_rate = 3.14 / 4  # float("inf")
        self.limiting_scales = torch.ones(
            (4,), dtype=torch.float32, device=self.joint_pos_desired.device
        )

        self.joint_pos_desired_limited = torch.clone(self.joint_pos_desired)
        self.joint_vel_desired_limited = torch.clone(self.joint_vel_desired)
        self.last_timestamp = torch.zeros(
            (2,), dtype=torch.int32, device=self.joint_pos_desired.device
        )

    def forward(self, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Args:
            state_dict: A dictionary containing robot states

        Returns:
            A dictionary containing the controller output
        """
        self.joint_pos_desired_ema += self.ema_decay * (
            self.joint_pos_desired - self.joint_pos_desired_ema
        )
        self.joint_vel_desired_ema += self.ema_decay * (
            self.joint_vel_desired - self.joint_vel_desired_ema
        )

        now_timestamp = state_dict["timestamp"]
        if self.last_timestamp.sum() == 0:
            # after init, don't allow arbitrarily much since a lot of time has past but clamp first step
            # need to use 1e-6 time here, as zero causes issues with unbounded case as 0 * inf = nan
            secs_since_last = 1e-6 * torch.ones(
                (), dtype=torch.float32, device=self.last_timestamp.device
            )
        else:
            secs_since_last = timestamp_diff_seconds(now_timestamp, self.last_timestamp)
        # secs_since_last = timestamp_diff_seconds(now_timestamp, self.last_timestamp)
        self.last_timestamp.copy_(now_timestamp)

        pos_change_limit = secs_since_last * self.max_desired_pos_rate_norm
        vel_change_limit = secs_since_last * self.max_desired_vel_rate_norm

        target_delta_joint_pos = (
            self.joint_pos_desired_ema - self.joint_pos_desired_limited
        )
        target_delta_joint_vel = (
            self.joint_vel_desired_ema - self.joint_vel_desired_limited
        )
        self.limiting_scales[0] = clamp_norm(
            target_delta_joint_pos,
            pos_change_limit,
        )

        self.limiting_scales[1] = clamp_norm(
            target_delta_joint_vel,
            vel_change_limit,
        )

        ee_pos_change_limit = secs_since_last * self.max_desired_ee_pos_rate_norm
        ee_angle_change_limit = secs_since_last * self.max_desired_ee_ang_rate

        ee_pos_desired_ema, ee_quat_desired_ema = self.robot_model.forward_kinematics(
            self.joint_pos_desired_ema
        )

        ee_pos_desired_limited, ee_quat_desired_limited = (
            self.robot_model.forward_kinematics(self.joint_pos_desired_limited)
        )

        target_delta_ee_pos = ee_pos_desired_ema - ee_pos_desired_limited
        target_delta_ee_angle = rel_quaternion_angle(
            ee_quat_desired_limited, ee_quat_desired_ema
        )

        self.limiting_scales[2] = clamp_norm(
            target_delta_ee_pos,
            ee_pos_change_limit,
        )
        self.limiting_scales[3] = clamp_norm(
            target_delta_ee_angle,
            ee_angle_change_limit,
        )

        target_delta_scale = self.limiting_scales.min()

        self.joint_pos_desired_limited += target_delta_scale * target_delta_joint_pos
        self.joint_vel_desired_limited += target_delta_scale * target_delta_joint_vel

        # change of position target is also a form of velocity target.
        # E.g. if we are supposed to hold a constant velocity but then change the position
        # then we really should target a changed velocity
        joint_pos_rate = (target_delta_scale * target_delta_joint_pos) / secs_since_last

        # State extraction
        joint_pos_current = state_dict["joint_positions"]
        joint_vel_current = state_dict["joint_velocities"]

        # Control logic
        torque_feedback = self.joint_pd(
            joint_pos_current,
            joint_vel_current,
            self.joint_pos_desired_limited,
            self.joint_vel_desired_limited + joint_pos_rate,
            self.robot_model.compute_jacobian(joint_pos_current),
        )
        torque_feedforward = self.invdyn(
            joint_pos_current, joint_vel_current, torch.zeros_like(joint_pos_current)
        )  # coriolis
        torque_out = torque_feedback + torque_feedforward

        return {"joint_torques": torque_out}


class CartesianImpedanceControl(toco.PolicyModule):
    """
    Performs impedance control in Cartesian space.
    Errors and feedback are computed in Cartesian space, and the resulting forces are projected back into joint space.
    """

    def __init__(
        self,
        joint_pos_current: TensorLike,
        Kp: TensorLike,
        Kd: TensorLike,
        robot_model: RobotModelPinocchio,
        ignore_gravity: bool = True,
    ):
        """
        Args:
            joint_pos_current: Current joint positions
            Kp: P gains in Cartesian space
            Kd: D gains in Cartesian space
            robot_model: A robot model from torchcontrol.models
            ignore_gravity: `True` if the robot is already gravity compensated, `False` otherwise
        """
        super().__init__()

        # Initialize modules
        self.robot_model = robot_model
        self.invdyn = toco.modules.feedforward.InverseDynamics(
            self.robot_model, ignore_gravity=ignore_gravity
        )
        self.pose_pd = toco.modules.feedback.CartesianSpacePDFast(Kp, Kd)

        # Reference pose
        joint_pos_current = to_tensor(joint_pos_current)
        ee_pos_current, ee_quat_current = self.robot_model.forward_kinematics(
            joint_pos_current
        )
        self.ee_pos_desired = torch.nn.Parameter(ee_pos_current)
        self.ee_quat_desired = torch.nn.Parameter(ee_quat_current)
        self.ee_vel_desired = torch.nn.Parameter(torch.zeros(3))
        self.ee_rvel_desired = torch.nn.Parameter(torch.zeros(3))

    def forward(self, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Args:
            state_dict: A dictionary containing robot states

        Returns:
            A dictionary containing the controller output
        """
        # State extraction
        joint_pos_current = state_dict["joint_positions"]
        joint_vel_current = state_dict["joint_velocities"]

        # Control logic
        ee_pos_current, ee_quat_current = self.robot_model.forward_kinematics(
            joint_pos_current
        )
        jacobian = self.robot_model.compute_jacobian(joint_pos_current)
        ee_twist_current = jacobian @ joint_vel_current

        wrench_feedback = self.pose_pd(
            ee_pos_current,
            ee_quat_current,
            ee_twist_current,
            self.ee_pos_desired,
            self.ee_quat_desired,
            torch.cat([self.ee_vel_desired, self.ee_rvel_desired]),
        )
        torque_feedback = jacobian.T @ wrench_feedback

        torque_feedforward = self.invdyn(
            joint_pos_current, joint_vel_current, torch.zeros_like(joint_pos_current)
        )  # coriolis

        torque_out = torque_feedback + torque_feedforward

        return {"joint_torques": torque_out}


def clamp_norm(tensor: torch.Tensor, limit: float):
    norm_factor = limit / (tensor.norm(p=2, dim=-1) + 1e-9)
    # don't scale up, only down
    return norm_factor.clamp(max=1.0)


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
