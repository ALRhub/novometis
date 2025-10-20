# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import math

import torch

import torchcontrol as toco
from torchcontrol.models.torchscript_pinocchio import RobotModelPinocchio
from torchcontrol.transform import Rotation as R
from torchcontrol.transform import Transformation as T
from torchcontrol.types import TensorLike
from torchcontrol.utils.tensor_utils import diagonalize_gain, to_tensor
from torchcontrol.utils.time_utils import timestamp_diff_ms, timestamp_diff_seconds


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
        # State extraction
        joint_pos_current = state_dict["joint_positions"]
        joint_vel_current = state_dict["joint_velocities"]

        # Control logic
        torque_feedback = self.joint_pd(
            joint_pos_current,
            joint_vel_current,
            self.joint_pos_desired_ema,
            self.joint_vel_desired_ema,
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

        self.ema_decay = 1
        self.ee_pos_desired_ema = torch.clone(self.ee_pos_desired)
        self.ee_quat_desired_ema = torch.clone(self.ee_quat_desired)

    def forward(self, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Args:
            state_dict: A dictionary containing robot states

        Returns:
            A dictionary containing the controller output
        """
        self.ee_pos_desired_ema += self.ema_decay * (
            self.ee_pos_desired - self.ee_pos_desired_ema
        )
        self.ee_quat_desired_ema += self.ema_decay * (
            self.ee_quat_desired - self.ee_quat_desired_ema
        )
        # lerp + norm is close enough, don't need slerp for small angles here
        self.ee_quat_desired_ema = R.functional.normalize_quaternion(
            self.ee_quat_desired_ema
        )

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
            self.ee_pos_desired_ema,
            self.ee_quat_desired_ema,
            torch.cat([self.ee_vel_desired, self.ee_rvel_desired]),
        )
        torque_feedback = jacobian.T @ wrench_feedback

        torque_feedforward = self.invdyn(
            joint_pos_current, joint_vel_current, torch.zeros_like(joint_pos_current)
        )  # coriolis

        torque_out = torque_feedback + torque_feedforward

        return {"joint_torques": torque_out}


class AdvancedCartesianImpedanceControl(toco.PolicyModule):
    """
    Advanced impedance controller:
    - Takes joint configuration target
    - Converts to EE pose via forward kinematics
    - Applies EMA to target EE pose
    - Runs PD in task space, clips position error, transforms wrench to joint torques
    - Adds damping torque in null-space
    - Adds workspace limit forces to wrench
    - Adds nullspace torque for singularity avoidance
    - Adds feedforward torques for dynamics

    Args:
        joint_pos_current: Current joint positions (Tensor)
        joint_pos_target: Target joint positions (Tensor)
        Kp: Cartesian P gain (Tensor)
        Kd: Cartesian D gain (Tensor)
        K_null: Nullspace damping gain (Tensor)
        K_sing: Singularity avoidance gain (Tensor)
        robot_model: RobotModelPinocchio
        workspace_limits: dict of workspace limits
        ignore_gravity: bool
        ema_decay: float
    """

    workspace_box_limits: list[tuple[int, tuple[float, float]]]

    def __init__(
        self,
        joint_pos_current,
        Kp,
        Kd,
        K_null,
        K_sing,
        robot_model,
        ignore_gravity=True,
    ):
        super().__init__()
        self.robot_model = robot_model
        self.invdyn = toco.modules.feedforward.InverseDynamics(
            self.robot_model, ignore_gravity=ignore_gravity
        )
        self.pose_pd = toco.modules.feedback.CartesianSpacePDFast(Kp, Kd)
        self.K_null = torch.nn.Parameter(diagonalize_gain(to_tensor(K_null)))
        self.K_sing = torch.nn.Parameter(diagonalize_gain(to_tensor(K_sing)))
        self.workspace_box_limits: list[
            tuple[int, tuple[float, float]]
        ] = [] # [(0, (0.3, 0.31))]
        self.ema_decay = 1.0
        # Clamping limits
        self.pos_error_limit = float("inf")
        self.quat_radian_limit = float("inf")
        self.torque_rate_limit = 1000.0 # Nm/s

        # Target joint config
        self.joint_pos_desired = torch.nn.Parameter(to_tensor(joint_pos_current))

        # Initial EE pose
        joint_pos_current = to_tensor(joint_pos_current)
        ee_pos_target, ee_quat_target = self.robot_model.forward_kinematics(
            joint_pos_current
        )
        self.ee_pos_target = torch.nn.Parameter(ee_pos_target)
        self.ee_quat_target = torch.nn.Parameter(ee_quat_target)
        self.ee_pos_target_ema = torch.clone(self.ee_pos_target)
        self.ee_quat_target_ema = torch.clone(self.ee_quat_target)

        self.last_torque = torch.clone(self.joint_pos_desired)
        self.last_timestamp = torch.zeros((2,), dtype=torch.int32, device=self.joint_pos_desired.device)


    def forward(self, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        joint_pos_current = state_dict["joint_positions"]
        joint_vel_current = state_dict["joint_velocities"]

        # 1. Convert joint target to EE pose
        # TODO: do only once in/after update()
        ee_pos_target, ee_quat_target = self.robot_model.forward_kinematics(
            self.joint_pos_desired
        )
        self.ee_pos_target.data.copy_(ee_pos_target)
        self.ee_quat_target.data.copy_(ee_quat_target)

        # 2. EMA on target EE pose
        self.ee_pos_target_ema += self.ema_decay * (
            self.ee_pos_target - self.ee_pos_target_ema
        )
        # TODO slerp
        self.ee_quat_target_ema += self.ema_decay * (
            self.ee_quat_target - self.ee_quat_target_ema
        )
        self.ee_quat_target_ema = R.functional.normalize_quaternion(
            self.ee_quat_target_ema
        )

        # 3. Get current EE pose and jacobian
        ee_pos_current, ee_quat_current = self.robot_model.forward_kinematics(
            joint_pos_current
        )
        jacobian = self.robot_model.compute_jacobian(joint_pos_current)
        ee_twist_current = jacobian @ joint_vel_current

        # 4. Clip position error before PD
        pos_error = self.ee_pos_target_ema - ee_pos_current
        pos_error_clipped = clamp_elementwise(pos_error, self.pos_error_limit)

        # Compose desired pose for PD
        ee_pos_desired = ee_pos_current + pos_error_clipped
        # ee_quat_desired = clamp_quaternion(ee_quat_current, self.ee_quat_target_ema, self.quat_radian_limit)
        ee_quat_desired = self.ee_quat_target_ema

        # 5. PD controller in task space
        wrench_feedback = self.pose_pd(
            ee_pos_current,
            ee_quat_current,
            ee_twist_current,
            ee_pos_desired,
            ee_quat_desired,
            torch.zeros(6, device=ee_pos_current.device),
        )

        # 6. Workspace limit forces (simple box constraints)
        # TODO: project pos/quat desired into workspace instead of this shit.
        for i, (low, high) in self.workspace_box_limits:
            if ee_pos_current[i] < low:
                wrench_feedback[i] += 1000.0 * (low - ee_pos_current[i])
            elif ee_pos_current[i] > high:
                wrench_feedback[i] += 1000.0 * (high - ee_pos_current[i])

        # 7. Transform wrench to joint torques (Jacobian transpose)
        torque_task = jacobian.T @ wrench_feedback

        # 8. Null-space damping torque
        # Project joint velocities into nullspace of Jacobian
        # J_pinv = torch.linalg.pinv(jacobian)
        # nullspace_projector = (
        #     torch.eye(joint_pos_current.shape[0], device=joint_pos_current.device)
        #     - J_pinv @ jacobian
        # )
        # torque_null_damping = 50.0 * self.K_null @ (nullspace_projector @ joint_vel_current)
        # torque_null_damping = self.K_null @ joint_vel_current

        # 9. Singularity avoidance nullspace torque
        # grad_sing = self.singularity_avoidance_gradient(joint_pos_current)
        # torque_null_sing = self.K_sing @ (nullspace_projector @ grad_sing)

        # 10. Feedforward torque
        torque_ff = self.invdyn(
            joint_pos_current, joint_vel_current, torch.zeros_like(joint_pos_current)
        )

        # Total torque
        # torque_out = torque_task - torque_null_damping + torque_null_sing + torque_ff
        torque_out = torque_task + torque_ff

        # 11. Torque rate limiting
        # Get time and time step
        now_timestamp = state_dict["timestamp"]
        secs_since_last = timestamp_diff_seconds(now_timestamp, self.last_timestamp)
        self.last_timestamp.copy_(now_timestamp)

        # Friction dithering torque
        time_now_secs = 1e-9 * now_timestamp[1].float()
        # dither frequence must be integer Hz such that we can just ignore the seconds part of the time.
        dither_f_Hz = int(10)
        dither_torque = 0 * torch.sin(2 * math.pi * dither_f_Hz * time_now_secs)
        torque_out += jacobian.T @ torch.ones_like(wrench_feedback) * dither_torque

        # limit rates
        # torque_out = self.last_torque + clamp_elementwise(torque_out-self.last_torque, self.torque_rate_limit * secs_since_last)
        # self.last_torque.copy_(torque_out)

        return {"joint_torques": torque_out}

    def singularity_avoidance_gradient(self, joint_pos):
        """
        Placeholder for singularity avoidance gradient.
        Should return a vector in joint space pointing away from singularities.
        """
        # TODO: Implement actual singularity avoidance logic
        return torch.zeros_like(joint_pos)


def clamp_elementwise(tensor: torch.Tensor, limit: float):
    return tensor.clamp(-limit, limit)


def clamp_norm(tensor: torch.Tensor, limit: float):
    norm_factor = limit / (tensor.norm(p=2, dim=-1) + 1e-9)
    # don't scale up, only down
    clamped_norm_factor = norm_factor.clamp(max=1.0)
    return tensor * clamped_norm_factor


def _slerp(q0, q1, t):
    # TODO: check the slop
    # q0, q1: [...,4], normalized. t: scalar or [...,1]
    q0 = R.functional.normalize_quaternion(q0)
    q1 = R.functional.normalize_quaternion(q1)
    # compute dot
    dot = (q0 * q1).sum(dim=-1, keepdim=True)  # [...,1]
    # take shortest path
    q1b = q1.clone()
    neg_mask = dot < 0.0
    if neg_mask.any():
        q1b[neg_mask.expand_as(q1b)] = -q1b[neg_mask.expand_as(q1b)]
        dot = (q0 * q1b).sum(dim=-1, keepdim=True)
    dot_clamped = dot.clamp(-1.0, 1.0)
    theta = torch.acos(dot_clamped)  # [...,1]
    sin_theta = torch.sin(theta)
    # handle small angle -> lerp
    small_mask = (sin_theta.abs() < 1e-6).squeeze(-1)
    out = torch.empty_like(q0)
    # normalized lerp branch
    if small_mask.any():
        t_exp = t.expand_as(q0)[small_mask]
        lerp = (1.0 - t_exp) * q0[small_mask] + t_exp * q1b[small_mask]
        out[small_mask] = R.functional.normalize_quaternion(lerp)
    # slerp branch
    if (~small_mask).any():
        idx = ~small_mask
        theta_ns = theta[idx]  # [...,1]
        sin_theta_ns = sin_theta[idx]
        t_ns = t.expand_as(dot)[idx]  # [...,1]
        s0 = torch.sin((1.0 - t_ns) * theta_ns) / sin_theta_ns
        s1 = torch.sin(t_ns * theta_ns) / sin_theta_ns
        s0 = s0.unsqueeze(-1)
        s1 = s1.unsqueeze(-1)
        out[idx] = s0 * q0[idx] + s1 * q1b[idx]
    return R.functional.normalize_quaternion(out)


def clamp_quaternion(current_q, target_q, max_angle_rad, EPS=1e-7):
    """
    Returns a quaternion target_clamped that is at most max_angle_rad away from current_q,
    moving from current_q toward target_q. If angle <= max_angle_rad, returns normalized target_q.
    Inputs:
      current_q, target_q: [...,4] (w,x,y,z)
      max_angle_rad: scalar float
    Output:
      new_target_q: [...,4] normalized
    """
    current_q = R.functional.normalize_quaternion(current_q)
    target_q = R.functional.normalize_quaternion(target_q)
    # relative rotation: q_rel = target * inv(current)
    q_rel = R.functional.quaternion_multiply(
        target_q, R.functional.invert_quaternion(current_q)
    )
    q_rel = R.functional.normalize_quaternion(q_rel)
    angle = R.functional.quat2angle(q_rel)  # [...,], in radians, in [0, pi]
    # if angle <= max -> keep target
    keep_mask = angle <= max_angle_rad
    if keep_mask.dim() == 0:
        if keep_mask:
            return target_q
    else:
        if keep_mask.all():
            return target_q

    # For rotations that exceed threshold, build a rotation of max_angle around same axis
    # Handle near-zero rotation: return target (nothing to do)
    too_small = angle < EPS
    if too_small.dim() == 0:
        if too_small:
            return target_q
    else:
        # where angle is near zero, treat as keep
        keep_mask = keep_mask | too_small
        if keep_mask.all():
            return target_q

    # rotation axis from q_rel; quat2axis should yield unit axis [...,3]
    axis = R.functional.quat2axis(q_rel)  # [...,3]
    # Build desired relative quaternion with angle = max_angle_rad (preserve sign/direction)
    # For safety, preserve rotation direction: axis * sign(angle)
    # If quat2angle returns positive scalar, axis already encodes direction (depending on implementation).
    # We'll create rotvec = axis * max_angle_rad and convert to quaternion via exp map:
    # Use quat2rotvec inverse via constructing: q = [cos(a/2), axis*sin(a/2)]
    half = 0.5 * max_angle_rad
    sin_half = math.sin(half)
    cos_half = math.cos(half)
    # axis may be [...,3]; expand to quaternion [...,4]
    rot_q = torch.zeros_like(current_q)
    rot_q[..., 0] = cos_half
    rot_q[..., 1:] = axis * sin_half

    # Now new_target = current * rot_q
    new_target = R.functional.quaternion_multiply(current_q, rot_q)
    new_target = R.functional.normalize_quaternion(new_target)

    # For entries that were within threshold, keep original target
    if keep_mask.dim() == 0:
        return target_q if keep_mask else new_target
    out = target_q.clone()
    out[~keep_mask] = new_target[~keep_mask]
    return R.functional.normalize_quaternion(out)
