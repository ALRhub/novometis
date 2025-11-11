import torch

import torchcontrol as toco
from torchcontrol.models.torchscript_pinocchio import RobotModelPinocchio
from torchcontrol.types import TensorLike
from torchcontrol.utils import to_tensor


class HumanControl(toco.PolicyModule):
    """Robot is controlled by a human physically moving the robot through
    space. Externally produced torques are amplified to make it easier to move
    the robot.

    Args:
        robot_model: A robot model from torchcontrol.models
        torque_gain: gains for external torque on each joint
        joint_limit_avoidance_torque: `True` if an additional torque should be
            applied to push the robot away from joint limits
        ignore_gravity: `True` if the robot is already gravity compensated,
            `False` otherwise
    """

    def __init__(
        self,
        robot_model: RobotModelPinocchio,
        torque_gain: TensorLike | None = None,
        joint_limit_avoidance_torque: bool = False,
        ignore_gravity: bool = True,
    ):
        super().__init__()

        # Initialize modules
        self.robot_model = robot_model
        self.invdyn = toco.modules.feedforward.InverseDynamics(
            self.robot_model, ignore_gravity=ignore_gravity
        )

        # define gain
        if torque_gain is None:
            # TODO: put this default value somewhere else
            self.torque_gain = torch.Tensor([0.3, 0.12, 0.40, 1.11, 1.10, 0.6, 0.85])
        else:
            self.torque_gain = to_tensor(torque_gain)

        # Get joint limits for joint limit avoidance torque
        limits = robot_model.get_joint_angle_limits()
        self.joint_pos_min = limits[0]
        self.joint_pos_max = limits[1]

        if joint_limit_avoidance_torque:
            # TODO: put this default value somewhere else
            self.joint_limit_torque_factor = torch.Tensor(
                [5.0, 2.2, 1.3, 0.3, 0.1, 0.1, 0.0]
            )
        else:
            self.joint_limit_torque_factor = torch.Tensor(
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            )

    def forward(self, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # State extraction
        joint_pos_current = state_dict["joint_positions"]
        joint_vel_current = state_dict["joint_velocities"]
        external_torque = state_dict["motor_torques_external"]

        # Amplify external torque applied by human
        human_torque = -self.torque_gain * external_torque

        torque_feedforward = self.invdyn(
            joint_pos_current, joint_vel_current, torch.zeros_like(joint_pos_current)
        )  # coriolis

        # Compute additional torque to push away from joint limits
        # TODO: test this
        left_boundary = 1 / torch.clamp(
            torch.abs(self.joint_pos_min - joint_pos_current), 1e-8, 100000
        )
        right_boundary = 1 / torch.clamp(
            torch.abs(self.joint_pos_max - joint_pos_current), 1e-8, 100000
        )
        torque_joint_limit_avoidance = self.joint_limit_torque_factor * (
            left_boundary - right_boundary
        )

        torque_out = human_torque + torque_feedforward + torque_joint_limit_avoidance

        return {"joint_torques": torque_out}
