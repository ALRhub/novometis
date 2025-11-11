import logging

import hydra
import pygame
import torch
from omegaconf import DictConfig

from polymetis import RobotInterface

from torchcontrol.policies.human import HumanControl
from torchcontrol.policies.impedance import InterpolatingHybridImpedanceControl

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="./conf", config_name="teleop")
def main(cfg: DictConfig) -> None:
    fps = cfg.fps
    slowdown_factor = cfg.get("slowdown_factor", 1.0)
    joint_pos_rate_limit = cfg.get("joint_pos_rate_limit") or float("inf")
    joint_vel_rate_limit = cfg.get("joint_vel_rate_limit") or float("inf")
    ee_pos_rate_limit = cfg.get("ee_pos_rate_limit") or float("inf")
    ee_angle_rate_limit = cfg.get("ee_angle_rate_limit") or float("inf")

    # connect to robot arms (we don't need the grippers)
    leader = RobotInterface(
        name=cfg.leader.name,
        ip_address=cfg.leader.ip_address,
        port=cfg.leader.arm_port,
        enforce_version=False,
    )
    log.info(f'Connected to leader "{cfg.leader.name}" at {cfg.leader.ip_address}')

    if cfg.get("go_home", False):
        if (home_pose := cfg.get("home_pose", None)) is not None:
            log.info(f"Setting home pose: {home_pose}")
            leader.set_home_pose(torch.tensor(home_pose))

        leader.go_home()

    leader_policy = HumanControl(leader.robot_model)
    leader.send_torch_policy(leader_policy, blocking=False)

    follower = RobotInterface(
        name=cfg.follower.name,
        ip_address=cfg.follower.ip_address,
        port=cfg.follower.arm_port,
        enforce_version=False,
    )
    log.info(f'Connected to leader "{cfg.follower.name}" at {cfg.follower.ip_address}')

    if cfg.get("go_home", False):
        if (home_pose := cfg.get("home_pose", None)) is not None:
            log.info(f"Setting home pose: {home_pose}")
            follower.set_home_pose(torch.tensor(home_pose))

        follower.go_home()

    follower_policy = InterpolatingHybridImpedanceControl(
        joint_pos_current=follower.get_joint_positions(),
        Kq=follower.Kq_default,
        Kqd=follower.Kqd_default,
        Kx=follower.Kx_default,
        Kxd=follower.Kxd_default,
        robot_model=follower.robot_model,
        ignore_gravity=follower.use_grav_comp,
        update_hz=fps,
        slowdown_factor=slowdown_factor,
        joint_pos_rate_limit=joint_pos_rate_limit,
        joint_vel_rate_limit=joint_vel_rate_limit,
        ee_pos_rate_limit=ee_pos_rate_limit,
        ee_angle_rate_limit=ee_angle_rate_limit,
    )
    follower.send_torch_policy(follower_policy, blocking=False)

    pygame.init()
    clock = pygame.time.Clock()
    running = True

    while running:
        leader_state = leader.get_state()
        follower.update_desired_joint_positions(leader_state.joint_pos)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        clock.tick(fps)

    pygame.quit()


if __name__ == "__main__":
    main()
