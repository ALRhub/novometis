import logging
import sys
import time

import hydra
import pygame
import torch
from omegaconf import DictConfig

from polymetis import GripperInterface

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="./conf", config_name="gripper_free_play")
def main(cfg: DictConfig) -> None:

    gripper = GripperInterface(
        ip_address=cfg.robot.ip_address,
        port=cfg.robot.gripper_port,
    )

    log.info("Gripper initialized")
    log.info(f"Max width: {gripper.metadata.max_width}")

    gripper_speed = cfg.robot.gripper_speed
    gripper_force = cfg.robot.gripper_force
    open_width = cfg.open_width
    closed_width = cfg.closed_width
    grasp_width = cfg.grasp_width
    epsilon_inner = cfg.epsilon_inner
    epsilon_outer = cfg.epsilon_outer
    goto_blocking = cfg.goto_blocking
    grasp_blocking = cfg.grasp_blocking

    pygame.init()
    # Tiny, borderless window so we can capture keys (must be focused at least once)
    pygame.display.set_mode((1, 1), pygame.NOFRAME)
    pygame.display.set_caption("Gripper Control (press Esc to quit)")
    pygame.event.set_allowed([pygame.QUIT, pygame.KEYDOWN])

    clock = pygame.time.Clock()
    running = True
    fps = cfg.fps

    state = gripper.get_state()
    now_ms = state.timestamp.ToMilliseconds()

    while running:
        # Always get and print the current state
        state = gripper.get_state()
        next_ms = state.timestamp.ToMilliseconds()
        elapsed_ms = next_ms - now_ms

        log.info(
            f"elapsed (ms): {elapsed_ms:05.1f}, "
            f"width: {state.width:06.4f}, "
            f"moving: {state.is_moving!s:>5}, "
            f"grasped: {state.is_grasped!s:>5}, "
            f"prev_success: {state.prev_command_successful!s:>5}"
        )

        now_ms = next_ms

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    if goto_blocking:
                        start = time.perf_counter()
                    gripper.goto(
                        width=open_width,
                        speed=gripper_speed,
                        force=gripper_force,
                        blocking=goto_blocking,
                    )
                    if goto_blocking:
                        elapsed = time.perf_counter() - start
                        log.info(
                            f"Gripper moved to open width in {elapsed:.2f} seconds"
                        )
                elif event.key == pygame.K_DOWN:
                    if goto_blocking:
                        start = time.perf_counter()
                    gripper.goto(
                        width=closed_width,
                        speed=gripper_speed,
                        force=gripper_force,
                        blocking=goto_blocking,
                    )
                    if goto_blocking:
                        elapsed = time.perf_counter() - start
                        log.info(
                            f"Gripper moved to closed width in {elapsed:.2f} seconds"
                        )
                elif event.key == pygame.K_SPACE:
                    if grasp_blocking:
                        start = time.perf_counter()
                    gripper.grasp(
                        speed=gripper_speed,
                        force=gripper_force,
                        grasp_width=grasp_width,
                        epsilon_inner=epsilon_inner,
                        epsilon_outer=epsilon_outer,
                        blocking=grasp_blocking,
                    )
                    if grasp_blocking:
                        elapsed = time.perf_counter() - start
                        log.info(f"Gripper grasped object in {elapsed:.2f} seconds")

        clock.tick(fps)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
