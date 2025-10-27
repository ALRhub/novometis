import math
from math import atan2, sqrt
from time import sleep
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchcontrol as toco
from polymetis_pb2 import RobotState
from torchcontrol.transform import Rotation as R

from polymetis import RobotInterface


def plot_robot_states_by_joint(robot_states: List["RobotState"]):
    """
    Creates N figures (one per joint). Each figure has 3 subplots stacked vertically:
      1) Joint position (line, no markers)
      2) Joint velocity (line, no markers)
      3) All torque channels for that joint (multiple colored lines)
    Timestamps (protobuf Timestamp) are used if present; otherwise indices are used as time.
    """
    if not robot_states:
        raise ValueError("robot_states list is empty")

    # Extract times in seconds (fallback to index)
    times = []
    for i, rs in enumerate(robot_states):
        ts = getattr(rs, "timestamp", None)
        if (
            ts is not None
            and hasattr(ts, "seconds")
            and (ts.seconds != 0 or getattr(ts, "nanos", 0) != 0)
        ):
            times.append(float(ts.seconds) + float(getattr(ts, "nanos", 0)) * 1e-9)
        else:
            times.append(float(i))
    times = np.array(times)
    times = times - times[0]

    # Helper: longest repeated field length
    def longest_length(field_name):
        return max((len(getattr(rs, field_name)) for rs in robot_states), default=0)

    torque_fields = [
        ("joint_torques_computed", "joint_torques_computed"),
        ("prev_joint_torques_computed", "prev_joint_torques_computed"),
        ("prev_joint_torques_computed_safened", "prev_joint_torques_computed_safened"),
        ("motor_torques_measured", "motor_torques_measured"),
        ("motor_torques_external", "motor_torques_external"),
        ("motor_torques_desired", "motor_torques_desired"),
    ]

    n_pos = longest_length("joint_positions")
    n_vel = longest_length("joint_velocities")
    n_torque = max((longest_length(fname) for fname, _ in torque_fields), default=0)
    n_joints = max(n_pos, n_vel, n_torque)
    if n_joints == 0:
        raise ValueError("No joint data found in robot_states")

    T = len(robot_states)

    def build_array(field_name, n_cols):
        arr = np.full((T, n_cols), np.nan, dtype=float)
        for t, rs in enumerate(robot_states):
            vals = list(getattr(rs, field_name, []))
            for j, v in enumerate(vals[:n_cols]):
                arr[t, j] = float(v)
        return arr

    pos_arr = build_array("joint_positions", n_joints)
    vel_arr = build_array("joint_velocities", n_joints)
    torque_arrs = {fname: build_array(fname, n_joints) for fname, _ in torque_fields}

    # Colors for torque lines
    torque_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for j in range(n_joints):
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        fig.suptitle(f"Joint {j}")

        # Position (line only)
        axes[0].plot(
            times,
            pos_arr[:, j],
            linestyle="-",
            marker=None,
            color="C0",
            label="position",
        )
        axes[0].set_ylabel("Position")
        axes[0].grid(True)
        axes[0].legend()

        # Velocity (line only)
        axes[1].plot(
            times,
            vel_arr[:, j],
            linestyle="-",
            marker=None,
            color="C1",
            label="velocity",
        )
        axes[1].set_ylabel("Velocity")
        axes[1].grid(True)
        axes[1].legend()

        # Torques: multiple lines with different colors
        for idx, (fname, label) in enumerate(torque_fields):
            arr = torque_arrs[fname][:, j]
            color = torque_colors[idx % len(torque_colors)]
            axes[2].plot(
                times, arr, linestyle="-", marker=None, color=color, label=label
            )
        axes[2].set_ylabel("Torque (Nm)")
        axes[2].set_xlabel("Time (s)")
        axes[2].grid(True)
        axes[2].legend(loc="upper right", fontsize="small")

        fig.tight_layout(rect=[0, 0, 1, 0.96])

    plt.show()


def plot_robot_states_grid(robot_states: List["RobotState"]):
    """
    Creates a single figure with grid of subplots:
      - Columns: one per joint (N)
      - Rows: 3 rows per column:
          Row 0: Position (line, no markers)
          Row 1: Velocity (line, no markers)
          Row 2: All torque channels for that joint (multiple colored lines)
    Timestamps (protobuf Timestamp) are used if present; otherwise indices are used as time.
    """
    if not robot_states:
        raise ValueError("robot_states list is empty")

    # Extract times in seconds (fallback to index)
    times = []
    for i, rs in enumerate(robot_states):
        ts = getattr(rs, "timestamp", None)
        if (
            ts is not None
            and hasattr(ts, "seconds")
            and (ts.seconds != 0 or getattr(ts, "nanos", 0) != 0)
        ):
            times.append(float(ts.seconds) + float(getattr(ts, "nanos", 0)) * 1e-9)
        else:
            times.append(float(i))
    times = np.array(times)
    times = times - times[0]

    # Helper: longest repeated field length
    def longest_length(field_name):
        return max((len(getattr(rs, field_name)) for rs in robot_states), default=0)

    torque_field_names = [
        "joint_torques_computed",
        "prev_joint_torques_computed",
        "prev_joint_torques_computed_safened",
        "motor_torques_measured",
        "motor_torques_external",
        "motor_torques_desired",
    ]

    n_pos = longest_length("joint_positions")
    n_vel = longest_length("joint_velocities")
    n_torque = max((longest_length(f) for f in torque_field_names), default=0)
    n_joints = max(n_pos, n_vel, n_torque)
    if n_joints == 0:
        raise ValueError("No joint data found in robot_states")

    T = len(robot_states)

    def build_array(field_name, n_cols):
        arr = np.full((T, n_cols), np.nan, dtype=float)
        for t, rs in enumerate(robot_states):
            vals = list(getattr(rs, field_name, []))
            for j, v in enumerate(vals[:n_cols]):
                arr[t, j] = float(v)
        return arr

    pos_arr = build_array("joint_positions", n_joints)
    vel_arr = build_array("joint_velocities", n_joints)
    torque_arrs = {f: build_array(f, n_joints) for f in torque_field_names}

    # Create grid: 3 rows x n_joints columns
    cols = n_joints
    rows = 3
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows), sharex="col")
    # Ensure axes is 2D array even if cols==1
    if cols == 1:
        axes = axes.reshape(rows, 1)

    torque_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for j in range(n_joints):
        # Position subplot (row 0)
        ax_pos = axes[0, j]
        ax_pos.plot(times, pos_arr[:, j], linestyle="-", marker=None, color="C0")
        ax_pos.set_ylabel("Position")
        ax_pos.set_title(f"Joint {j}")
        ax_pos.grid(True)

        # Velocity subplot (row 1)
        ax_vel = axes[1, j]
        ax_vel.plot(times, vel_arr[:, j], linestyle="-", marker=None, color="C1")
        ax_vel.set_ylabel("Velocity")
        ax_vel.grid(True)

        # Torques subplot (row 2)
        ax_tq = axes[2, j]
        for idx, fname in enumerate(torque_field_names):
            arr = torque_arrs[fname][:, j]
            color = torque_colors[idx % len(torque_colors)]
            ax_tq.plot(times, arr, linestyle="-", marker=None, color=color, label=fname)
        ax_tq.set_ylabel("Torque (Nm)")
        ax_tq.set_xlabel("Time (s)")
        ax_tq.grid(True)
        if j == cols - 1:
            # place legend only on last column to avoid overlap
            ax_tq.legend(loc="upper left", fontsize="small")

    fig.tight_layout()
    plt.show()


def plot_robot_states_grid_linked_x(
    robot_states: List["RobotState"], robot_model: toco.models.RobotModelPinocchio
):
    """
    Single figure: 3 rows x N columns (one column per joint).
    All subplots for a given time axis are coupled so zooming/panning the x-axis
    in one subplot affects all others.
    """
    if not robot_states:
        raise ValueError("robot_states list is empty")

    # Extract times in seconds (fallback to index)
    times = []
    for i, rs in enumerate(robot_states):
        ts = getattr(rs, "timestamp", None)
        if (
            ts is not None
            and hasattr(ts, "seconds")
            and (ts.seconds != 0 or getattr(ts, "nanos", 0) != 0)
        ):
            times.append(float(ts.seconds) + float(getattr(ts, "nanos", 0)) * 1e-9)
        else:
            times.append(float(i))
    times = np.array(times)
    times = times - times[0]

    # Helper: longest repeated field length
    def longest_length(field_name):
        return max((len(getattr(rs, field_name)) for rs in robot_states), default=0)

    torque_field_names = [
        "joint_torques_computed",
        "prev_joint_torques_computed",
        "prev_joint_torques_computed_safened",
        "motor_torques_measured",
        "motor_torques_external",
        "motor_torques_desired",
    ]

    n_pos = longest_length("joint_positions")
    n_vel = longest_length("joint_velocities")
    n_torque = max((longest_length(f) for f in torque_field_names), default=0)
    n_posquat = 7
    n_joints = max(n_pos, n_vel, n_torque, n_posquat)
    if n_joints == 0:
        raise ValueError("No joint data found in robot_states")

    T = len(robot_states)

    def build_array(field_name, n_cols):
        arr = np.full((T, n_cols), np.nan, dtype=float)
        for t, rs in enumerate(robot_states):
            vals = list(getattr(rs, field_name, []))
            for j, v in enumerate(vals[:n_cols]):
                arr[t, j] = float(v)
        return arr

    def build_posquat_array(n_cols):
        arr = np.full((T, n_cols), np.nan, dtype=float)
        for t, rs in enumerate(robot_states):
            pos, quat = robot_model.forward_kinematics(torch.tensor(rs.joint_positions))
            for j, v in enumerate(pos[:n_cols]):
                arr[t, j] = float(v)
            for j, v in enumerate(quat[: n_cols - 3]):
                arr[t, 3 + j] = float(v)
        return arr

    pos_arr = build_array("joint_positions", n_joints)
    vel_arr = build_array("joint_velocities", n_joints)
    torque_arrs = {f: build_array(f, n_joints) for f in torque_field_names}
    posquat_arr = build_posquat_array(n_joints)

    cols = n_joints
    rows = 4
    # sharex='all' couples x-axis across all subplots. Also keep sharex='col' would share per column;
    # use 'all' to ensure every subplot shares the same x-axis.
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows), sharex="all")
    if cols == 1:
        axes = axes.reshape(rows, 1)

    torque_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for j in range(n_joints):
        # Position subplot (row 0)
        ax_pos = axes[0, j]
        ax_pos.plot(times, pos_arr[:, j], linestyle="-", marker=None, color="C0")
        ax_pos.set_ylabel("Position")
        ax_pos.set_title(f"Joint {j}")
        ax_pos.grid(True)

        # Velocity subplot (row 1)
        ax_vel = axes[1, j]
        ax_vel.plot(times, vel_arr[:, j], linestyle="-", marker=None, color="C1")
        ax_vel.set_ylabel("Velocity")
        ax_vel.grid(True)

        # Torques subplot (row 2)
        ax_tq = axes[2, j]
        for idx, fname in enumerate(torque_field_names):
            arr = torque_arrs[fname][:, j]
            color = torque_colors[idx % len(torque_colors)]
            ax_tq.plot(times, arr, linestyle="-", marker=None, color=color, label=fname)
        ax_tq.set_ylabel("Torque (Nm)")
        ax_tq.set_xlabel("Time (s)")
        ax_tq.grid(True)
        if j == cols - 1:
            ax_tq.legend(loc="upper left", fontsize="small")

        # Cartesian subplot (row 3)
        ax_posquat = axes[3, j]
        ax_posquat.plot(
            times, posquat_arr[:, j], linestyle="-", marker=None, color="C2"
        )
        ax_posquat.set_ylabel("Position/Quaternion")
        ax_posquat.grid(True)

    fig.tight_layout()

    # this just causes crash
    # # Optional: explicitly link x-limits for all axes to ensure immediate coupling in some backends
    # all_axes = fig.get_axes()
    # def on_xlim_changed(ax):
    #     xmin, xmax = ax.get_xlim()
    #     for other in all_axes:
    #         if other is not ax:
    #             other.set_xlim(xmin, xmax)
    #     fig.canvas.draw_idle()

    # # Connect the callback to each axis
    # for ax in all_axes:
    #     ax.callbacks.connect('xlim_changed', lambda ax_instance: on_xlim_changed(ax_instance))

    plt.show()


def output_episode_stats(episode_name, robot_states):
    latency_arr = np.array(
        [robot_state.prev_controller_latency_ms for robot_state in robot_states]
    )
    latency_mean = np.mean(latency_arr)
    latency_std = np.std(latency_arr)
    latency_max = np.max(latency_arr)
    latency_min = np.min(latency_arr)

    success_arr = np.array(
        [robot_state.prev_command_successful for robot_state in robot_states]
    )
    success_rate = np.mean(success_arr)

    print(
        f"{episode_name}: {latency_mean:.4f}/ {latency_std:.4f} / {latency_max:.4f} / {latency_min:.4f} / {100 * success_rate:.2f}%"
    )


def list_franka_singularities(current_config: torch.Tensor):
    # based on https://arxiv.org/pdf/2211.02516
    # Robot parameters (replace with actual values if different)
    # copied from https://inria.hal.science/hal-02265293/document or https://github.com/Pradn1l/Rospy-FK-7Axis-Robot?tab=readme-ov-file
    a5 = -0.0825
    a7 = 0.088
    d3 = 0.316
    d5 = 0.384

    cc1 = current_config[0]
    cc2 = current_config[1]
    cc3 = current_config[2]
    cc4 = current_config[3]
    cc5 = current_config[4]
    cc6 = current_config[5]
    cc7 = current_config[6]

    configs = []

    # A) s(q2)=0 ∧ c(q3)=0 ∧ c(q5)=0
    # s(q2)=0 -> q2 = 0 or pi
    q2_A = 0.0 if cc2 < math.pi / 2 else math.pi
    # c(q3)=0 -> q3 = pi/2 or -pi/2
    q3_A = -math.pi / 2 if cc3 < 0 else math.pi / 2
    # c(q5)=0 -> q5 = pi/2 or -pi/2
    q5_A = -math.pi / 2 if cc5 < 0 else math.pi / 2
    qA = torch.tensor(
        [cc1, q2_A, q3_A, cc4, q5_A, cc6, cc7],
        dtype=torch.float32,
    )
    configs.append(qA)

    # B) c(q5)=0 ∧ fsing,1(q4,q6)=0
    # c(q5)=0 -> q5 = pi/2
    q5_B = math.pi / 2
    # fsing,1 = c(q4)*a5*(a7 + (d3+d5)*s(q6)) + s(q4)*( -a7*d3 + (a5**2 - d5*d3)*s(q6) ) = 0
    # Solve for q6 given chosen q4. choose q4 = 0.42
    q4_B = cc4
    # Let S = s(q6). Solve linear equation in S: [c(q4)*a5*(d3+d5) + s(q4)*(a5**2 - d5*d3)] * S + c(q4)*a5*a7 - s(q4)*a7*d3 = 0
    coeff_S = math.cos(q4_B) * a5 * (d3 + d5) + math.sin(q4_B) * (a5**2 - d5 * d3)
    const_term = math.cos(q4_B) * a5 * a7 - math.sin(q4_B) * a7 * d3
    # S = -const_term / coeff_S  (check domain)
    S = -const_term / coeff_S if abs(coeff_S) > 1e-12 else 0.0
    # clamp S into [-1,1]
    S = max(-1.0, min(1.0, S))
    q6_B = math.asin(S)
    qB = torch.tensor([cc1, cc2, cc3, q4_B, q5_B, q6_B, cc7], dtype=torch.float32)
    configs.append(qB)

    # C) q4 = arctan( a5(d3 + d5) / ( -a5**2 + d5*d3 ) ) ∧ s(q5)=0
    num = a5 * (d3 + d5)
    den = -(a5**2) + d5 * d3
    q4_C = math.atan2(num, den)
    # s(q5)=0 -> q5 = 0 or pi; choose 0
    q5_C = 0.0
    qC = torch.tensor([cc1, cc2, cc3, q4_C, q5_C, cc6, cc7], dtype=torch.float32)
    configs.append(qC)

    # D) s(q2)=0 ∧ fsing,2(q3,q4,q5,q6)=0
    # s(q2)=0 -> q2 = 0
    q2_D = 0.0
    # fsing,2 is complicated. We'll pick q3,q4,q5 and solve for q6 numerically.
    # choose q3 = 0.42, q4 = 0.42, q5 = 0.42 (note s(q2)=0 requirement is already met)
    q3_D = cc3
    q4_D = cc4
    q5_D = cc5

    # define fsing_2 as python function (using paper's expression)
    def fsing2(q3, q4, q5, q6):
        x = math.tan(q4)
        y = math.sqrt(x * x + 1.0)
        c3 = math.cos(q3)
        s3 = math.sin(q3)
        c5 = math.cos(q5)
        s5 = math.sin(q5)
        c2_q5 = c5 * c5
        term1 = -a5 * (x**2 * a5 + y * x * d3 + (1 - y) * a5) * c3 * a7 * c2_q5
        term2 = a5 * s3 * (x**2 * a5 + y * x * d3 + (1 - y) * a5) * a7 * s5 * c5
        inner = (a5**2 - d5 * d3) * x + (d3 + d5) * a5
        term3 = (
            -(math.sin(q6) * inner - a7 * (d3 * x - a5)) * c3 * (y * a5 + d5 * x - a5)
        )
        return term1 + term2 + term3

    # solve for q6 by scanning for sign change and then bisection
    def find_q6_for_fsing2(q3, q4, q5):
        # search interval [-pi, pi]
        N = 361
        qs = [-math.pi + 2 * math.pi * i / (N - 1) for i in range(N)]
        vals = [fsing2(q3, q4, q5, q) for q in qs]
        for i in range(len(qs) - 1):
            if vals[i] == 0 or vals[i] * vals[i + 1] < 0:
                a = qs[i]
                b = qs[i + 1]
                fa = vals[i]
                fb = vals[i + 1]
                for _ in range(50):
                    m = 0.5 * (a + b)
                    fm = fsing2(q3, q4, q5, m)
                    if abs(fm) < 1e-9:
                        return m
                    if fa * fm <= 0:
                        b = m
                        fb = fm
                    else:
                        a = m
                        fa = fm
                return 0.5 * (a + b)
        # fallback: return default
        return cc6

    q6_D = find_q6_for_fsing2(q3_D, q4_D, q5_D)
    qD = torch.tensor([cc1, q2_D, q3_D, q4_D, q5_D, q6_D, cc7], dtype=torch.float32)
    configs.append(qD)

    # return list
    configs_list = configs

    # Print / return
    for i, c in enumerate(configs_list, start=1):
        print(f"Singularity {i} config:", c.tolist())

    return configs_list


if __name__ == "__main__":
    robot = RobotInterface()

    print(
        "Control loop latency stats in milliseconds (avg / std / max / min / success_rate): "
    )

    init = robot.get_joint_positions()
    init_pos, init_quat = robot.get_ee_pose()

    robot.start_cartesian_impedance(
        # Kx=torch.zeros_like(robot.Kx_default), Kxd=torch.zeros_like(robot.Kxd_default)
    )
    robot.update_desired_ee_pose(init_pos + torch.tensor([0.05, 0, 0]), init_quat)
    # robot.update_desired_ee_pose(init_pos, R.functional.quaternion_multiply(init_quat, R.functional.rotvec2quat(torch.tensor([0,0,3.14/2]))))
    sleep(2)  # wait to finish
    # break to settle
    # robot.update_desired_joint_positions(robot.get_joint_positions())
    sleep(2)  # wait to finish
    # robot.update_desired_joint_positions(init)
    robot.update_desired_ee_pose(init_pos, init_quat)
    sleep(2)  # wait to finish
    robot_states = robot.get_previous_log()
    plot_robot_states_grid_linked_x(robot_states, robot.robot_model)

    # # test singularities
    # singular_configs = list_franka_singularities(init)
    # for singular_joints in singular_configs:
    #     robot_states = robot.move_to_joint_positions(
    #         singular_joints
    #     )
    #     robot_states = robot.move_to_joint_positions(
    #         singular_joints
    #     )
    #     plot_robot_states_grid_linked_x(robot_states, robot.robot_model)
    #     robot.start_cartesian_impedance()
    #     robot.update_desired_ee_pose(
    #         robot.get_ee_pose()[0] + 0.05 * torch.svd(robot.get_jacobian(robot.get_joint_positions())).U[:3,-1], robot.get_ee_pose()[1]
    #     )
    #     sleep(1)
    #     robot_states = robot.get_previous_log()
    #     plot_robot_states_grid_linked_x(robot_states, robot.robot_model)

    # Test cartesian PD
    robot_states = robot.move_to_ee_pose(
        init_pos + torch.tensor([0.05, 0, 0]), init_quat
    )
    plot_robot_states_grid_linked_x(robot_states, robot.robot_model)

    # Test cartesian PD
    robot_states = robot.move_to_ee_pose(init_pos, init_quat)
    plot_robot_states_grid_linked_x(robot_states, robot.robot_model)

    # Test joint PD
    # robot_states = robot.move_to_joint_positions(0.5 * init, Kq=Kq, Kqd=Kqd)
    # robot_states = robot.move_to_joint_positions(0.5 * init, Kq=Kq, Kqd=Kqd, time_to_go=15.0)
    robot_states = robot.move_to_joint_positions(0.5 * init)
    plot_robot_states_grid_linked_x(robot_states, robot.robot_model)
    robot_states = robot.move_to_joint_positions(
        1.0 * init,
    )
    # plot_robot_states_grid_linked_x(robot_states)
    output_episode_stats("Joint PD", robot_states)

    # Test cartesian PD
    robot_states = robot.move_to_ee_pose(init_pos, init_quat)
    # plot_robot_states_grid_linked_x(robot_states)
    output_episode_stats("Cartesian PD", robot_states)
