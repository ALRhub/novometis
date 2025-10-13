from time import sleep
import matplotlib.pyplot as plt
from typing import List
import numpy as np


from polymetis import RobotInterface
import torch

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

def plot_robot_states_grid_linked_x(robot_states: List['RobotState']):
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
        if ts is not None and hasattr(ts, "seconds") and (ts.seconds != 0 or getattr(ts, "nanos", 0) != 0):
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

    cols = n_joints
    rows = 3
    # sharex='all' couples x-axis across all subplots. Also keep sharex='col' would share per column;
    # use 'all' to ensure every subplot shares the same x-axis.
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows), sharex='all')
    if cols == 1:
        axes = axes.reshape(rows, 1)

    torque_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for j in range(n_joints):
        # Position subplot (row 0)
        ax_pos = axes[0, j]
        ax_pos.plot(times, pos_arr[:, j], linestyle='-', marker=None, color='C0')
        ax_pos.set_ylabel("Position")
        ax_pos.set_title(f"Joint {j}")
        ax_pos.grid(True)

        # Velocity subplot (row 1)
        ax_vel = axes[1, j]
        ax_vel.plot(times, vel_arr[:, j], linestyle='-', marker=None, color='C1')
        ax_vel.set_ylabel("Velocity")
        ax_vel.grid(True)

        # Torques subplot (row 2)
        ax_tq = axes[2, j]
        for idx, fname in enumerate(torque_field_names):
            arr = torque_arrs[fname][:, j]
            color = torque_colors[idx % len(torque_colors)]
            ax_tq.plot(times, arr, linestyle='-', marker=None, color=color, label=fname)
        ax_tq.set_ylabel("Torque (Nm)")
        ax_tq.set_xlabel("Time (s)")
        ax_tq.grid(True)
        if j == cols - 1:
            ax_tq.legend(loc='upper left', fontsize='small')

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


if __name__ == "__main__":
    robot = RobotInterface()

    print(
        "Control loop latency stats in milliseconds (avg / std / max / min / success_rate): "
    )



    init = robot.get_joint_positions()
    init_pos, init_quat = robot.get_ee_pose()

    robot.start_cartesian_impedance(Kx=torch.zeros_like(robot.Kx_default), Kxd=torch.zeros_like(robot.Kxd_default))
    robot.update_desired_ee_pose(init_pos + torch.tensor([0.05, 0, 0]), init_quat)
    sleep(1) # wait to finish    
    robot.update_desired_ee_pose(init_pos, init_quat)
    sleep(1) # wait to finish
    robot_states = robot.get_previous_log()
    plot_robot_states_grid_linked_x(robot_states)
    

    # Test cartesian PD
    robot_states = robot.move_to_ee_pose(init_pos + torch.tensor([0.05, 0, 0]), init_quat)
    plot_robot_states_grid_linked_x(robot_states)

    # Test cartesian PD
    robot_states = robot.move_to_ee_pose(init_pos, init_quat)
    plot_robot_states_grid_linked_x(robot_states)

    # Test joint PD
    # robot_states = robot.move_to_joint_positions(0.5 * init, Kq=Kq, Kqd=Kqd)
    # robot_states = robot.move_to_joint_positions(0.5 * init, Kq=Kq, Kqd=Kqd, time_to_go=15.0)
    robot_states = robot.move_to_joint_positions(0.5 * init)
    plot_robot_states_grid_linked_x(robot_states)
    robot_states = robot.move_to_joint_positions(1.0 * init,)
    # plot_robot_states_grid_linked_x(robot_states)
    output_episode_stats("Joint PD", robot_states)

    # Test cartesian PD
    robot_states = robot.move_to_ee_pose(init_pos, init_quat)
    # plot_robot_states_grid_linked_x(robot_states)
    output_episode_stats("Cartesian PD", robot_states)
