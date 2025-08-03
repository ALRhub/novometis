# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
import queue
import threading
import time
import weakref

import grpc

import polymetis_pb2
import polymetis_pb2_grpc

log = logging.getLogger(__name__)

EMPTY = polymetis_pb2.Empty()


class GripperInterface:
    """Gripper interface class to initialize a connection to a gRPC gripper server.

    Args:
        ip_address: IP address of the gRPC-based gripper server.
        port: Port to connect to on the IP address.
        rate_limit_ms: Time after a command (in ms) when subsequent commands
            are ignored.
        nominal_width: Expected max_width (m) of the gripper. A warning is
            issued if the actual max_width is below this value.
        open_width_threshold: Threshold (m) below the max_width where the
            subsequent commands to open the gripper are ignored.
    """

    def __init__(
        self,
        ip_address: str = "localhost",
        port: int = 50052,
        rate_limit_ms: float = 100.0,
        nominal_width: float = 0.08,  # max_width of the franka gripper should be 0.0801m
        open_width_threshold: float = 0.01,
    ):
        # Connect to server
        self.channel = grpc.insecure_channel(f"{ip_address}:{port}")
        self.grpc_connection = polymetis_pb2_grpc.GripperServerStub(self.channel)
        self._finalizer = weakref.finalize(self, self.channel.close)

        # Get metadata
        try:
            self.metadata = self.grpc_connection.GetRobotClientMetadata(EMPTY)
        except grpc.RpcError:
            log.warning("Metadata unavailable from server.")

        if self.metadata.max_width < nominal_width:
            log.warning(
                f"Gripper max width ({self.metadata.max_width}) is below "
                f"nominal width ({nominal_width}). This usually indicates that "
                "the server was launched when the gripper was grasping "
                "something and therefore could not fully close. This may result "
                "in the gripper not being able to close fully. Remove the "
                "object from the gripper and restart the gripper server."
            )

        self.open_width_threshold = open_width_threshold
        self.open_width = self.metadata.max_width - self.open_width_threshold

        # Execute commands from cache in separate thread
        self._command_thr = threading.Thread(
            target=self._command_executor,
            daemon=True,
        )

        self._command_queue = queue.Queue(maxsize=1)
        self._command_thr.start()

        self.wait_ns = int(rate_limit_ms * 1_000_000)  # Convert ms to ns
        # time (in ns) since the last command message was sent
        self._last_cmd_time_ns = 0

    def close(self):
        """Close the gRPC connection."""
        if self._finalizer.alive:
            self._finalizer()
            log.debug("Closed gRPC connection to polymetis gripper server.")

    def _command_executor(self):
        while True:
            command, msg = self._command_queue.get()
            try:
                command(msg)
            except grpc.RpcError as e:
                raise grpc.RpcError(f"GRIPPER SERVER ERROR --\n{e.details()}") from None
            self._command_queue.task_done()

    def _send_gripper_command(self, command, msg, blocking: bool = True) -> None:
        self._command_queue.put((command, msg))

        if blocking:
            self._command_queue.join()

    def get_state(self) -> polymetis_pb2.GripperState:
        """Returns the state of the gripper.

        Returns:
            gripper state (polymetis_pb2.GripperState)
        """
        return self.grpc_connection.GetState(EMPTY)

    def goto(
        self, width: float, speed: float, force: float = 0.0, blocking: bool = True
    ) -> None:
        """Commands the gripper to a certain width.

        CAUTION: The server will block if the gripper is not able to move to
        the desired width.

        Args:
            width: Target width (m)
            speed: Velocity of the movement (m/s)
            force: Maximum force the gripper will exert (N) (ignored by the Franka)
            blocking: If True, wait for the command to be sent before returning.
                In practice, sending the command is almost instantaneous.
        """
        cmd = polymetis_pb2.GripperCommand(
            width=width, speed=speed, force=force, grasp=False
        )
        cmd.timestamp.GetCurrentTime()

        # Update the timestamp of the last command message
        self._last_cmd_time_ns = time.perf_counter_ns()

        self._send_gripper_command(
            self.grpc_connection.Goto,
            cmd,
            blocking=blocking,
        )

    def grasp(
        self,
        speed: float,
        force: float,
        grasp_width: float = 0.0,
        epsilon_inner: float = -1.0,
        epsilon_outer: float = -1.0,
        blocking: bool = True,
    ):
        """Commands the gripper to close.

        For Robotiq grippers, this is equivalent to calling `goto` with
        grasp_width (0 by default).

        For the Franka Hand, see documentation for franka::Gripper::move in
        libfranka. The gripper closes until the maximum force is exceeded. If
        `grasp_width - epsilon_inner < final_width < grasp_width + epsilon_outer`,
        then the grasp was successful (indicated by `is_grasped=True` in the
        GripperState), and the gripper continues to exert the maximum force.
        Otherwise the gripper stops exerting force and the command finishes.

        Args:
            speed: Velocity of the movement (m/s)
            force: Maximum force the gripper will exert (N)
            grasp_width: Target width of the grasp (m)
            epsilon_inner: Maximum tolerated deviation when the actual grasped
                width is smaller than the commanded grasp width (m)
            epsilon_outer: Maximum tolerated deviation when the actual grasped
                width is larger than the commanded grasp width (m)
        """
        cmd = polymetis_pb2.GripperCommand(
            width=grasp_width,
            speed=speed,
            force=force,
            grasp=True,
            epsilon_inner=epsilon_inner,
            epsilon_outer=epsilon_outer,
        )
        cmd.timestamp.GetCurrentTime()

        # Update the timestamp of the last command message
        self._last_cmd_time_ns = time.perf_counter_ns()

        self._send_gripper_command(
            self.grpc_connection.Goto,
            cmd,
            blocking=blocking,
        )

    def set_state(
        self, target: float, speed: float, force: float, blocking: bool = True
    ):
        """Tries to update the state of the gripper to match the target.
        If state >= 0.0, opens the gripper by moving to its max_width. No force
        is exerted, and the server stays locked if the movement cannot be
        completed.
        If state < 0.0, closes the gripper by grasping to a width of 0. After
        the gripper stops moving, the grasp always succeeds, since we set the
        outer epsilon to be larger than the width of the gripper.

        This command is designed to be called at high frequency (e.g. >=30Hz)
        without causing the server to deadlock. This means that some commands
        may be ignored if they are received too quickly. This is intended for
        policy rollout or tele-operation, where desired open/close states are
        sent continuously.

        Args:
            speed: Speed of the movement (m/s)
            force: Maximum force that will be applied while closing (N)
            blocking: If True, wait for the command to be sent before returning.
                In practice, sending the command is almost instantaneous.
        """
        now = time.perf_counter_ns()

        if now - self._last_cmd_time_ns < self.wait_ns:
            # The gripper state (i.e. is_moving) might not update for some time
            # after a command is sent. However, the gripper server might go into a
            # deadlock if it receives too many requests too quickly. Therefore
            # we enforce a simple rate limit.
            return

        state = self.get_state()

        if state.is_moving:
            # Don't send a new command is the gripper is still moving. In
            # general, the server can accept new commands while a command is
            # running (it just keeps the last command), but it's unnecessary
            # and risks causing the server to deadlock.
            return

        # Don't open the gripper again if it's already open
        if target >= 0.0 and state.width < self.open_width:
            self.goto(
                width=self.metadata.max_width,
                speed=speed,
                blocking=blocking,
            )

        # Don't close the gripper again if it's already grasping something
        elif target < 0.0 and not state.is_grasped:
            self.grasp(
                speed=speed,
                force=force,
                grasp_width=0.0,
                # probably unnecessary, since we set the grasp_width to be 0.0
                epsilon_inner=0.01,
                # much wider than the width of the gripper, so any final width is a success
                epsilon_outer=1.0,
                blocking=blocking,
            )
