# Novometis: the Latest Fork of Polymetis!

Novometis is a fork of [a fork](https://github.com/intuitive-robots/irl_polymetis) of [a fork](https://github.com/hengyuan-hu/monometis) of... you know what? Just check the commit history.

Novometis is based on [facebook's polymetis](https://github.com/facebookresearch/fairo/tree/main/polymetis), which is unfortunately no longer being maintained.
It is primarily intended for robot setups involving a real-time computer without a GPU (the server) and a GPU computer used for policy inference (the client).
Regarding robot hardware, we focus on the [Franka Emika Robot](https://robodk.com/robot/Franka/Emika-Panda), although the source code for other robots has not been removed.

## Why Polymetis?

The original goal of Polymetis was to provide a unified interface for writing robot controllers that can be used in both simulation and on real hardware.
In practice, such an interface is almost impossible to create.
Not only do modern simulators (e.g. IsaacSim, ManiSkill) have differing paradigms for robot control (or may not support custom controllers), but even defining a common API for all robot hardware can be too restrictive.

Despite these shortcomings, Polymetis is a powerful framework for controlling robot hardware, and the source code remains a valuable resource.
One of the main advantages of Polymetis is the ability to write robot controllers in Python instead of C++, which are then jit compiled to overcome the performance limitations of Python for real-time execution.
As a result, these controllers don't even need to be predefined on the server, and a client can inject (almost) arbitrary code to be executed in the control loop at 1kHz!

# Installation

## On GPU Computer

1. Install the system build requirements (requires sudo rights).

```bash
sudo apt install build-essential cmake libssl-dev
```

2. Clone the repository.

```bash
git clone git@github.com:ALRhub/novometis.git
cd novometis
```

3. Set up a conda/mamba environment for compilation.
This environment contains some minimal libraries that should likely not affect any of your machine learning code.

```bash
mamba env create -n polymetis -f polymetis/environment_client.yml
mamba activate polymetis
```

If you already have an existing conda environment activated, run the following instead:
```bash
mamba env update -f polymetis/environment_client.yml
```

4. Install your desired version of pytorch.
You should probably install via pip, as this is currently the preferred method.

```bash
# e.g. substitute with any version of pytorch you like
pip3 install torch --index-url https://download.pytorch.org/whl/test/cu128
```

5. Compile and install polymetis.

```bash
cmake -S polymetis -B polymetis/build -DCMAKE_BUILD_TYPE=Release
cmake --build polymetis/build -j --target install
```

6. Install python runtime dependencies and the polymetis package itself.
```bash
pip install -r polymetis/requirements.txt
pip install -e ./polymetis
```

## On Real-Time Computer

1) Create a cpu environment with dependencies.

2) Build libfranka to communicate with the robot.

3) Build & install polymetis

```bash
# clone & create env
git clone git@github.com:ALRhub/novometis.git
cd novometis
mamba env create -n "novometis_server_cpu" -f polymetis/environment_server_cpu.yml
mamba activate novometis_server_cpu
pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch==2.8.0" torchvision
pip install hydra-core mujoco

# if build_libfranka.sh fails because of readline errors (bash version but conda only knows incompatible readline 8.2 still)
# then overwrite env libreadline with system one (there is probably a better proper way to doing this)
ln -s /usr/lib/libreadline.so.8.3 "${CONDA_PREFIX}/lib/libreadline.so.8.3"   
ln -sf "${CONDA_PREFIX}/lib/libreadline.so.8.3" "${CONDA_PREFIX}/lib/libreadline.so.8"

# pytorch ships some old version of protobuf which causes compilation issues.
# renaming seems like a dumb approach but I can't figure out how to make cmake ignore that subdir otherwise
mv "${CONDA_PREFIX}/lib/python3.13/site-packages/torch/include/google/protobuf" "${CONDA_PREFIX}/lib/python3.13/site-packages/torch/include/google/protobuf-backup"

# you probably need libfranka, so build it
./scripts/build_libfranka.sh
# but the first try will probably fail because the current poco version requires C++17, so modify
sed -i 's/^set(CMAKE_CXX_STANDARD 14)$/set(CMAKE_CXX_STANDARD 17)/' polymetis/src/clients/franka_panda_client/third_party/libfranka/CMakeLists.txt
# to `set(CMAKE_CXX_STANDARD 17)` and retry
./scripts/build_libfranka.sh

# compile and install stuff
./scripts/build_polymetis.sh

# undo renaming (if you care, seems to work fine in renamed state and makes recompile easier)
mv "${CONDA_PREFIX}/lib/python3.13/site-packages/torch/include/google/protobuf-backup" "${CONDA_PREFIX}/lib/python3.13/site-packages/torch/include/google/protobuf"

# try 
launch_robot.py -cp=$(pwd)/polymetis robot_client=empty_statistics_client use_real_time=False
# depending on your distro/.bashrc, you might need an empty export
# export LD_LIBRARY_PATH=

# or launch this
launch_robot.py -cp=$(pwd)/polymetis robot_client=franka_hardware use_real_time=False robot_client.executable_cfg.mock=true   
# or launch this
launch_robot.py -cp=$(pwd)/polymetis robot_client=franka_mujoco_sim use_real_time=False

# and then run (separately)
benchmark_control_latency.py
```

## Launch Polymetis

To launch the robot or gripper server:

```bash
# start robot server
./scripts/start_robot.sh 101

# start gripper server
./scripts/start_gripper.sh 201
```

You need to specify the robots id. It is the same as the last part if the robots ip address (10.10.10.**101**).

Additional arguments for the commands can be seen in the table below.

Argument                | Description
----------------------- | -----------
-h, --help              | Display a help message.
-i, --pc-ip [IP]        | Change the ip address, where the server is running. Default is *localhost*.
-p, --port [PORT]       | Change the port of the server. Default is *50051*.
-c, --conda [CONDA_ENV] | Change the conda environment, where polymetis is installed. Default is *poly*.
-r, --readonly          | Starts the server in readonly mode. For usage with the robots white mode. Only for the robot server.

---

## Polymetis: A real-time PyTorch controller manager

**Write [PyTorch](http://pytorch.org/) controllers for robots, test them in simulation, and seamlessly transfer to real-time hardware.**

**Polymetis** powers robotics research at [Facebook AI Research](https://ai.facebook.com/). If you want to write your robot policies in PyTorch for simulation and immediately transfer them to high-frequency (1kHz) policies on real-time hardware (e.g. Franka Panda), read on!

## Features

- **Unified simulation & hardware interface**: Write all your robot controllers just once -- immediately transfer them to real-time hardware. You can even train neural network policies using reinforcement learning in simulation and transfer them to hardware, with just a single configuration toggle.
- **Write your own robot controllers:** Use the building blocks in our [TorchControl](https://facebookresearch.github.io/fairo/polymetis/torchcontrol-doc.html) library to write complex robot controllers, including operational space control. Take advantage of our wrapping of the [Pinocchio](https://github.com/stack-of-tasks/pinocchio) dynamics library for your robot dynamics.
- **Drop-in replacement for [PyRobot](https://pyrobot.org/)**: If you're already using PyRobot, you can use the exact same interface, but immediately gain access to arbitrary, custom high-frequency robot controllers.

## Get started

To get started, you only need one line:

```
conda install -c pytorch -c fair-robotics -c aihabitat -c conda-forge polymetis
```

You can immediately start running the [example scripts](https://github.com/facebookresearch/fairo/tree/main/polymetis/examples) in both simulation and hardware. See [installation](https://facebookresearch.github.io/fairo/polymetis/installation.html) and [usage](https://facebookresearch.github.io/fairo/polymetis/usage.html) documentation for details.

## Documentation

All documentation on the [website](https://facebookresearch.github.io/fairo/polymetis/). Includes:

- Guides on setting up your [Franka Panda](https://frankaemika.github.io/docs/libfranka.html) hardware for real-time control
- How to quickly get started in [PyBullet](https://github.com/bulletphysics/bullet30) simulation
- Writing developing your own custom controllers in PyTorch
- Full [autogenerated documentation](https://facebookresearch.github.io/fairo/polymetis/modules.html)

## Benchmarking

To run benchmarking, first configure the [script](polymetis/tests/python/polymetis/benchmarks/benchmark_robustness.py) to point to your hardware instance, then run

```bash
asv run --python=python --set-commit-hash $(git rev-parse HEAD)
```

To update the dashboard, run:

```bash
asv publish
```

Commit the result under `.asv/results` and `docs/`; it will show up under the benchmarking page in the documentation.

## Citing
If you use Polymetis in your research, please use the following BibTeX entry.
```
@misc{Polymetis2021,
  author =       {Lin, Yixin and Wang, Austin S. and Sutanto, Giovanni and Rai, Akshara and Meier, Franziska},
  title =        {Polymetis},
  howpublished = {\url{https://facebookresearch.github.io/fairo/polymetis/}},
  year =         {2021}
}
```

Note: Giovanni Sutanto contributed to the repository during his research internship at Facebook Artificial Intelligence Research (FAIR) in Fall 2019.

## Contributing

See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out. [Make an issue](https://github.com/facebookresearch/fairo/issues/new/choose) for bugs and feature requests, or contribute a new robot controller by making a [pull request](https://github.com/facebookresearch/fairo/pulls)!

## License
Polymetis is MIT licensed, as found in the [LICENSE](LICENSE) file.
