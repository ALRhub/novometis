# Polymetis for Newer PyTorch

This is a "fork" of [Monometis](https://github.com/hengyuan-hu/monometis).
It is further developed by the Intuitive-Learning-Robots Lab to enhance their Robot handling.

## Install & Compile

### On NUC

1) Create a cpu environment with dependencies.

2) Build libfranka to communicate with the robot.

3) Build & install polymetis

```bash
# clone & create env
git clone git@github.com:intuitive-robots/irl_polymetis.git
cd monometis/
mamba env create -f polymetis/environment_cpu.yml
conda activate robo

# compile stuff
./scripts/build_libfranka.sh
mkdir -p ./polymetis/build
cd ./polymetis/build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_FRANKA=ON
make -j
cd ../..

# inside the project root
pip install -e ./polymetis
```

### On Workstation

**Note**: this installation was done on Ubuntu 22.04 with an RTX 5080.
Nvidia's 50 series GPUs currently only support the latest version of pytorch, either 2.7.* or the nightly build.


First, install system requirements (requires sudo rights)

```bash
sudo apt install build-essential cmake libssl-dev
```

Next, set up a conda/mamba environment for the compilation step:

```bash
# clone & create *gpu* env
git clone git@github.com:intuitive-robots/irl_polymetis.git
cd irl_polymetis/
conda env create -n polymetis -f polymetis/environment_client.yml
conda activate polymetis
```

Next, install pytorch via pip, as this is currently the preferred installation method

```bash
# install pytorch 2.7
# INFO: When this .md got updated, there is only a pytorch2.7 version which is on the test wheel. I assume in near future, a stable version of pytorch 2.7 will released. Please add then to the environment.yml files again!
pip3 install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/test/cu128
```

Compile and install polymetis:

```bash
# no need to build libfranka on this machine
cmake -S polymetis -B polymetis/build -DCMAKE_BUILD_TYPE=Release
cmake --build polymetis/build -j
```

Install python runtime dependencies and the polymetis package:
```bash
pip install -r polymetis/requirements.txt
pip install -e ./polymetis
```

for irl users, try this cmake + downgrade mkl
```
conda install mkl==2024.0.0
CUDACXX=/usr/local/cuda-12.4/bin/nvcc /usr/bin/cmake -S polymetis -B polymetis/build -DCMAKE_BUILD_TYPE=Release
```

## Launch Polymetis (IRL)

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
