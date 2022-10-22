# An environment of CommonRoad

This repository is an Python3.6 virtual environment of CommondRoad. You can clone this repository and use it directly, without installing any modules, bacause all modules needed are installed in the virtual environment. But it just works on Ubuntu 18.04 now, other operate system will be adapted later. However, there is an installation instructions of CommonRoad on other system(just Ubuntu now).

 ## 1. Instructions for direct use

You can use the Python3.6  virtual environment directly on Ubuntu 18.04.

### 1.1 Clone

To git clone this repository on you computor where you prefe. Fox example, the file path is `FILE_PATH`.

### 1.2 Activate

```shell
cd FILE_PATH
source CommonRoad-Env/Ubuntu/environment/bin/activate
```

Run the above command, you can activate the Python3.6  virtual environment. Then, you can run the domes to verify.  Fox example,

```shell
cd demos/demo-1
python demo-1.py
```

After running  above command, you can get the image below.

![](./demos/demo-1/demo_1.png)

### 1.3 Deactivate

In the Python3.6  virtual environment, run the command below.

```shell
deactive
```

## 2. Instructions for CommandRoad Installation

If you wan to in install CommandRoad by yourself. Or your operate system is not Ubuntu 18.04.

Run  the command below to install base modules of CommandRoad.

```shell
pip3 install commonroad-all
```

You may encounter some problems. If you encounter the error below.

```shell
Could not find a version that satisfies the requirement sumocr>=2022.1 (from commonroad-all) (from versions: 2021.1, 2021.2, 2021.3, 2021.4, 2021.5)
No matching distribution found for sumocr>=2022.1 (from commonroad-all)
```

Need to install each tool separately.

```shell
pip3 install commonroad-io
pip3 install commonroad-drivability-checker
pip3 install commonroad-vehicle-models
pip3 install commonroad-route-planner
pip3 install sumocr
```

If you encounter problem below.

```shell
Command "python setup.py egg_info" failed with error code 1 in /tmp/pip-build-izjldkk1/pyproj/
```

Run:

```shell
sudo python3 -m pip install --upgrade --force pip
sudo pip install setuptools==33.1.1
```

If you encounter problem below:

```shell
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
launchpadlib 1.10.6 requires testresources, which is not installed.
```

Run:

```shell
python3 -m pip install launchpadlib
```