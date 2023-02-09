# Artificial Intelligence in Robotics

This repository contains the code for a Semester project on Artificial Intelligence in Robotics.

## Setting up Coppelia Sim

1. Download Coppelia Sim EDU [Coppelia Sim[(https://coppeliarobotics.com/downloads)
2. Open Coppelia Sim using `/usr/lib/coppelia_sim_edu/coppeliaSim.sh`
3. Open the scene `UIR/T1a-ctrl/vrep-scenes` (blocks.ttt for a single robots, blocks_multirobot.ttt for two robots)

## Running the Program

1. Install [WSL (Windows Subsystem for Linux)](https://docs.microsoft.com/en-us/windows/wsl/install-win10) and [Python 3.8](https://www.python.org/downloads/)
2. Clone this repository using Git: `git clone https://github.com/malek-luky/Artificial-Intelligence-in-Robotics`
3. Navigate to the repository: `cd Artificial-Intelligence-in-Robotics`
4. Install the required packages using pip: `pip install -r requirements.txt` (The file is missing, sorry)
5. Run the program: `python Explorer.py` (single robot) `python Explorer_multi.py`(two robots)

## Setup

This program was developed and tested on a WSL (Ubuntu 20.04) with Python 3.8.

## Credits

This program was developed by malek-luky. I would like to acknowledge the following sources for their help and inspiration:
[Moodle UIR](https://cw.fel.cvut.cz/b221/courses/uir/hw/start)