# Readme

## Folder structure

Folder structure should be:

vpp-gym/
  README.md
  setup.py
  vpp_gym/
    __init__.py
    envs/
      __init__.py
      vpp_gym.py

## Install 

After you have installed your package locally with pip install -e vpp-gym, you can create an instance of the environment with gym.make('vpp_gym:vpp-v0')
