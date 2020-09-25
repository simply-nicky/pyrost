# Robust Ptychographic Speckle Tracking project
This project takes over [Andrew Morgan's Ptychographic Speckle Tracking](https://github.com/andyofmelbourne/speckle-tracking) as an improved version aiming to add robustness to the optimisation algorithm in the case of high noise present in the measured data. The project is written in Python and compatible with Python 3.X

## 1. Speckle Tracking simulation
Speckle Tracking simulation (st_sim) is based on Fresnel Diffraction wavefield propagation equations. The package is capable of generating a series of convergent X-ray beam snapshots produced by a lens with third order abberations profile. The goes through a barcode sample translated in transverse diraection at every frame.

The st_sim framework is comprised of STParams, STSim, and STConverter classes:

- STParams stores all the simulation parameters (see the docstring for the whole list of parameters), parameters() method returns the default set of parameters

- STSim performs the main calculations

- STConverter converts the simulated to data to a CXI file, that could be processed with Andrew Morgan's speckle-tracking package.

## 2. Robust Speckle Tracking algorithm
An improved Speckle Tracking algorithm aimed to improve the phase sensitivity against noise. Works with the same cxi file protocol used in Andrew Morgan's speckle-tracking package.

The fromawork consists of STLoader and STData classes:

- STLoader loads cxi file according to the provided protocol, cxi_protocol() returns the default cxi file protocol.

- STData performs the lens' wavefront and umabbirated sample profile inference according to the robust speckle-tracking algorithm (work in progress)

## Instalation

Required dependencies:

- Numpy
- h5py
- cython
- GNU Scientific Library

Execute the following command in order to compile the c library used for the most computational heavy parts of code:

```
$ st_simulation % cd st_sim/bin
$ bin % python setup.py build_ext -i
```