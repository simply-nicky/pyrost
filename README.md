# Robust Ptychographic Speckle Tracking project
This project takes over [Andrew Morgan's Ptychographic Speckle Tracking](https://github.com/andyofmelbourne/speckle-tracking) as an improved version aiming to add robustness to the optimisation algorithm in the case of the high noise present in the measured data. The project is written in Python and compatible with Python 3.X

## 1. Speckle Tracking simulation
Speckle Tracking simulation (robust_speckle_tracking.simulation) is based on the Fresnel Diffraction wavefield propagation equations. The package is capable of generating a series of convergent X-ray beam snapshots produced by a lens with third order abberations profile. The X-ray beam goes through a barcode sample translated in transverse direction by a step at each frame.

The st_sim framework is comprised of STParams, STSim, and STConverter classes:

- STParams stores all the simulation parameters (see the docstring for the whole list of parameters), parameters() method returns the default set of parameters

- STSim performs the main calculations

- STConverter converts the simulated to data to the CXI file, that could be processed with Andrew Morgan's speckle-tracking package.

## 2. Robust Speckle Tracking algorithm
An improved Speckle Tracking algorithm aimed to improve the phase sensitivity in the case of low signal-to-noise ratio. Works with the same cxi file protocol used in Andrew Morgan's speckle-tracking package.

The fromawork consists of Protocol, STLoader, STData, SpeckleTracking1D, and AbberationsFit classes:

- Protocol provides a list of data attribute's paths used in the cxi file and default data types accordingly

- STLoader loads a cxi file according to the provided protocol and yields an STData object

- STData serves a data container purpose for all the data necessary to perform the speckle tracking. The class provides a list of handy functions to work with the data (for instance, masking frames, generating a new whitefield etc.)

- SpeckleTracking1D performs the lens' wavefront and umabbirated sample profile inference according to the robust speckle tracking algorithm

- AbberationsFit is capable of least squares model regression of the provided lens' abberations profile in order to attain the abberation coefficients

## Instalation

Required dependencies:

- Cython
- CythonGSL
- GNU Scientific Library
- h5py
- numpy
- scipy

Execute the following command in order to compile the c library used for the most computational heavy parts of code:

```
$ st_simulation % python setup.py build
```