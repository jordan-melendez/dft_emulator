# dft_emulator

[![DOI](https://zenodo.org/badge/381869941.svg)](https://zenodo.org/badge/latestdoi/381869941)

This repository houses the nuclear density functional theory (DFT) emulation code based on the reduced basis (RB) method.
To my knowledge this is the first application of RB method to nuclear DFT.
This work was presented on July 12, 2021 to the BAND Collaboration, and again on December 13, 2021 at the BAND camp for the ISNET conference at Michigan State University.
The slides for each of these presentations can be found [here](https://github.com/jordan-melendez/quantum-emulator-examples).

# Running the code 
There is source code sitting in `dft/`, which gets called in the jupyter notebooks in `notebooks/`.

With `conda` installed, run
```bash
conda env create -f environment.yml
```
to create the `dft-env` environment.
It should install all the necessary packages. To install the code in `dft/`, run
```bash
pip install -e .
```
while in this repository. This will allow one to run `from dft import *` in python scripts.


Please feel free to raise an `Issue` here on GitHub if anything is broken or unclear.
I'd welcome a `Pull Request` to help improve it as well!
