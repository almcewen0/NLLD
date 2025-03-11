# NLLD - NICER Light-Leak Diagnostics

## Description

Some of the scripts that have been utilized to perform NICER light-leak mitigation. They mainly operate on MKF files.
This library is not intended for the purposes of any NICER data analysis, and thus is not for the average NICER user.

## Installation

At the moment, NLLD can be installed locally after cloning the directory or simply downloading 
and untarring it. Then from the NLLD root directory:

```bash
  python -m pip install .
```

You may add the '-e' option to pip to install as an editable package.

The code has been tested on Python 3.9.16

## Scripts

Upon installation, these command line scripts will be available to you: **mkfdiagnostics**,
**comparemkfs**, and **ags3withbrightearth**. You can get their help messages and the list of required and optional arguments with the usual '-h' 
option.

## Notebooks

There are several notebooks that parse the data output from the above two scripts. They produce visualization plots and
videos of some statistical elements of each NICER detector.

## Disclaimer

NLLD is unrelated to [NICERDAS](https://heasarc.gsfc.nasa.gov/docs/nicer/nicer_analysis.html), the official data analysis software package that is part of the HEASoft 
distribution. NLLD is not intended for any NICER data analysis either. Nothing here should be construed as formally 
recommended by the NICER instrument team.

## License

[MIT](https://choosealicense.com/licenses/mit/)