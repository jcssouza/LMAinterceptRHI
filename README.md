This code is used for obtaining the results on J. Souza and E. Bruning, Assessment of turbulence intensity in different spots of lightning flash propagation submitted to Geophysical Reserach Letters. 

It combines radar and lightining data into the same framework allowing to retrieve radar detected and derived fields where lightning flash propagated through the radar scan.

Getting the dataset
--------

The radar and lightning dataset are available on [Zenodo](). The radar folder contains TTU Ka-band RHI scans for storms occurred during the spring and summer of 2015 - 2016 in the South Plains of the Texas Caprock. The files were processed in the netcdf format by the KTaL field experiment members. The lightning folder contains the LMA source data as events table in the hdf5 format after applying the flash sorting process available in [lmatools](https://github.com/deeplycloudy/lmatools).

Constructing the dataset
--------
The notebook go through the steps to obtain the tables for posterior analysis.


Example Use Case
--------
The notebook shows the chosen analysis for the dataset obtained by the previous steps.


Dependencies
--------
 * NumPy
 * SciPy
 * Matplotlib
 * Pandas
 * Xarray
 * Cartopy
 * [Py-ART](https://github.com/ARM-DOE/pyart)
 * [lmatools](https://github.com/deeplycloudy/lmatools)
 * [PyTDA](https://github.com/nasa/PyTDA)

