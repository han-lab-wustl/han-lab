# han-lab

Project-specific and general scripts from Han Lab @ WUSTL

## projects > axonal-GCamp_dopamine

How to run preprocessing and motion correction on axonal-GCamp images from two-photon

NOTE:*most of Zahra's run scripts take command line arguments*

Relies on some dependencies in Python as well as a downloaded version of [Suite2p](https://github.com/MouseLand/suite2p) in your environment
```
pip install tifffile matplotlib numpy pandas
```

`run_axonal_dopamine_motion_reg.py`

On the command line (on Windows, Anaconda Powershell Prompt), navigate to the `axonal-GCamp_dopamine` folder

Type `python run_axonal_dopamine_motion_reg.py -h` for description of input arguments

To make folder structure:
```
python .\run_axonal_dopamine_motion_reg.py 0 e194 X:\dopamine_imaging\ --day 6
```

0 = step (making folder structure)

e194 = mouse name

X:\dopamine_imaging = drive containing mouse folder and imaging day subfolders within it

6 = day (optional argument); this is what the folder will be named

I suggest having a lookup table of day folder to experiment, imaging notes, camera acquisition etc. in a separate spreadsheet