# get_TESS_lightcurves
 Module to quickly retrieve light curve data from the NASA TESS mission, and search for transit signals. 
 
 Written by Maritza Soto.
 
 Please send your comments or issues to marisotov@gmail.com.
 
 # Installation
 
 This module uses the following dependencies:
 
 * `numpy`
 * `matplotlib`
 * `PyAstronomy` (https://pyastronomy.readthedocs.io/en/latest/)
 * `astropy` (https://www.astropy.org)
 * `lightkurve` (http://docs.lightkurve.org)
 
 [Optional]
 
 * `Transit Least Squares` (https://transitleastsquares.readthedocs.io/en/latest/index.html)
 * `juliet` (https://juliet.readthedocs.io/en/latest/index.html)


 # Usage
 
 This module is used from the command-line, and you can change the default parameters by adding flags to the call. A typical call would look something like this:
 
`python get_TESS_lightcurves.py -flag1 -flag2 -flag3 ...`
 
The flags are:
 
Flag | Default | Description
---- | ------- | -----------
`-TIC` | `307210830` | TIC number for each target. Could also be a comma-separated string with all the TIC identifiers. You must specify either this parameter or the `-from_file` one.
`-from_file` | `any` | Name of the file where the TIC identifiers are stored. You must specify either this parameter or the `-TIC` one.
`-tic_id` | `TIC_id` | Name of the column containing the TIC identifiers.
`-path_plots` | `.` | Path where the final plots will be saved. Default is the current directory.
`-interactive` | | Show the plots created for each star, before going to the next one.
`-no_detrend` | | Disable the detrending of the data.
`-window` | `1501` | Length of the window used during the detrending. Must be an odd number. Affects only the normal detrending using the `lightkurve` package, not the Gaussian Processes detrending.
`-GP` | | Use Gaussian Processes to do the detrending. Requires that `juliet` is istalled in your computer. Use this option after you have performed a simple detrending, and only when you see strong features of stellar activity or rotational modulation, otherwise it is highly probable that the trend will be overfitted, and some of the transit features might be removed.
`-from_fits` | | If the light curve data is to be read from fits files.
`-path_fits` | `None` | Path to the fits files with the light curve data.
`-noTLS` | | Disable the TLS computation. If the TLS module is not installed in your machine, this flag will be automatically used.
`-tofile` | | Save the processed and detrended light curve to an ascii file. The file will have three columns: `time`, `flux` and `flux_err`.
`-tojuliet` | | Save the processed and detrended light curve to an ascii file, that can then be used with juliet. The file will have four columns: `time`, `flux`, `flux_err`, and `instrument`. Have to be used together with the `-tofile` flag.

### Example

Get the light curves for TIC 307210830 and TIC 307210831, use a detrending window of length 2001, and save the final light curves to an ascii file:
`python get_TESS_lightcurves -TIC 307210830,307210831 -window 2001 -tofile`

### Output

The output will consist on a set of .pdf files, one for each target, showing a total of six different plots:

**Plot 1**: Extracted light curve from the TESS archive, with the data separated by sectors, and with the trend detected.

![TESS_p1](https://github.com/msotov/images/blob/master/TESSlc_p1.png)
 
**Plot 2**: Detrended light curve, which will be used to search for transit features.

![Tess_p2](https://github.com/msotov/images/blob/master/TESSlc_p2.png)

**Plots 3 and 4**: BLS periodogram, and folded light curve to best BLS period. In plot 3 the dashed vertical lines are the aliases of the best period. In plot 4, the orange circles define the binned light curve, and the green line is the best BLS model.

![Tess_p3](https://github.com/msotov/images/blob/master/TESSlc_p3.png)

**Plots 5 and 6**: TLS periodogram, and folded light curve to the best TLS period. These two plots are only created if the `-noTLS` flag is not used, and the TLS module is installed in your computer. In plot 5 the dashed vertical lines are the aliases of the best period. In plot 6, the orange circles define the binned light curve, and the green line is the best TLS model.

![Tess_p4](https://github.com/msotov/images/blob/master/TESSlc_p4.png)
