"""
Module to extract light curve data from the NASA TESS Mission,
perform a detrending, and search for transit signals using
the BLS periodogram and the Transit Least Square (TLS) method.

author: Maritza Soto
Date: 2020-09-23

It is used from the command-line, by calling it with the following parameters.

Parameters:
    -TIC            TIC number for each target. Could also be a comma-separated
                    string with all the TIC identifiers.
                    You must specify either this parameter or the '-from_file' one.

    -from_file      Name of the file where the TIC identifiers are stored.
                    You must specify either this parameter or the '-TIC' one.

    -tic_id         Name of the column containing the TIC identifiers.

    [Optional]

    -path_plots     Path where the final plots will be saved.
                    Default is the current directory.

    -interactive    Show the plots created for each star, before going to the next one.

    -no_detrend     Disable the detrending of the data.

    -window         Default is 1501.
                    Length of the window used during the detrending.
                    Must be an odd number.

    -GP             Use Gaussian Processes to do the detrending.
                    Requires that 'juliet' is istalled in your computer.

    -from_fits      If the light curve data is to be read from fits files.

    -path_fits      Path to the fits files with the light curve data.

    -noTLS          Disable the TLS computation.

    -tofile         Save the processed and detrended light curve to an ascii file.

    -tojuliet       Save the processed and detrended light curve to an ascii file,
                        that can then be used with juliet.

Example:

python get_TESS_lightcurves -TIC 307210830,307210831 -GP -tofile

"""

import os
import sys
import warnings
import glob
import argparse
import lightkurve as lk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib
from PyAstronomy.pyasl import binningx0dt
from astropy.io import ascii

rcParams["figure.dpi"] = 150
plt.style.use(['seaborn-muted'])
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

warnings.simplefilter("ignore")

class BLS:
    """
    Class to store the BLS computation output and parameters.
    """
    def __init__(self, pg, period_max, t0_max, depth_max, duration_max):
        self.bls = pg
        self.power = pg.power
        self.period = pg.period
        self.period_max = period_max
        self.t0_max = t0_max
        self.depth_max = depth_max
        self.duration_max = duration_max


class LCObject:
    """
    Main light curve object, which contains all the information from the data,
    and the different processes that are done to finally extract transit
    information.
    """
    def __init__(self, TIC, lc, sectors, nlc):
        self.TIC = TIC
        self.alllc = lc
        self.nlc = nlc
        self.sectors = sectors
        self.nsectors = max(sectors)-min(sectors)+1
        self.stitched = lc.stitch()
        self.lc = self.stitched.copy()
        self.flat = None
        self.trend = None
        self.normalized_no_outlayers = None
        self.BLS = None
        self.tls = None
        self.folded = None

    def _to_use(self, lci):
        self.lc = lci.copy()

    def clean_stitched(self):
        """
        Cleans the data by removing invalid points and outlayers.
        """
        if np.any(np.isnan(self.stitched.flux)):
            lc_stitched_old = self.stitched.copy()
            idx = np.where(~np.isnan(lc_stitched_old.flux))[0]
            lc_stitched = lk.LightCurve(time=lc_stitched_old.time[idx],
                                        flux=lc_stitched_old.flux[idx],
                                        flux_err=lc_stitched_old.flux_err[idx])
            del lc_stitched_old

        self.stitched = lc_stitched.remove_outliers(sigma_lower=float('inf'),
                                                    sigma_upper=5.0).copy()
        del lc_stitched

        self._to_use(self.stitched)

    def _juliet_detrend(self):
        """
        Calls juliet and performs the Gaussian Process on the data.
        """
        try:
            import juliet
        except ModuleNotFoundError:
            print('Juliet is not present. Skipping')
            return np.ones(self.lc.flux.size)

        out_folder = 'TIC%d' % self.TIC

        times, fluxes, fluxes_error = {},{},{}
        times['TESS'], fluxes['TESS'], fluxes_error['TESS'] = self.lc.time+2457000, self.lc.flux,\
                                                              self.lc.flux_err

        # Name of the parameters to be fit:
        params_inst = ['mdilution_TESS', 'mflux_TESS', 'sigma_w_TESS']
        params_gp = ['GP_sigma_TESS', 'GP_timescale_TESS']

        dists_inst = ['fixed', 'normal', 'loguniform']
        dists_gp = ['loguniform', 'loguniform']

        hyperps_inst = [1.0, [0.,0.1], [0.1, 1000.]]
        hyperps_gp = [[1e-6, 1e6], [1e-3, 1e3]]

        params = params_inst
        dists = dists_inst
        hyperps = hyperps_inst

        params += params_gp
        dists += dists_gp
        hyperps += hyperps_gp

        priors = {}
        for param, dist, hyperp in zip(params, dists, hyperps):
            priors[param] = {}
            priors[param]['distribution'], priors[param]['hyperparameters'] = dist, hyperp

        dataset = juliet.load(priors=priors, t_lc = times, y_lc = fluxes,\
                              yerr_lc = fluxes_error, GP_regressors_lc=times,\
                              out_folder = out_folder, verbose=True)
        results = dataset.fit()
        model_fit = results.lc.evaluate('TESS')

        del dataset, results, times, fluxes, fluxes_error
        return model_fit

    def detrend(self, window, detrend, gp_detrend):
        """
        Performs the detrending of the data. It can be done following the .flatten()
        function from lightkurve, or by using Gaussian Processes through juliet.
        It is also possible to disable the detrending.
        """
        self.flat, self.trend = self.lc.flatten(window_length=window, return_trend=True, niters=3)
        self._to_use(self.flat)
        if detrend:
            self.normalized_no_outlayers = self.lc.remove_outliers(sigma_lower=float('inf'),
                                                                   sigma_upper=5.0)
            self._to_use(self.normalized_no_outlayers)
        else:
            self._to_use(self.stitched)

        if gp_detrend:
            self._to_use(self.stitched)
            gp_trend = self._juliet_detrend()
            if not np.all(gp_trend == 1.0):
                lc_no_gp = lk.LightCurve(time=self.lc.time, flux=self.lc.flux/gp_trend,
                                         flux_err=self.lc.flux_err/gp_trend)
                self.normalized_no_outlayers = lc_no_gp.remove_outliers(sigma_lower=float('inf'),
                                                                        sigma_upper=5.0)
                self.trend.flux = gp_trend
                os.system('rm -f TIC%d/jomnest*' % self.TIC)
                os.system('rm -f TIC%d/GP_lc_regressors.dat' % self.TIC)
                os.system('rm -f TIC%d/lc.dat' % self.TIC)

                self._to_use(self.normalized_no_outlayers)

                del lc_no_gp
            del gp_trend

    def save_to_file(self, tojuliet):
        """
        Saves the processed and detrended data to an ascii file.
        The data can also be saved for latter use with juliet, including an extra column
        with the name of the instrument used to collect the data (in this case, TESS).
        """
        if self.lc.time[0] < 1e4:
            self.lc.time += 2457000
        ascii.write([self.lc.time, self.lc.flux, self.lc.flux_err], 'TIC%d.dat' % self.TIC,
                     format='fixed_width_no_header', delimiter=' ', overwrite=True)
        if tojuliet:
            ascii.write([self.lc.time, self.lc.flux, self.lc.flux_err,
                         ['TESS' for _ in self.lc.time]], 'TIC%d_juliet.dat' % self.TIC,
                         format='fixed_width_no_header', delimiter=' ', overwrite=True)


    def BLS_search(self):
        """
        Performs a BLS search on the data, and saves the model in a BLS object.
        """
        duration_transit = np.linspace(0.02, 0.1, 10)
        pg = self.lc.to_periodogram("bls", frequency_factor=8*self.nsectors**2,
                                    minimum_period=0.15, duration=duration_transit)
        periods = pg.period.value
        transit_times = pg.transit_time
        power = pg.power
        duration = pg.duration.value
        depth = pg.depth

        i = np.where(periods > 0.70)[0]
        period_max = periods[i][np.argmax(power[i])]
        t0_max = transit_times[i][np.argmax(power[i])]
        depth_max = depth[i][np.argmax(power[i])]
        duration_max = duration[i][np.argmax(power[i])]

        self.BLS = BLS(pg, period_max, t0_max, depth_max, duration_max)
        self.folded = self.lc.fold(period_max, t0=t0_max)

        del periods, transit_times, power, depth, i, duration, pg


    @staticmethod
    def _get_colors(N):
        cmap = matplotlib.cm.get_cmap('viridis')
        n = np.linspace(0.0, 1.0, N)
        c = [cmap(x) for x in n]
        return c


    def plot(self, noTLS, path_plots, interactive):
        """
        Creates the final plot for the target.
        """
        fig = plt.figure(figsize=(10,12))
        ax1 = fig.add_subplot(4, 1, 1)
        ax2 = fig.add_subplot(4, 1, 2)
        ax3 = fig.add_subplot(4, 2, 5)
        ax4 = fig.add_subplot(4, 2, 6)
        ax5 = fig.add_subplot(4, 2, 7)
        ax6 = fig.add_subplot(4, 2, 8)

        # First panel: data from each sector
        colors = self._get_colors(self.nlc)
        for i, lci in enumerate(self.alllc):
            p = lci.normalize().remove_outliers(sigma_lower=5.0, sigma_upper=5.0)
            p.bin(5).scatter(ax=ax1, label='Sector %d' % self.sectors[i], color=colors[i])
        self.trend.plot(ax=ax1, color='orange', lw=2, label='Trend')
        ax1.legend(fontsize='small', ncol=4)

        # Second panel: Detrended light curve
        self.lc.remove_outliers(sigma_lower=5.0, sigma_upper=5.0).bin(5).scatter(ax=ax2,
                                                                                 color='black',
                                                                                 label='Detrended')

        # Third panel: BLS
        self.BLS.bls.plot(ax=ax3, label='_no_legend_', color='black')
        mean_SR = np.mean(self.BLS.power)
        std_SR = np.std(self.BLS.power)
        best_power = self.BLS.power[np.where(self.BLS.period.value == self.BLS.period_max)[0]]
        SDE = (best_power - mean_SR)/std_SR
        ax3.axvline(self.BLS.period_max, alpha=0.4, lw=4)
        for n in range(2, 10):
            if n*self.BLS.period_max <= max(self.BLS.period.value):
                ax3.axvline(n*self.BLS.period_max, alpha=0.4, lw=1, linestyle="dashed")
            ax3.axvline(self.BLS.period_max / n, alpha=0.4, lw=1, linestyle="dashed")
        sx, ex = ax3.get_xlim()
        sy, ey = ax3.get_ylim()
        ax3.text(ex-(ex-sx)/3, ey-(ey-sy)/3,
                 'P$_{MAX}$ = %.3f d\nT0 = %.2f\nDepth = %.4f\nDuration = %.2f d\nSDE = %.3f' %
                 (self.BLS.period_max, self.BLS.t0_max,
                  self.BLS.depth_max, self.BLS.duration_max, SDE))


        # Fourth panel: lightcurve folded to the best period from the BLS
        self.folded.bin(1*self.nlc).scatter(ax=ax4, label='_no_legend_', color='black',
                                            marker='.', alpha=0.5)
        l = max(min(4*self.BLS.duration_max/self.BLS.period_max, 0.5), 0.02)
        nbins = int(50*0.5/l)
        r1, dt1 = binningx0dt(self.folded.phase, self.folded.flux, x0=-0.5, nbins=nbins)
        ax4.plot(r1[::,0], r1[::,1], marker='o', ls='None',
                 color='orange', markersize=5, markeredgecolor='orangered', label='_no_legend_')

        lc_model = self.BLS.bls.get_transit_model(period=self.BLS.period_max,
                                                  duration=self.BLS.duration_max,
                                                  transit_time=self.BLS.t0_max)
        lc_model_folded = lc_model.fold(self.BLS.period_max, t0=self.BLS.t0_max)
        ax4.plot(lc_model_folded.phase, lc_model_folded.flux, color='green', lw=2)
        ax4.set_xlim(-l, l)
        h = max(lc_model.flux)
        l = min(lc_model.flux)
        ax4.set_ylim(l-4.*(h-l), h+5.*(h-l))
        del lc_model, lc_model_folded, r1, dt1


        if not noTLS:
            # Fifth panel: TLS periodogram
            ax5.axvline(self.tls.period, alpha=0.4, lw=3)
            ax5.set_xlim(np.min(self.tls.periods), np.max(self.tls.periods))
            for n in range(2, 10):
                ax5.axvline(n*self.tls.period, alpha=0.4, lw=1, linestyle="dashed")
                ax5.axvline(self.tls.period / n, alpha=0.4, lw=1, linestyle="dashed")
            ax5.set_ylabel(r'SDE')
            ax5.set_xlabel('Period (days)')
            ax5.plot(self.tls.periods, self.tls.power, color='black', lw=0.5)
            ax5.set_xlim(0, max(self.tls.periods))

            period_tls = self.tls.period
            T0_tls = self.tls.T0
            depth_tls = self.tls.depth
            duration_tls = self.tls.duration
            FAP_tls = self.tls.FAP

            sx, ex = ax5.get_xlim()
            sy, ey = ax5.get_ylim()
            ax5.text(ex-(ex-sx)/3, ey-(ey-sy)/3,
                     'P$_{MAX}$ = %.3f d\nT0 = %.1f\nDepth = %.4f\nDuration = %.2f d\nFAP = %.4f' %
                                        (period_tls, T0_tls, 1.-depth_tls, duration_tls, FAP_tls))

            # Sixth panel: folded light curve to the best period from the TLS
            ax6.plot(self.tls.folded_phase, self.tls.folded_y, color='black', marker='.',
                     alpha=0.5, ls='None', markersize=0.7)
            l = max(min(4*duration_tls/period_tls, 0.5), 0.02)
            nbins = int(50*0.5/l)
            r1, dt1 = binningx0dt(self.tls.folded_phase, self.tls.folded_y,
                                  x0=0.0, nbins=nbins, useBinCenter=True)
            ax6.plot(r1[::,0], r1[::,1], marker='o', ls='None', color='orange',
                     markersize=5, markeredgecolor='orangered', label='_no_legend_')
            ax6.plot(self.tls.model_folded_phase, self.tls.model_folded_model, color='green', lw=2)
            ax6.set_xlim(0.5-l, 0.5+l)
            h = max(self.tls.model_folded_model)
            l = min(self.tls.model_folded_model)
            ax6.set_ylim(l-4.*(h-l), h+5.*(h-l))
            ax6.set_xlabel('Phase')
            ax6.set_ylabel('Relative flux')
            del r1, dt1

        fig.subplots_adjust(top=0.98, bottom=0.05, wspace=0.25, left=0.1, right=0.97)
        fig.savefig(os.path.join(path_plots, 'TIC%d.pdf' % self.TIC))
        if interactive:
            plt.show()
        plt.close('all')
        del fig


def get_lk(TIC, path_plots='.', interactive=False, detrend=True, window=1501,
           gp_detrend=False, from_fits=False, path_fits=None, noTLS=False, tofile=False,
           tojuliet=False):
    """
    Performs the computation for each star.
    """

    download = True

    if from_fits and path_fits:
        nfiles = glob.glob(os.path.join(path_fits, 'tess*%d*.fits' % TIC))
        lc = []
        for f in nfiles:
            lci = lk.search.open(f)
            lc.append(lci)
        lc = lk.collections.LightCurveFileCollection(lc)
        nlc = len(lc)
        if nlc > 0:
            download = False

    if download:
        lc = lk.search_lightcurvefile("TIC %d" % TIC, mission='TESS').download_all()
        if lc is None:
            lc = lk.search_lightcurvefile("TIC %d" % TIC, mission='TESS').download()
            if lc is None:
                print('\tCould not find TESS data for TIC %d' % TIC)
                return 0
            lc = [lc]

    nlc = len(lc)
    sectors = []
    for lci in lc:
        if lci.header()['TICID'] != TIC:
            print('\tINCORRECT TIC')
            print('\tFound lc for TIC %d' % lci.header()['TICID'])
            print('\tStopping the computation.')
            return 0
        sectors.append(lci.header()['SECTOR'])

    print('\tFound %d light curve(s) from sector(s) %s' % (nlc,
                                                           ", ".join(np.array(sectors,
                                                                              dtype='str'))))
    lc = lc.PDCSAP_FLUX

    LC = LCObject(TIC, lc, sectors, nlc)
    LC.clean_stitched()
    LC.detrend(window, detrend, gp_detrend)

    if tofile:
        LC.save_to_file(tojuliet)

    LC.BLS_search()

    # Period search with TLS
    if not noTLS:
        try:
            from transitleastsquares import transitleastsquares
            model = transitleastsquares(LC.lc.time, LC.lc.flux)
            LC.tls = model.power(oversampling_factor=max(1, int(5/LC.nsectors)),
                                 duration_grid_step=1.02)
            del model
        except ModuleNotFoundError:
            print('transitleastsquares is not present. Skipping TLS computation.')
            noTLS = True

    LC.plot(noTLS, path_plots, interactive)

    del LC, lc

    return 0

def main():
    """
    Module to extract light curve data from the NASA TESS Mission,
    perform a detrending, and search for transit signals using
    the BLS periodogram and the Transit Least Square (TLS) method.

    It is used from the command-line, by calling it with the
    following parameters.

    Parameters:
        -TIC            TIC number for each target. Could also be a comma-separated
                        string with all the TIC identifiers.
                        You must specify either this parameter or the '-from_file' one.

        -from_file      Name of the file where the TIC identifiers are stored.
                        You must specify either this parameter or the '-TIC' one.

        -tic_id         Name of the column containing the TIC identifiers.

        [Optional]

        -path_plots     Path where the final plots will be saved.
                        Default is the current directory.

        -interactive    Show the plots created for each star, before going to the next one.

        -no_detrend     Disable the detrending of the data.

        -window         Default is 1501.
                        Length of the window used during the detrending.
                        Must be an odd number.

        -GP             Use Gaussian Processes to do the detrending.
                        Requires that 'juliet' is istalled in your computer.

        -from_fits      If the light curve data is to be read from fits files.

        -path_fits      Path to the fits files with the light curve data.

        -noTLS          Disable the TLS computation.

        -tofile         Save the processed and detrended light curve to an ascii file.

        -tojuliet       Save the processed and detrended light curve to an ascii file,
                        that can then be used with juliet.

    Example:

    python get_TESS_lightcurves -TIC 307210830,307210831 -GP -tofile

    """

    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('-TIC', default=307210830, type=str)
    PARSER.add_argument('-from_file', default='any')
    PARSER.add_argument('-tic_id', default='TIC_ID')
    PARSER.add_argument('-path_plots', default='.')
    PARSER.add_argument('-interactive', action='store_true', default=False)
    PARSER.add_argument('-no_detrend', action='store_true', default=False)
    PARSER.add_argument('-window', default=1501, type=int)
    PARSER.add_argument('-GP', default=False, action='store_true')
    PARSER.add_argument('-from_fits', default=False, action='store_true')
    PARSER.add_argument('-path_fits', default=None)
    PARSER.add_argument('-noTLS', default=False, action='store_true')
    PARSER.add_argument('-tofile', default=False, action='store_true')
    PARSER.add_argument('-tojuliet', default=False, action='store_true')

    ARGS = PARSER.parse_args()

    if ARGS.from_file != 'any':
        TICs = []
        if os.path.isfile(ARGS.from_file):
            try:
                TICs = ascii.read(ARGS.from_file)[ARGS.tic_id]
            except KeyError:
                print('Column %s does not exist' % ARGS.tic_id)
        else:
            print('Filename does not exist')

    else:
        TICs = [float(i) for i in ARGS.TIC.split(',')]

    detrend = not ARGS.no_detrend
    path_plots = ARGS.path_plots
    interactive = ARGS.interactive
    window = ARGS.window
    gp = ARGS.GP
    from_fits = ARGS.from_fits
    path_fits = ARGS.path_fits
    noTLS = ARGS.noTLS
    tofile = ARGS.tofile
    tojuliet = ARGS.tojuliet

    if len(TICs) > 0:
        for i, TIC in enumerate(TICs):
            try:
                print('TIC %d (%d/%d)' % (TIC, i+1, len(TICs)))
                _ = get_lk(TIC, path_plots=path_plots, detrend=detrend, interactive=interactive,
                           window=window, gp_detrend=gp, from_fits=from_fits, path_fits=path_fits,
                           noTLS=noTLS, tofile=tofile, tojuliet=tojuliet)
            except Exception as e:
                print('Problem with object TIC %d' % TIC)
                _, _, exc_tb = sys.exc_info()
                print('line %d: %s' % (exc_tb.tb_lineno, e))

if __name__ == "__main__":
    main()
