import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

def gauss(p, x) -> np.ndarray:
    """
    Produce a Gaussian function.
    
    Parameters
    ----------
    p : array_like
        Coefficients of the Gaussian function.
    x : array_like
        The x range over which the Gaussian function is to be produced.

    Returns
    -------
    numpy.ndarray
        The Gaussian function.
    """
    return p[0] * np.exp(-(x - p[1])**2 / (2 * p[2]**2)) + p[3]

def residuals(p, func, x, y, s=1) -> np.ndarray:
    """
    Find the residuals from fitting data to a function.
    
    Parameters
    ----------
    p : array_like
        Coefficients of the function func.
    x : array_like
        The x data to be fit.
    y : array_like
        The y data to be fit.
    func : callable
        The function to be fit.
    s : float, default 1
        The standard deviation.

    Returns
    -------
    numpy.ndarray
        The residuals of the fit function.
    """
    return (y - func(p, x)) / s

def fitting(p, x, y, func, s=1):
    """
    Fit data to a function.
    
    Parameters
    ----------
    p : array_like
        Initial guess at the values of the coefficients to be fit
    x : array_like
        The x data to be fit.
    y : array_like
        The y data to be fit.
    func : callable
        The function to fit the data to.
    s : float, default 1
        The standard deviation.

    Returns
    -------
    p_fit : numpy.ndarray
        The array of the coefficients after fitting the function to the data.
    """
    # Fit the data and find the uncertainties
    r = opt.least_squares(residuals, p, args=(func, x, y, s))
    p_fit = r.x
    hessian = np.dot(r.jac.T, r.jac) #estimate the hessian matrix
    K_fit = np.linalg.inv(hessian) #covariance matrix
    unc_fit = np.sqrt(np.diag(K_fit)) #stdevs
    
    return p_fit

def round_sig_fig_uncertainty(value, uncertainty):
    """
    Round to the first significant figure of the uncertainty.
    
    Parameters
    ----------
    value : float or array_like
        The value(s) to be rounded.
    uncertainty : float or array_like
        The uncertaint(y/ies) in this value, must be the same size as value.

    Returns
    -------
    value_out : numpy.ndarray or float
        The rounded array of values.
    uncertainty_out : numpy.ndarray or float
        The rounded array of uncertainties.

    See Also
    --------
    round_sig_fig : Round to a given number of significant figures.
    """
    # check if numpy array/list or float/int
    if isinstance(value, np.ndarray) or isinstance(value, list):
        value_out = np.array([])
        uncertainty_out = np.array([])
        for i in range(len(value)):
            # Check if some of the values are 0
            if uncertainty[i] == 0:
                value_out = np.append(value_out, value[i])
                uncertainty_out = np.append(uncertainty_out, uncertainty[i])
            # Check if the leading digit in the error is 1, and if so round to an extra significant figure
            elif np.floor(uncertainty[i] / (10**np.floor(np.log10(uncertainty[i])))) != 1.0:
                uncertainty_rnd = np.round(uncertainty[i], int(-(np.floor(np.log10(uncertainty[i])))))
                value_rnd = np.round(value[i], int(-(np.floor(np.log10(uncertainty_rnd)))))

                value_out = np.append(value_out, value_rnd)
                uncertainty_out = np.append(uncertainty_out, uncertainty_rnd)
            else:
                uncertainty_rnd = np.round(uncertainty[i], int(1 - (np.floor(np.log10(uncertainty[i])))))
                value_rnd = np.round(value[i], int(1 - (np.floor(np.log10(uncertainty_rnd)))))

                value_out = np.append(value_out, value_rnd)
                uncertainty_out = np.append(uncertainty_out, uncertainty_rnd)
        return value_out, uncertainty_out
   
    elif isinstance(value, float) or isinstance(value, int):
        if uncertainty == 0:
            return value, uncertainty
        elif np.floor(uncertainty / (10**np.floor(np.log10(uncertainty)))) != 1.0:
            uncertainty_out = np.round(uncertainty, int(-(np.floor(np.log10(uncertainty)))))
            value_out = np.round(value, int(-(np.floor(np.log10(uncertainty_out)))))

            return value_out, uncertainty_out
        else:
            uncertainty_out = np.round(uncertainty, int(1 - (np.floor(np.log10(uncertainty)))))
            value_out = np.round(value, int(1 - (np.floor(np.log10(uncertainty_out)))))

            return value_out, uncertainty_out
    else:
        return value, uncertainty
    
def fit_line_red_plotting(wavelength, intensity, line, name=None, amplitude=0.5, centre=None, std=1, y_shift=0.5, start=0, end=200):
    """
    Fits and plots a Gaussian curve to a spectral line which has been redshifted.
    
    Parameters
    ----------
    wavelength : array_like
        The wavelength data to fit.
    Intensity : array_like
        The intensity data to fit.
    Line : float
        The wavelength of the spectral line.
    Name : string, default None
        The name associated with the spectrum.
    y_shift : float, default 0.5
        The initial guess at the contiuum.
    centre : float, default None
        The initial guess at the peak wavelength when redshifted, if left empty the point at which the intensuty is highest is taken as the peak.
    amplitude : float, default 0.5
        The initial guess at the amplitude of the line.
    std : float, default 1
        The initial guess at the standard deviation of the peak.
    start : float defaul 0
        The number of wavelength units beyond the at rest peak to start fitting.
    end : float defaul 300
        The number of wavelength units beyond the at rest peak to stop fitting.
    
    See Also
    --------
    find_redshift : Fits a Gaussian curve to a spectral line which has been redshifted and calculates the redshift.
    """
    # Limit to fitting at wavelengths longer than the line's to [end] longer
    index = (wavelength >= line + start) * (wavelength <= line + end)
    wavelength = wavelength[index]
    intensity = intensity[index]
    
    # Normalise the intensity
    intensity = intensity / np.amax(intensity)
    
    # Array for plotting the guess and fit
    wl = np.linspace(min(wavelength), max(wavelength), 1000)

    if centre == None:
        # Guess at where the peak is
        centre = np.argmax(intensity)
    else:
        centre = find_nearest_index(centre, wavelength)
    
    p0 = [amplitude, wavelength[centre], std, y_shift]

    p_fit = fitting(p0, wavelength, intensity, gauss)

    plt.plot(wavelength, intensity, label='Spectrum')
    plt.axvline(line, c='red', ls='--', label='Spectral line of interest\n(no redshift)')
    if name != None:
        plt.title(f'Host galaxy and supernova spectrum\nfor {name} with a spectral line fitted')
        
    plt.xlabel(r'$\lambda$ [$\AA$]')
    plt.ylabel('I')

    # Plot guess and fit
    plt.plot(wl, gauss(p0, wl), label='guess')
    plt.plot(wl, gauss(p_fit, wl), label='fit')
    plt.legend()
    plt.grid()
    plt.show()

def find_redshift(wavelength, intensity, line, y_shift=0.5, centre=None, amplitude=0.5, std=1, start=0, end=300):
    """
    Fits a Gaussian curve to a spectral line which has been redshifted and calculates the redshift.
    
    Parameters
    ----------
    wavelength : array_like
        The wavelength data to fit.
    Intensity : array_like
        The intensity data to fit.
    Line : float
        The wavelength of the spectral line.
    Name : string
        The name associated with the spectrum.
    y_shift : float
        The initial guess at the contiuum.
    centre : float, default None
        The initial guess at the peak wavelength when redshifted, if left empty the point at which the intensuty is highest is taken as the peak.
    amplitude : float, default 0.5
        The initial guess at the amplitude of the line.
    std : float, default 1
        The initial guess at the standard deviation of the peak.
    start : float defaul 0
        The number of wavelength units beyond the at rest peak to start fitting.
    end : float defaul 300
        The number of wavelength units beyond the at rest peak to stop fitting.
    
    Returns
    -------
    z : float
        The calculated redshift.
    z_unc : float
        The associated uncertainty with the value of z (FWHM of the peak / line wavelength at rest).
    
    See Also
    --------
    fit_line_red_plotting : Fits and plots a Gaussian curve to a spectral line which has been redshifted.
    """
    # Limit to fitting at wavelengths longer than the line's to [end] longer
    index = (wavelength >= line + start) * (wavelength <= line + end)
    wavelength = wavelength[index]
    intensity = intensity[index]
    
    # Normalise the intensity
    intensity = intensity / np.amax(intensity)

    if centre == None:
        # Guess at where the peak is
        centre = np.argmax(intensity)
    else:
        centre = find_nearest_index(centre, wavelength)
    
    p0 = [amplitude, wavelength[centre], std, y_shift]

    # Fit
    p_fit = fitting(p0, wavelength, intensity, gauss)
    
    # Save the peak and calculate z
    peak = p_fit[1]
    z = (peak - line)/line

    # Take FWHM as uncertainty in peak position
    FWHM = 2 * np.sqrt(2*np.log(2)) * p_fit[2]
    
    z_unc = np.abs(FWHM/line)
    
    return z, z_unc

def find_nearest_index(value, array):
    """
    Find the index of the value in the array closest to the inport value.
    
    Parameters
    ----------
    value : float
        The value to search for.
    array : array_like
        The array to search.
        
    Returns
    -------
    index : int
        The index of the closest value in the array to value.
    """
    index = np.searchsorted(array, value, side="left")
    return index

def deltaBm15_plotting(time, B, B_unc, degree=6, name=None):
    """
    Finds the value of Delta m_15 (B) for a supernova light curve.

    Parameters
    ----------
    time : array_like
        The time series data.
    B : array_like
        The intensity data.
    B_unc : array_like
        The uncertainty in the intensity data.
    Degree : int default 6
        The degree of the polynomial to be fit
    Name : string
        The name of the supernova light curve.

    See Also
    --------
    deltaBm15 : Finds and returns the value of Delta m_15 (B) for a supernova light curve.
    """

    plt.errorbar(time, B, B_unc, fmt='.', label='Data')
    plt.gca().invert_yaxis()
    plt.title(f'The light curve of {name}')
    plt.ylabel('Magnitude')
    plt.xlabel('Time')

    t = np.linspace(np.amin(time), np.amax(time), 10000)

    # Fitting
    fit, cov = np.polyfit(time, B, degree, cov=True, w=1/B_unc)
    fit_curve = np.poly1d(fit)

    # Only look for peak in the first 30 days (which occurs for all light curves)
    ind = t < 30

    peak_loc = np.argmin(fit_curve(t)[ind])

    peak = fit_curve(t)[peak_loc]
    peak_time = t[peak_loc]

    m15 = np.polyval(fit, peak_time + 15)
    
    # Plot fit parameters
    plt.plot(t, fit_curve(t), label='Polynomial fit')
    
    plt.axhline(peak, color='red', ls='--', label=f'Light curve peak')
    plt.axhline(m15, color='black', ls='--', label='Magnitude after 15 days')

    plt.axvline(peak_time, color='red', ls=':', label='Peak in light curve')
    plt.axvline(peak_time+15, color='black', ls=':', label='15 days after peak')

    plt.legend()
    plt.grid()
    plt.show()

def deltaBm15(time, B, B_unc, degree=6):
    """
    Finds and returns the peak value and the value of Delta m_15 (B) for a supernova light curve and their uncertainties.
    
    Parameters
    ----------
    time : array_like
        The time series data.
    B : array_like
        The intensity data.
    B_unc : array_like
        The uncertainty in the intensity data.
    Degree : int default 6
        The degree of the polynomial to be fit
    
    Returns
    -------
    peak : float
        The peak intensity.
    del_m15 : float
        The value of Delta m_15 (B).
    peak_unc : float
        The uncertainty in the peak intensity.
    del_m15_unc : float
        The uncertainty in the value of Delta m_15 (B).

    See Also
    --------
    deltaBm15 : Finds and returns the peak value of and the value of Delta m_15 (B) for a supernova light curve.
    deltaBm15_plotting : Finds the value of Delta m_15 (B) for a supernova light curve and plots the data.
    """

    t = np.linspace(np.amin(time), np.amax(time), 100000)
    
    # Fitting
    fit, cov = np.polyfit(time, B, degree, cov=True, w=1/B_unc)
    fit_curve = np.poly1d(fit)

    # Uncertainty in the polynomial coefficients
    p_unc = np.sqrt(np.diag(cov))

    # Only look for peak in the first 30 days (which occurs for all light curves)
    ind = t < 30

    # find the location of the peak
    peak_loc = np.argmin(fit_curve(t)[ind])

    # Find the peak and the time it occurs
    peak_time = t[peak_loc]
    peak = np.polyval(fit, peak_time)

    # Find uncertainty in the peak
    peak_unc = 0
    for i in range(len(p_unc)):
        peak_unc += p_unc[i]**2 * peak_time**(degree - i)
    peak_unc = np.sqrt(peak_unc)

    # Find magnitude after 15 days
    m15 = np.polyval(fit, peak_time + 15)

    # Find uncertainty in magnitude after 15 days
    m15_unc = 0
    for i in range(len(p_unc)):
        m15_unc += p_unc[i]**2 * (peak_time + 15)**(degree - i)
    m15_unc = np.sqrt(m15_unc)

    # Find difference after 15 days and associated uncertatinty
    del_m15 = np.abs(peak - m15)
    del_m15_unc = np.sqrt(peak_unc**2 + m15_unc**2)
    
    return peak, del_m15, peak_unc, del_m15_unc

def residuals_data(observed, expected, s=1) -> np.ndarray:
    """
    Find the residuals between observed data and a model.
    
    Parameters
    ----------
    observed : array_like
        The observed data.
    expected : array_like
        The expected values for the data from the model.
    s : array_like, default 1
        The uncertainty in the data.

    Returns
    -------
    numpy.ndarray
        The residuals of the fit function.
    """
    return (observed - expected) / s

def chi2(observed, expected, s=1) -> float:
    """
    Find the chi squared value for a set of data and expected values.

    Parameters
    ----------
    observed : array_like
        The observed data.
    expected : array_like
        The expected values for the data.
    s : array_like, default 1
        The uncertainty in the data.

    Returns
    -------
    float
        The chi squared value for the dataset.
    """
    return np.sum(residuals_data(observed, expected, s)**2 ) 

