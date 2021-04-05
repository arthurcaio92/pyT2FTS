from numpy import exp, ones_like, zeros_like, arange, multiply, \
     subtract, add, minimum, maximum, sign, c_, argmax, \
     array, where
from numpy import sum as npsum
import matplotlib.pyplot as plt
from math import isclose
from itertools import product
from numpy import linspace
import numpy as np

"""

OBS: Se der erro float division by zero

O problema está na funcao config_inicial. O numero de pontos está dando zero porque
dominio_sup - dominio_inf é um float tipo 0,16. Ai arrendonda para zero. Corrige isso
tirando a exigencia de ser inteiro da variavel 'pontos'
"""




def zero_mf(x, params=None):
    """
    All zero membership function.
    
    Parameters
    ----------
    x : 
        numpy (n,) shaped array
        
        The array like input x indicates the points from universe of 
        discourse in which the membership function would be evaluated.
    params : 
        List 
        
        Additional parameters for the membership function, which is not 
        needed for zero memebership function.
    
    Returns
    -------
    ndarray
        Returns a vector of membership values corresponding with the input.
    
    Examples
    --------
    
    >>> x = linspace(0, 1, 201)
    >>> membership_value = zero_mf(x)
    """
    return zeros_like(x)


def singleton_mf(x, params):
    """
    Singleton membership function.
    
    Parameters
    ----------
    x : 
        numpy (n,) shaped array
        
        The array like input x indicates the points from universe of 
        discourse in which the membership function would be evaluated.
    params : 
        List 
        
        Additional parameters for the membership function. params[0] indicates
        singleton center and params[1] indicates singleton height.
    
    Returns
    -------
    ndarray
        Returns membership values corresponding with the input.
        
    Notes
    -----
    The singleton center, params[0], must be in the discretized universe 
    of discourse.
    
    Examples
    --------
    
    >>> x = linspace(0, 1, 201)
    >>> membership_value = singleton_mf(x, [0.5, 1])
    """
    return multiply(params[1], x == params[0])


def const_mf(x, params):
    """
    Constant membership function.
    
    Parameters
    ----------
    x : 
        numpy (n,) shaped array
        
        The array like input x indicates the points from universe of 
        discourse in which the membership function would be evaluated.
    params : 
        List 
        
        Additional parameters for the membership function. params[0] indicates
        constant membership function's height.
    
    Returns
    -------
    ndarray
        Returns membership values corresponding with the input.
    
    Examples
    --------
    
    >>> x = linspace(0, 1, 201)
    >>> membership_value = const_mf(x, [0.5])
    """
    return multiply(params[0], ones_like(x))


def tri_mf(x, params):
    """
    Triangular membership function.

    Parameters
    ----------
    x : 
        numpy (n,) shaped array
        
        The array like input x indicates the points from universe of 
        discourse in which the membership function would be evaluated.
    params : 
        List 
        
        Additional parameters for the membership function. The left end, 
        the center, the right end, and the height of the triangular 
        membership function is indicated by params[0], params[1], params[2], 
        and params[3], respectively.
    
    Returns
    -------
    ndarray
        Returns membership values corresponding with the input.
    
    Examples
    --------
    
    >>> x = linspace(0, 1, 201)
    >>> membership_value = tri_mf(x, [0.1, 0.3, 0.5, 1])
    
    
    Observação: Codigo original aqui em baixo:
    return minimum(1, maximum(0, ((params[3] * (x - params[0]) / (params[1] - params[0])) * (x <= params[1]) + \
               ((params[3] * ((params[2] - x) / (params[2] - params[1]))) * (x > params[1]))) ))
    
    Mudei para tratar a excecao do erro da divisao por zero:
        
    try:
        res = minimum(1, maximum(0, ((params[3] * (x - params[0]) / (params[1] - params[0])) * (x <= params[1]) + \
               ((params[3] * ((params[2] - x) / (params[2] - params[1]))) * (x > params[1]))) ))
    except ZeroDivisionError:
        res = 0
    """
    
    #print(params)

    return minimum(1, maximum(0, ((params[3] * (x - params[0]) / (params[1] - params[0])) * (x <= params[1]) + \
               ((params[3] * ((params[2] - x) / (params[2] - params[1]))) * (x > params[1]))) ))
    
    
    #return res
        
        



def rtri_mf(x, params):
    """
    Right triangular membership function.

    Parameters
    ----------
    x : 
        numpy (n,) shaped array
        
        The array like input x indicates the points from universe of 
        discourse in which the membership function would be evaluated.
    params : 
        List 
        
        Additional parameters for the membership function.  
        The right end, the center, and the height of the triangular 
        membership function is indicated by params[0], params[1], and params[2], 
        respectively.
    
    Returns
    -------
    ndarray
        Returns membership values corresponding with the input.
    
    Examples
    --------
    
    >>> x = linspace(0, 1, 201)
    >>> membership_value = ltri_mf(x, [0.5, 0.2, 1])
    """
    return minimum(1, maximum(0, (params[2] * (x <= params[1]) + \
               ((params[2] * ((params[0] - x) / (params[0] - params[1]))) * (x > params[1]))) ))

def ltri_mf(x, params):
    """
    Left triangular membership function.

    Parameters
    ----------
    x : 
        numpy (n,) shaped array
        
        The array like input x indicates the points from universe of 
        discourse in which the membership function would be evaluated.
    params : 
        List 
        
        Additional parameters for the membership function. The left end, 
        the center, and the height of the triangular 
        membership function is indicated by params[0], params[1] and params[2], 
        respectively.
    
    Returns
    -------
    ndarray
        Returns membership values corresponding with the input.
    
    Examples
    --------
    
    >>> x = linspace(0, 1, 201)
    >>> membership_value = rtri_mf(x, [0.3, 0.5, 1])
    """
    return minimum(1, maximum(0, ((params[2] * (x - params[0]) / (params[1] - params[0])) * (x <= params[1]) + \
               (params[2] * (x > params[1])) )))
    

def trapezoid_mf(x, params):
    """
    Trapezoidal membership function

    Parameters
    ----------
    x : 
        numpy (n,) shaped array
        
        The array like input x indicates the points from universe of 
        discourse in which the membership function would be evaluated.
    params : 
        List 
        
        Additional parameters for the membership function. The left end, 
        the left center, the right center, the right end, and the height 
        of the trapezoidal membership function is indicated by params[0], 
        params[1], params[2], params[3], and params[4], respectively.
    
    Returns
    -------
    ndarray
        Returns membership values corresponding with the input.
    
    Examples
    --------
    
    >>> x = linspace(0, 1, 201)
    >>> membership_value = trapezoid_mf(x, [0.1, 0.3, 0.5, 0.7, 1])
    """
    return minimum(1, maximum(0, ((((params[4] * ((x - params[0]) / (params[1] - params[0]))) * (x <= params[1])) +
                   ((params[4] * ((params[3] - x) / (params[3] - params[2]))) * (x >= params[2]))) +
               (params[4] * ((x > params[1]) * (x < params[2])))) ))


def gaussian_mf(x, params):
    """
    Gaussian membership function
    
    Parameters
    ----------
    x : 
        numpy (n,) shaped array
        
        The array like input x indicates the points from universe of 
        discourse in which the membership function would be evaluated.
    params : 
        List 
        
        Additional parameters for the membership function. The center, 
        the standard deviation, and the height 
        of the gaussian membership function is indicated by params[0], 
        params[1], and params[2], respectively.
    
    Returns
    -------
    ndarray
        Returns membership values corresponding with the input.
    
    Examples
    --------
    
    >>> x = linspace(0, 1, 201)
    >>> membership_value = gaussian_mf(x, [0.5, 0.05, 1])
    """
    return params[2] * exp(-(((params[0] - x) ** 2) / (2 * params[1] ** 2)))


def gauss_uncert_mean_umf(x, params):
    """
    Gaussian with uncertain mean UMF
    
    Parameters
    ----------
    x : 
        numpy (n,) shaped array
        
        The array like input x indicates the points from universe of 
        discourse in which the membership function would be evaluated.
    params : 
        List 
        
        Additional parameters for the membership function. The lower limit 
        of mean, the upper limit of mean, the standard deviation, and the 
        height of the gaussian membership function is indicated by params[0], 
        params[1], params[2], and params[3], respectively.
    
    Returns
    -------
    ndarray
        Returns membership values corresponding with the input.
    
    Examples
    --------
    
    >>> x = linspace(0, 1, 201)
    >>> membership_value = gauss_uncert_mean_umf(x, [0.3, 0.7, 0.05, 1])
    """
    return (((gaussian_mf(x, [params[0], params[2], params[3]]) * (x <= params[0])) +
             (gaussian_mf(x, [params[1], params[2], params[3]]) * (x >= params[1]))) +
               (params[3] * ((x > params[0]) * (x < params[1]))))


def gauss_uncert_mean_lmf(x, params):
    """
    Gaussian with uncertain mean LMF
    
    Parameters
    ----------
    x : 
        numpy (n,) shaped array
        
        The array like input x indicates the points from universe of 
        discourse in which the membership function would be evaluated.
    params : 
        List 
        
        Additional parameters for the membership function. The lower limit 
        of mean, the upper limit of mean, the standard deviation, and the 
        height of the gaussian membership function is indicated by params[0], 
        params[1], params[2], and params[3], respectively.
    
    Returns
    -------
    ndarray
        Returns membership values corresponding with the input.
    
    Examples
    --------
    
    >>> x = linspace(0, 1, 201)
    >>> membership_value = gauss_uncert_mean_lmf(x, [0.3, 0.7, 0.2, 1])
    """
    return ((gaussian_mf(x, [params[0], params[2], params[3]]) * (x >= (params[0] + params[1]) / 2)) +
            (gaussian_mf(x, [params[1], params[2], params[3]]) * (x < (params[0] + params[1]) / 2)))


def gauss_uncert_std_umf(x, params):
    """
    Gaussian with uncertain standard deviation UMF
    
    Parameters
    ----------
    x : 
        numpy (n,) shaped array
        
        The array like input x indicates the points from universe of 
        discourse in which the membership function would be evaluated.
    params : 
        List 
        
        Additional parameters for the membership function. The center, 
        the lower limit of std., the upper limit of std., and the 
        height of the gaussian membership function is indicated by params[0], 
        params[1], params[2], and params[3], respectively.
    
    Returns
    -------
    ndarray
        Returns membership values corresponding with the input.
    
    Examples
    --------
    
    >>> x = linspace(0, 1, 201)
    >>> membership_value = gauss_uncert_std_umf(x, [0.5, 0.2, 0.5, 1])
    """
    return gaussian_mf(x, [params[0], params[2], params[3]])


def gauss_uncert_std_lmf(x, params):
    """
    Gaussian with uncertain standard deviation LMF
    
    Parameters
    ----------
    x : 
        numpy (n,) shaped array
        
        The array like input x indicates the points from universe of 
        discourse in which the membership function would be evaluated.
    params : 
        List 
        
        Additional parameters for the membership function. The center, 
        the lower limit of std., the upper limit of std., and the 
        height of the gaussian membership function is indicated by params[0], 
        params[1], params[2], and params[3], respectively.
    
    Returns
    -------
    ndarray
        Returns membership values corresponding with the input.
    
    Examples
    --------
    
    >>> x = linspace(0, 1, 201)
    >>> membership_value = gauss_uncert_std_lmf(x, [0.5, 0.2, 0.5, 1])
    """
    return gaussian_mf(x, [params[0], params[1], params[3]])


def rgauss_uncert_std_umf(x, params):
    """
    Right Gaussian with uncertain standard deviation UMF
    
    Parameters
    ----------
    x : 
        numpy (n,) shaped array
        
        The array like input x indicates the points from universe of 
        discourse in which the membership function would be evaluated.
    params : 
        List 
        
        Additional parameters for the membership function. The center, 
        the lower limit of std., the upper limit of std., and the 
        height of the gaussian membership function is indicated by params[0], 
        params[1], params[2], and params[3], respectively.
    
    Returns
    -------
    ndarray
        Returns membership values corresponding with the input.
    
    Examples
    --------
    
    >>> x = linspace(0, 1, 201)
    >>> membership_value = rgauss_uncert_std_umf(x, [0.5, 0.2, 0.5, 1])
    """
    return (x < params[0]) * params[3] + gauss_uncert_std_umf(x, params) * (x >= params[0])

def rgauss_uncert_std_lmf(x, params):
    """
    Right Gaussian with uncertain standard deviation LMF
    
    Parameters
    ----------
    x : 
        numpy (n,) shaped array
        
        The array like input x indicates the points from universe of 
        discourse in which the membership function would be evaluated.
    params : 
        List 
        
        Additional parameters for the membership function. The center, 
        the lower limit of std., the upper limit of std., and the 
        height of the gaussian membership function is indicated by params[0], 
        params[1], params[2], and params[3], respectively.
    
    Returns
    -------
    ndarray
        Returns membership values corresponding with the input.
    
    Examples
    --------
    
    >>> x = linspace(0, 1, 201)
    >>> membership_value = rgauss_uncert_std_lmf(x, [0.5, 0.2, 0.5, 1])
    """
    return (x < params[0]) * params[3] + gauss_uncert_std_lmf(x, params) * (x >= params[0])

def lgauss_uncert_std_umf(x, params):
    """
    Left Gaussian with uncertain standard deviation UMF
    
    Parameters
    ----------
    x : 
        numpy (n,) shaped array
        
        The array like input x indicates the points from universe of 
        discourse in which the membership function would be evaluated.
    params : 
        List 
        
        Additional parameters for the membership function. The center, 
        the lower limit of std., the upper limit of std., and the 
        height of the gaussian membership function is indicated by params[0], 
        params[1], params[2], and params[3], respectively.
    
    Returns
    -------
    ndarray
        Returns membership values corresponding with the input.
    
    Examples
    --------
    
    >>> x = linspace(0, 1, 201)
    >>> membership_value = lgauss_uncert_std_umf(x, [0.5, 0.2, 0.5, 1])
    """
    return (x > params[0]) * params[3] + gauss_uncert_std_umf(x, params) * (x <= params[0])

def lgauss_uncert_std_lmf(x, params):
    """
    Left Gaussian with uncertain standard deviation LMF
    
    Parameters
    ----------
    x : 
        numpy (n,) shaped array
        
        The array like input x indicates the points from universe of 
        discourse in which the membership function would be evaluated.
    params : 
        List 
        
        Additional parameters for the membership function. The center, 
        the lower limit of std., the upper limit of std., and the 
        height of the gaussian membership function is indicated by params[0], 
        params[1], params[2], and params[3], respectively.
    
    Returns
    -------
    ndarray
        Returns membership values corresponding with the input.
    
    Examples
    --------
    
    >>> x = linspace(0, 1, 201)
    >>> membership_value = lgauss_uncert_std_lmf(x, [0.5, 0.2, 0.5, 1])
    """
    return (x > params[0]) * params[3] + gauss_uncert_std_lmf(x, params) * (x <= params[0])


class FuzzySet(object):
    """Interval Type 2 Fuzzy Set (IT2FS).
       
        Parameters
        ----------
        Parameters of the constructor function:
        
        domain:
            numpy (n,) shaped array
            
            Indicates the universe of discourse dedicated to the IT2FS.
        
        umf:
            Upper membership function
        
        umf_params:
            List of parameters of upper membership function
        
        lmf:
            Lower membership function
        
        lmf_params:
            List of parameters of lower membership function
        
        check_set:
            If it is True, then a function named check_set in IT2FS will 
            verify the LMF(x) < UMF(x) for any x in the domain. If the 
            user is sure that has selected the parameters of membership 
            functions correct, then calling this time-consuming function 
            is not needed. By default the parameter check_set is False.
            
        Functions
        ----------
        Functions defined in IT2FS class:
        
            copy:
                Returns a copy of the IT2FS.
            plot:
                Plots the IT2FS.
            negation operator -:
                Returns the negated IT2FS.
        
        Examples
        --------
        
        >>> mySet = FuzzySet(linspace(0., 1., 100), 
                          trapezoid_mf, [0, 0.4, 0.6, 1., 1.], 
                          tri_mf, [0.25, 0.5, 0.75, 0.6])
        >>> mySet.plot(filename="mySet")
        
        """

    def __init__(self, domain, umf=zero_mf, umf_params=[], lmf=zero_mf, lmf_params=[],nome='nome',check_set=False):
        self.umf = umf
        self.lmf = lmf
        self.umf_params = umf_params
        self.lmf_params = lmf_params
        self.nome = nome

        self.domain = domain
        self.upper = maximum(minimum(umf(domain, umf_params), 1), 0)
        self.lower = maximum(minimum(lmf(domain, lmf_params), 1), 0)
        if check_set:
            self.check_set()
            
    def __str__(self):
        return self.nome

    def check_set(self):
        """
        Verifies the LMF(x) < UMF(x) for any x in the domain.
        """
        for l, u in zip(self.lower, self.upper):
            if l > u:
                raise ValueError("LMF in some points in domain is larger than UMF.")

    def copy(self):
        """
        Copies the IT2FS.
        
        Returns
        -------
        IT2FS
        
        Returns a copy of the IT2FS.
        """
        return FuzzySet(self.domain, umf=self.umf, umf_params=self.umf_params, lmf=self.lmf, lmf_params=self.lmf_params)

    def plot(self, title=None, legend_text=None, filename=None):
        """
        Plots the IT2FS.
        
        Parameters
        ----------
        title:
            str
            
            If it is set, it indicates the title which would be 
            represented in the plot. If it is not set, the plot would not 
            have a title.
        
        legend_text:
            str
            
            If it is set, it indicates the legend text which would 
            be represented in the plot. If it is not set, the plot would 
            not contain a legend.
        
        filename:
            str
            
            If it is set, the plot would be saved as a filename + ".pdf" 
            file.
        
        Examples
        --------
        
        >>> mySet = FuzzySet(linspace(0., 1., 100), 
                          trapezoid_mf, [0, 0.4, 0.6, 1., 1.], 
                          tri_mf, [0.25, 0.5, 0.75, 0.6])
        >>> mySet.plot(filename="mySet")
        """
        plt.figure()
        plt.fill_between(self.domain, self.upper, self.lower)
        if legend_text is not None:
            plt.legend([legend_text])
        if title is not None:
            plt.title(title)
        plt.plot(self.domain, self.upper, color="black")
        plt.plot(self.domain, self.lower, color="black")
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.grid(True)
        plt.xlabel("Universo de Discurso",fontsize=18)
        plt.ylabel("Grau de pertinência",fontsize=18)
        if filename is not None:
            plt.savefig(filename + ".pdf", format="pdf", dpi=300, bbox_inches="tight")
        plt.show()

    def __neg__(self):
        """
        Negates the IT2FS.
        
        Returns
        -------
        IT2FS
        
        Returns a negated copy of the IT2FS.
        """
        neg_it2fs = FuzzySet(self.domain)
        neg_it2fs.upper = subtract(1, self.upper)
        neg_it2fs.lower = subtract(1, self.lower)
        return neg_it2fs


def IT2FS_Gaussian_UncertMean(domain, params):
    """
    Creates an Gaussian IT2FS with uncertain mean value.
    
    Parameters
    ----------
    
    domain:
        numpy (n,) shaped array
        
        Indicates the universe of discourse dedicated to the IT2FS.
    
    params:
        List
        
        The parameters of the Gaussian IT2FS with uncertain mean value, 
        the mean center, the mean spread, the standard deviation, and the height are 
        indicated by params[0], params[1], params[2], and params[3], respectively.
    
    Returns
    -------
    IT2FS
        Returns a Gaussian IT2FS with uncertain mean value with specified 
        parameters.
    
    Examples
    --------
    
    >>> domain = linspace(0., 1., 100)
    >>> mySet = IT2FS_Gaussian_UncertMean(domain, [0., 0.25, 0.2])
    >>> mySet.plot()
    
    """
    ml = params[0] - params[1] / 2.
    mr = params[0] + params[1] / 2.
    return FuzzySet(domain, 
                 gauss_uncert_mean_umf, [ml, mr, params[2], params[3]], 
                 gauss_uncert_mean_lmf, [ml, mr, params[2], params[3]])


def IT2FS_Gaussian_UncertStd(domain, params):
    """
    Creates a Gaussian IT2FS with uncertain standard deviation value.
    
    Parameters
    ----------
    
    domain:
        numpy (n,) shaped array
        
        Indicates the universe of discourse dedicated to the IT2FS.
    
    params:
        List
        
        The parameters of the Gaussian IT2FS with uncertain standard 
        deviation value, the mean, the standard deviation center, 
        the standard deviation spread, and the height are indicated by params[0], 
        params[1], params[2], and params[3], respectively.
    
    Returns
    -------
    IT2FS
        Returns a Gaussian IT2FS with uncertain standard deviation value 
        with specified parameters.
    
    Examples
    --------
    
    >>> domain = linspace(0., 1., 100)
    >>> mySet = IT2FS_Gaussian_UncertStd(domain, [0.5, 0.2, 0.05, 1.])
    >>> mySet.plot()
    
    """
    stdl = params[1] - params[2] / 2
    stdr = params[1] + params[2] / 2
    return FuzzySet(domain, 
                 gauss_uncert_std_umf, [params[0], stdl, stdr, params[3]], 
                 gauss_uncert_std_lmf, [params[0], stdl, stdr, params[3]])


def R_IT2FS_Gaussian_UncertStd(domain, params):
    """
    Creates a Right Gaussian IT2FS with uncertain standard deviation value.
    
    Parameters
    ----------
    
    domain:
        numpy (n,) shaped array
        
        Indicates the universe of discourse dedicated to the IT2FS.
    
    params:
        List
        
        The parameters of the Gaussian IT2FS with uncertain standard 
        deviation value, the mean, the standard deviation center, 
        the standard deviation spread, and the height are indicated by params[0], 
        params[1], params[2], and params[3], respectively.
    
    Returns
    -------
    IT2FS
        Returns a Gaussian IT2FS with uncertain standard deviation value 
        with specified parameters.
    
    Examples
    --------
    
    >>> domain = linspace(0., 1., 100)
    >>> mySet = R_IT2FS_Gaussian_UncertStd(domain, [0.5, 0.2, 0.05, 1.])
    >>> mySet.plot()
    
    """
    stdl = params[1] - params[2] / 2
    stdr = params[1] + params[2] / 2
    return FuzzySet(domain, 
                  rgauss_uncert_std_umf, 
                  [params[0], stdl, stdr, params[3]],
                  rgauss_uncert_std_lmf, 
                  [params[0], stdl, stdr, params[3]])

def L_IT2FS_Gaussian_UncertStd(domain, params):
    """
    Creates a Left Gaussian IT2FS with uncertain standard deviation value.
    
    Parameters
    ----------
    
    domain:
        numpy (n,) shaped array
        
        Indicates the universe of discourse dedicated to the IT2FS.
    
    params:
        List
        
        The parameters of the Gaussian IT2FS with uncertain standard 
        deviation value, the mean, the standard deviation center, 
        the standard deviation spread, and the height are indicated by params[0], 
        params[1], params[2], and params[3], respectively.
    
    Returns
    -------
    IT2FS
        Returns a Gaussian IT2FS with uncertain standard deviation value 
        with specified parameters.
    
    Examples
    --------
    
    >>> domain = linspace(0., 1., 100)
    >>> mySet = L_IT2FS_Gaussian_UncertStd(domain, [0.5, 0.2, 0.05, 1.])
    >>> mySet.plot()
    
    """
    stdl = params[1] - params[2] / 2
    stdr = params[1] + params[2] / 2
    return FuzzySet(domain, 
                  lgauss_uncert_std_umf, 
                  [params[0], stdl, stdr, params[3]],
                  lgauss_uncert_std_lmf, 
                  [params[0], stdl, stdr, params[3]])


def IT2FS_plot(*sets, title=None, legends=None, filename=None):
    """
    Plots multiple IT2FSs together in the same figure.
    
    Parameters
    ----------
    *sets:
        Multiple number of IT2FSs which would be plotted.
    
    title:
        str
        
        If it is set, it indicates the title which would be 
        represented in the plot. If it is not set, the plot would not 
        have a title.
        
    legends:
        List of strs
        
        List of legend texts to be presented in plot. If it is not 
        set, no legend would be in plot.
        
    filename:
        str
        
        If it is set, the plot would be saved as a filename + ".pdf" 
        file.
        
    Examples
    --------
    
    >>> domain = linspace(0., 1., 100)
    >>> it2fs1 = IT2FS_Gaussian_UncertStd(domain, [0.33, 0.2, 0.05])
    >>> it2fs2 = IT2FS_Gaussian_UncertStd(domain, [0.66, 0.2, 0.05])
    >>> IT2FS_plot(it2fs1, it2fs2, title="Plotting IT2FSs", 
                   legends=["First set", "Second set"])
    """
    'para gerar graficos maiores:'
    plt.figure(figsize=(15,5))
    for it2fs in sets:
        plt.fill_between(it2fs.domain, it2fs.upper, it2fs.lower,alpha=0.5)
    if legends is not None:
        plt.legend(legends)
    for it2fs in sets:
        plt.plot(it2fs.domain, it2fs.lower, color="black")
        plt.plot(it2fs.domain, it2fs.upper, color="black")
    if title is not None:
        plt.title(title)
    #plt.xlabel("Universe of Discourse",fontsize=25)
    #plt.ylabel("Degree of Membership",fontsize=25)
    
    plt.xlabel("Universo de Discurso",fontsize=25)
    plt.ylabel("Grau de pertinência",fontsize=25)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.grid(False)
    if filename is not None:
        plt.savefig(filename + ".pdf", format="pdf", dpi=300, bbox_inches="tight")
    plt.show()
    
def TR_plot(domain, tr, title=None, legend=None, filename=None):
    """
    Plots a type reduced IT2FS.
    
    Parameters
    ----------
    domain:
        numpy (n,) shaped array
        
        Indicates the universe of discourse dedicated to the IT2FS.
    
    tr:
        Tuple (l, r)
        
        Indicates the type reduced set to be plotted.
    
    title:
        str
        
        If it is set, it indicates the title which would be 
        represented in the plot. If it is not set, the plot would not 
        have a title.
        
    legend:
        str
        
        If it is set, it indicates the legend text which would 
        be represented in the plot. If it is not set, the plot would 
        not contain a legend.
        
    filename:
        str
        
        If it is set, the plot would be saved as a filename + ".pdf" 
        file.
        
    Examples
    --------
    
    >>> tr1 = (0.2, 0.3)
    >>> TR_plot(linspace(0., 1., 100), tr1)
    """
    plt.figure()
    plt.plot([min(domain), tr[0], tr[0], tr[1], tr[1], max(domain)], 
              [0, 0, 1, 1, 0, 0], linewidth=2)
    plt.xlim((min(domain), max(domain)))
    plt.xlabel("Domain")
    plt.ylabel("Membership degree")
    if title is not None:
        plt.title(title)
    if legend is not None:
        plt.legend([legend])
    plt.grid(True)
    if filename is not None:
        plt.savefig(filename + ".pdf", format="pdf", dpi=300, bbox_inches="tight")
    plt.show()

def crisp(tr):
    """
    Calculates the crisp number achieved from type reduced IT2FS.
    
    Parameters
    ----------
    
    tr:
        Tuple (l, r)
        
        Type reduced IT2FS
    
    Returns
    -------
    float
    
    Returns the crisp number achieved from type reduced IT2FS.
    
    Examples
    --------
    
    >>> tr1 = (0.1, 0.3)
    >>> print(crisp(tr1))
    """
    return (tr[0] + tr[1]) / 2

def crisp_list(trs, o=None):
    """
    Calculates the crisp outputs achieved by calling the evaluate_list 
    function from IT2FLS class.
    
    Parameters
    ----------
    
    trs:
        List of Tuple (l, r)
        
        Type reduced IT2FSs
    
    o:
        str
        
        The name of the output variable to be processed. If it is not given, 
        then the crisp outputs are calculated for all output variables.
    
    Returns
    -------
    List of float (or Dictionary of Lists of float)
    """
    if o is None:
        output = {}
        for key in trs[0].keys():
            output[key] = []
        for tr in trs:
            for key in trs[0].keys():
                output[key].append(crisp(tr[key]))
        return output
            
    else:
        output = []
        for tr in trs:
            output.append(crisp(tr[o]))
        return output

def min_t_norm(a, b):
    """
    Minimum t-norm function.
    
    Parameters
    ----------
    
    a:
        numpy (n,) shaped array
    
    b:
        numpy (n,) shaped array
    
    Returns
    -------
    Returns minimum t-norm of a and b
    
    Examples
    --------
    
    >>> a = random.random(10,)
    >>> b = random.random(10,)
    >>> c = min_t_norm(a, b)
    """
    return minimum(a, b)


def product_t_norm(a, b):
    """
    Product t-norm function.
    
    Parameters
    ----------
    
    a:
        numpy (n,) shaped array
    
    b:
        numpy (n,) shaped array
    
    Returns
    -------
    Returns product t-norm of a and b
    
    Examples
    --------
    
    >>> a = random.random(10,)
    >>> b = random.random(10,)
    >>> c = product_t_norm(a, b)
    """
    return multiply(a, b)


def max_s_norm(a, b):
    """
    Maximum s-norm function.
    
    Parameters
    ----------
    
    a:
        numpy (n,) shaped array
    
    b:
        numpy (n,) shaped array
    
    Returns
    -------
    Returns maximum s-norm of a and b
    
    Examples
    --------
    
    >>> a = random.random(10,)
    >>> b = random.random(10,)
    >>> c = max_s_norm(a, b)
    """
    return maximum(a, b)


def meet(domain, it2fs1, it2fs2, t_norm):
    """
    Meet operator for IT2FSs.
    
    Parameters
    ----------
    
    domain:
        numpy (n,) shaped array
        
        Indicates the universe of discourse dedicated to the IT2FS.
    
    it2fs1:
        IT2FS
        
        First input of the meet operator.
        
    it2fs2:
        IT2FS
        
        Second input of the meet operator.
    
    t_norm:
        function
        
        The t-norm function to be used.
    
    Returns
    -------
    IT2FS
    
    Return the meet of to input IT2FSs.
    
    Examples
    --------
    
    >>> domain = linspace(0., 1., 100)
    >>> it2fs1 = IT2FS_Gaussian_UncertStd(domain, [0.33, 0.2, 0.05])
    >>> it2fs2 = IT2FS_Gaussian_UncertStd(domain, [0.66, 0.2, 0.05])
    >>> it2fs3 = meet(domain, it2fs1, it2fs2, min_t_norm)
    >>> it2fs3.plot()
    """
    it2fs = FuzzySet(domain)
    it2fs.upper = t_norm(it2fs1.upper, it2fs2.upper)
    it2fs.lower = t_norm(it2fs1.lower, it2fs2.lower)
    return it2fs


def join(domain, it2fs1, it2fs2, s_norm):
    """
    Join operator for IT2FSs.
    
    Parameters
    ----------
    
    domain:
        numpy (n,) shaped array
        
        Indicates the universe of discourse dedicated to the IT2FS.
    
    it2fs1:
        IT2FS
        
        First input of the join operator.
        
    it2fs2:
        IT2FS
        
        Second input of the join operator.
    
    s_norm:
        function
        
        The s-norm function to be used.
    
    Returns
    -------
    IT2FS
    
    Return the join of to input IT2FSs.
    
    Examples
    --------
    
    >>> domain = linspace(0., 1., 100)
    >>> it2fs1 = IT2FS_Gaussian_UncertStd(domain, [0.33, 0.2, 0.05])
    >>> it2fs2 = IT2FS_Gaussian_UncertStd(domain, [0.66, 0.2, 0.05])
    >>> it2fs3 = join(domain, it2fs1, it2fs2, max_s_norm)
    >>> it2fs3.plot()
    """
    it2fs = FuzzySet(domain)
    it2fs.upper = s_norm(it2fs1.upper, it2fs2.upper)
    it2fs.lower = s_norm(it2fs1.lower, it2fs2.lower)
    return it2fs

def trim(intervals):
    v = intervals[:, 3]
    i, = where(v > 0)
    if i.size == 0:
        return False
    else:
        min1 = i[0]
        max1 = i[-1] + 1
    
        v = intervals[:, 2]
        i, = where(v > 0)
        if i.size == 0:
            min2 = min1
            max2 = max1
        else:
            min2 = i[0]
            max2 = i[-1] + 1
        return intervals[min(min1, min2):max(max1, max2), :]

def KM_algorithm(intervals, params=None):  # intervals = [[a1, b1, c1, d1], [a2, b2, c2, d2], ...]
    """
    KM algorithm
    
    Parameters
    ----------
    
    intervals:
        numpy (n, 4) array
        
        Y = intervals[:, 0:2]
        
        F = intervals[:, 2:4]
    
    params:
        List
        
        List of parameters of algorithm, if it is needed.
    
    Returns
    -------
    Tuple (l, r)
    """
    # left calculations
    intervals = intervals[intervals[:,0].argsort()]
    intervals = trim(intervals)
    
    if intervals is False:
        return 0, 0
    
    w_l = (intervals[:, 2] + intervals[:, 3]) / 2.

    N = len(intervals)
    y_l_prime_num = npsum(intervals[:, 0] * w_l)
    y_prime_den = npsum(w_l)
    
    y_l_prime = y_l_prime_num / y_prime_den
    while True:
        k_l = 0
        for i in range(1, N):
            if (intervals[i-1, 0] <= y_l_prime <= intervals[i, 0]) or \
                isclose(intervals[i-1, 0], y_l_prime) or \
                isclose(y_l_prime, intervals[i, 0]):
                k_l = i-1
                break
        
        ii = arange(N)
        w_l = (ii <= k_l) * intervals[:, 3] + (ii > k_l) * intervals[:, 2]
        y_l_num = npsum(intervals[:k_l+1, 0] * intervals[:k_l+1, 3])
        y_l_den = npsum(intervals[:k_l+1, 3])
        y_l_num += npsum(intervals[k_l+1:, 0] * intervals[k_l+1:, 2])
        y_l_den += npsum(intervals[k_l+1:, 2])
        y_l = y_l_num / y_l_den
        if y_l == y_l_prime:
            break
        else:
            y_l_prime = y_l
    # right calculations
    intervals = intervals[intervals[:, 1].argsort()]
    w_r = (intervals[:, 2] + intervals[:, 3]) / 2.
    
    y_r_prime_num = npsum(intervals[:, 1] * w_r)
    y_r_prime = y_r_prime_num / y_prime_den
    while True:
        k_r = 0
        for i in range(1, N):
            if (intervals[i-1, 1] <= y_r_prime <= intervals[i, 1]) or \
                isclose(intervals[i-1, 1], y_r_prime) or \
                isclose(y_r_prime, intervals[i, 1]):
                k_r = i-1
                break
        
        ii = arange(N)
        w_r = (ii <= k_r) * intervals[:, 2] + (ii > k_r) * intervals[:, 3]
        y_r_num = npsum(intervals[:k_r+1, 1] * intervals[:k_r+1, 2])
        y_r_den = npsum(intervals[:k_r+1, 2])
        y_r_num += npsum(intervals[k_r+1:, 1] * intervals[k_r+1:, 3])
        y_r_den += npsum(intervals[k_r+1:, 3])
        y_r = y_r_num / y_r_den
        if y_r == y_r_prime:
            break
        else:
            y_r_prime = y_r
    return y_l, y_r


def EKM_algorithm(intervals, params=None):
    """
    EKM algorithm
    
    Parameters
    ----------
    
    intervals:
        numpy (n, 4) array
        
        Y = intervals[:, 0:2]
        
        F = intervals[:, 2:4]
    
    params:
        List
        
        List of parameters of algorithm, if it is needed.
    
    Returns
    -------
    Tuple (l, r)
    """
    
    # Left calculations
    intervals = intervals[intervals[:,0].argsort()]
    intervals = trim(intervals)
    
    if intervals is False:
        return 0, 0
    
    N = len(intervals)
    
    k_l = round(N / 2.4)

    a_l = npsum(intervals[:k_l+1, 0] * intervals[:k_l+1, 3]) + \
          npsum(intervals[k_l+1:, 0] * intervals[k_l+1:, 2])
    b_l = npsum(intervals[:k_l+1, 3]) + npsum(intervals[k_l+1:, 2])
    y_l_prime = a_l / b_l
    while True:
        k_l_prime = 0
        for i in range(0, N-1):
            if (intervals[i, 0] <= y_l_prime <= intervals[i+1, 0]) or \
                isclose(intervals[i, 0], y_l_prime) or \
                isclose(y_l_prime, intervals[i+1, 0]):
                k_l_prime = i
                break
        if k_l_prime == k_l:
            y_l = y_l_prime
            break
        s_l = sign(k_l_prime - k_l)
        imin = min(k_l, k_l_prime) + 1
        imax = max(k_l, k_l_prime) + 1
        
        a_l_prime = a_l + s_l * npsum(intervals[imin:imax, 0] * \
                    (intervals[imin:imax, 3] - intervals[imin:imax, 2]))
        b_l_prime = b_l + s_l * \
                    npsum(intervals[imin:imax, 3] - intervals[imin:imax, 2])
        y_l_second = a_l_prime / b_l_prime
        
        k_l = k_l_prime
        y_l_prime = y_l_second
        a_l = a_l_prime
        b_l = b_l_prime
    # Right calculations
    intervals = intervals[intervals[:,1].argsort()]
    k_r = round(N / 1.7)
    a_r = npsum(intervals[:k_r+1, 1] * intervals[:k_r+1, 2]) + \
          npsum(intervals[k_r+1:, 1] * intervals[k_r+1:, 3])
    b_r = npsum(intervals[:k_r+1, 2]) + npsum(intervals[k_r+1:, 3])

    y_r_prime = a_r / b_r

    while True:
        k_r_prime = 0
        for i in range(0, N-1):
            if (intervals[i, 1] <= y_r_prime <= intervals[i+1, 1]) or \
                isclose(intervals[i, 1], y_r_prime) or \
                isclose(y_r_prime, intervals[i+1, 1]):
                k_r_prime = i
                break
        if k_r_prime == k_r:
            y_r = y_r_prime
            break
        
        s_r = sign(k_r_prime - k_r)
        
        imin = min(k_r, k_r_prime) + 1
        imax = max(k_r, k_r_prime) + 1
        a_r_prime = npsum(intervals[imin:imax, 1] * (intervals[imin:imax, 3] - 
                          intervals[imin:imax, 2]))
        b_r_prime = npsum(intervals[imin:imax, 3] - intervals[imin:imax, 2])
        
        a_r_prime = a_r - s_r * a_r_prime
        b_r_prime = b_r - s_r * b_r_prime
        y_r_second = a_r_prime / b_r_prime
        k_r = k_r_prime
        y_r_prime = y_r_second
        a_r = a_r_prime
        b_r = b_r_prime
    return y_l, y_r


def WEKM_algorithm(intervals, params=None):
    """
    WEKM algorithm
    
    Parameters
    ----------
    
    intervals:
        numpy (n, 4) array
        
        Y = intervals[:, 0:2]
        
        F = intervals[:, 2:4]
    
    params:
        List
        
        List of parameters of algorithm, if it is needed.
    
    Returns
    -------
    Tuple (l, r)
    """
    
    # Left calculations
    intervals = intervals[intervals[:,0].argsort()]
    intervals = trim(intervals)
    
    if intervals is False:
        return 0, 0
    
    N = len(intervals)
    
    k_l = round(N / 2.4)
    a_l = 0
    b_l = 0
    for i in range(k_l):
        a_l += params[i] * intervals[i, 0] * intervals[i, 3]
        b_l += params[i] * intervals[i, 3]
    for i in range(k_l, N):
        a_l += params[i] * intervals[i, 0] * intervals[i, 2]
        b_l += params[i] * intervals[i, 2]
    y_l_prime = a_l / b_l
    while True:
        k_l_prime = 0
        for i in range(1, N):
            if (intervals[i - 1, 0] <= y_l_prime <= intervals[i, 0]) or \
                isclose(intervals[i - 1, 0], y_l_prime) or \
                isclose(y_l_prime, intervals[i, 0]):
                k_l_prime = i - 1
                break
        if k_l_prime == k_l:
            y_l = y_l_prime
            break
        s_l = sign(k_l_prime - k_l)
        a_l_prime = 0
        b_l_prime = 0
        for i in range(min(k_l, k_l_prime) + 1, max(k_l, k_l_prime)):
            a_l_prime += params[i] * intervals[i, 0] * (intervals[i, 3] - intervals[i, 2])
            b_l_prime += params[i] * (intervals[i, 3] - intervals[i, 2])
        a_l_prime = a_l + s_l * a_l_prime
        b_l_prime = b_l + s_l * b_l_prime
        y_l_second = a_l_prime / b_l_prime
        k_l = k_l_prime
        y_l_prime = y_l_second
        a_l = a_l_prime
        b_l = b_l_prime
    # Right calculations
    intervals = intervals[intervals[:,1].argsort()]
    k_r = round(N / 1.7)
    a_r = 0
    b_r = 0
    for i in range(k_r):
        a_r += params[i] * intervals[i, 1] * intervals[i, 2]
        b_r += params[i] * intervals[i, 2]
    for i in range(k_r, N):
        a_r += params[i] * intervals[i, 1] * intervals[i, 3]
        b_r += params[i] * intervals[i, 3]
    y_r_prime = a_r / b_r
    while True:
        k_r_prime = 0
        for i in range(1, N):
            if (intervals[i - 1, 1] <= y_r_prime <= intervals[i, 1]) or \
                isclose(intervals[i - 1, 1], y_r_prime) or \
                isclose(y_r_prime, intervals[i, 1]):
                k_r_prime = i - 1
                break
        if k_r_prime == k_r:
            y_r = y_r_prime
            break
        s_r = sign(k_r_prime - k_r)
        a_r_prime = 0
        b_r_prime = 0
        for i in range(min(k_r, k_r_prime) + 1, max(k_r, k_r_prime)):
            a_r_prime += params[i] * intervals[i, 1] * (intervals[i, 3] - intervals[i, 2])
            b_r_prime += params[i] * (intervals[i, 3] - intervals[i, 2])
        a_r_prime = a_r - s_r * a_r_prime
        b_r_prime = b_r - s_r * b_r_prime
        y_r_second = a_r_prime / b_r_prime
        k_r = k_r_prime
        y_r_prime = y_r_second
        a_r = a_r_prime
        b_r = b_r_prime
    return y_l, y_r


def TWEKM_algorithm(intervals, params):
    """
    TWEKM algorithm
    
    Parameters
    ----------
    
    intervals:
        numpy (n, 4) array
        
        Y = intervals[:, 0:2]
        
        F = intervals[:, 2:4]
    
    params:
        List
        
        List of parameters of algorithm, if it is needed.
    
    Returns
    -------
    Tuple (l, r)
    """
    params = []
    N = len(intervals)
    for i in range(N):
        if i == 0 or i == N-1:
            params.append(0.5)
        else:
            params.append(1)
    return WEKM_algorithm(intervals, params)


def EIASC_algorithm(intervals, params=None):
    """
    EIASC algorithm
    
    Parameters
    ----------
    
    intervals:
        numpy (n, 4) array
        
        Y = intervals[:, 0:2]
        
        F = intervals[:, 2:4]
    
    params:
        List
        
        List of parameters of algorithm, if it is needed.
    
    Returns
    -------
    Tuple (l, r)
    """
    
    # Left calculations
    intervals = intervals[intervals[:,0].argsort()]
    intervals = trim(intervals)
    
    if intervals is False:
        return 0, 0
    
    N = len(intervals)
    b = npsum(intervals[:, 2])
    
    
    a_l = npsum(intervals[:, 0] * intervals[:, 2])
    b_l = b
    L = 0
    while True:
        d = intervals[L, 3] - intervals[L, 2]
        a_l += intervals[L, 0] * d
        b_l += d
        y_l = a_l / b_l
        L += 1
        if (y_l <= intervals[L, 0]) or isclose(y_l, intervals[L, 0]):
            break 
    # Right calculations
    intervals = intervals[intervals[:,1].argsort()]
    a_r = npsum(intervals[:, 1] * intervals[:, 2])
    b_r = b
    R = N - 1
    while True:
        d = intervals[R, 3] - intervals[R, 2]
        a_r += intervals[R, 1] * d
        b_r += d
        y_r = a_r / b_r
        R -= 1
        if (y_r >= intervals[R, 1]) or isclose(y_r, intervals[R, 1]):
            break  
    return y_l, y_r


def WM_algorithm(intervals, params=None):
    """
    WM algorithm
    
    Parameters
    ----------
    
    intervals:
        numpy (n, 4) array
        
        Y = intervals[:, 0:2]
        
        F = intervals[:, 2:4]
    
    params:
        List
        
        List of parameters of algorithm, if it is needed.
    
    Returns
    -------
    Tuple (l, r)
    """
    intervals = intervals[intervals[:,0].argsort()]
    intervals = trim(intervals)
    
    if intervals is False:
        return 0, 0
    
    F = intervals[:, 2:4]
    Y = intervals[:, 0:2]
    y_l_sup = min(npsum(F[:, 0] * Y[:, 0]) / npsum(F[:, 0]), 
                  npsum(F[:, 1] * Y[:, 0]) / npsum(F[:, 1]))
    y_r_inf = min(npsum(F[:, 1] * Y[:, 1]) / npsum(F[:, 1]), 
                  npsum(F[:, 0] * Y[:, 1]) / npsum(F[:, 0]))
    c = npsum(F[:, 1] - F[:, 0]) / (npsum(F[:, 0]) * npsum(F[:, 1]))
    y_l_inf = y_l_sup - c * (npsum(F[:, 0] * (Y[:, 0] - Y[0, 0])) * 
                             npsum(F[:, 1] * (Y[-1, 0] - Y[:, 0]))) / (npsum(F[:, 0] * (Y[:, 0] - Y[0, 0])) + npsum(F[:, 1] * (Y[-1, 0] - Y[:, 0])))
    y_r_sup = y_r_inf + c * (npsum(F[:, 1] * (Y[:, 1] - Y[0, 1])) * 
                             npsum(F[:, 0] * (Y[-1, 1] - Y[:, 1]))) / (npsum(F[:, 1] * (Y[:, 1] - Y[0, 1])) + npsum(F[:, 0] * (Y[-1, 1] - Y[:, 1])))
    y_l = (y_l_sup + y_l_inf) / 2
    y_r = (y_r_sup + y_r_inf) / 2
    return y_l, y_r


def BMM_algorithm(intervals, params):
    """
    BMM algorithm
    
    Parameters
    ----------
    
    intervals:
        numpy (n, 4) array
        
        Y = intervals[:, 0:2]
        
        F = intervals[:, 2:4]
    
    params:
        List
        
        List of parameters of algorithm, if it is needed.
    
    Returns
    -------
    float
    
    Crisp output
    """
    intervals = intervals[intervals[:,0].argsort()]
    intervals = trim(intervals)
    
    if intervals is False:
        return 0
    
    F = intervals[:, 2:4]
    Y = (intervals[:, 0] + intervals[:, 1]) / 2.
    m = params[0]
    n = params[1]
    #Y = Y.reshape((Y.size,))
    return m * npsum(F[:, 0] * Y) / npsum(F[:, 0]) + n * npsum(F[:, 1] * Y) / npsum(F[:, 1])


def LBMM_algorithm(intervals, params):
    """
    LBMM algorithm (BMM extended by Li et al.)
    
    Ref. An Overview of Alternative Type-ReductionApproaches for 
    Reducing the Computational Costof Interval Type-2 Fuzzy Logic 
    Controllers
    
    Parameters
    ----------
    
    intervals:
        numpy (n, 4) array
        
        Y = intervals[:, 0:2]
        
        F = intervals[:, 2:4]
    
    params:
        List
        
        List of parameters of algorithm, if it is needed.
    
    Returns
    -------
    float
    
    Crisp output
    """
    intervals = intervals[intervals[:,0].argsort()]
    intervals = trim(intervals)
    
    if intervals is False:
        return 0
    
    F = intervals[:, 2:4]
    Y = intervals[:, 0:2]
    m = params[0]
    n = params[1]
    return m * npsum(F[:, 0] * Y[:, 0]) / npsum(F[:, 0]) + n * npsum(F[:, 1] * Y[:, 1]) / npsum(F[:, 1])


def NT_algorithm(intervals, params=None):
    """
    NT algorithm
    
    Parameters
    ----------
    
    intervals:
        numpy (n, 4) array
        
        Y = intervals[:, 0:2]
        
        F = intervals[:, 2:4]
    
    params:
        List
        
        List of parameters of algorithm, if it is needed.
    
    Returns
    -------
    float
    
    Crisp output
    """
    intervals = intervals[intervals[:,0].argsort()]
    intervals = trim(intervals)
    
    if intervals is False:
        return 0
    
    F = intervals[:, 2:4]
    Y = (intervals[:, 0] + intervals[:, 1]) / 2.
    return (npsum(Y * F[:, 1]) + npsum(Y * F[:, 0])) / (npsum(F[:, 0]) + npsum(F[:, 1]))


def Centroid(it2fs, alg_func, domain, alg_params=None):
    """
    Centroid type reduction for an interval type 2 fuzzy set.
    
    Parameters
    ----------
    
    it2fs:
        IT2FS
        
        IT2FS which would be type reduced.
    
    alg_func:
        Function
        
        Type reduction algorithm to be used, which is one of these:
            
            KM_algorithm, EKM_algorithm, WEKM_algorithm, TWEKM_algorithm, 
            EIASC_algorithm, WM_algorithm, BMM_algorithm, LBMM_algorithm, and 
            NT_algorithm
    
    domain:
        numpy (n,) shaped array
        
        Indicates the universe of discourse dedicated to the IT2FS.
    
    alg_params:
        List
        
        List of parameters of type reduction algorithm if it is needed.
    
    Returns
    -------
    Based on selected type reduction algorithm tuple (l, r) or float
    
    Returns Centroid type reduction of the input IT2FS.
    """
    intervals = c_[domain, domain, it2fs.lower, it2fs.upper]
    return alg_func(intervals, alg_params)


def CoSet(firing_array, consequent_array, alg_func, domain, alg_params=None):
    """
    Center of sets type reduction.
    
    Parameters
    ----------
    
    firing_array:
        numpy (m, 2) shaped array
        
        Firing strength of consequents.
    
    consequent_array:
        List of IT2FS
        
        List of consequents corresponding with the rules of IT2FLS
    
    alg_func:
        Function
        
        Type reduction algorithm to be used, which is one of these:
            
            KM_algorithm, EKM_algorithm, WEKM_algorithm, TWEKM_algorithm, 
            EIASC_algorithm, WM_algorithm, BMM_algorithm, LBMM_algorithm, and 
            NT_algorithm
    
    domain:
        numpy (n,) shaped array
        
        Indicates the universe of discourse dedicated to the IT2FS.
    
    alg_params:
        List
        
        List of parameters of type reduction algorithm if it is needed.
    
    Returns
    -------
    Based on selected type reduction algorithm tuple (l, r) or float
    
    Returns Center of sets type reduction of the input IT2FS.
    """
    intervals = []
    for l in range(len(consequent_array)):
        tmp = Centroid(consequent_array[l], alg_func, domain)
        intervals.append([tmp[0], tmp[1], firing_array[l, 0], firing_array[l, 1]])
    return alg_func(array(intervals), alg_params)


def CoSum(it2fs_array, alg_func, domain, alg_params=None):
    """
    Center of sum type reduction for an interval type 2 fuzzy set.
    
    Parameters
    ----------
    
    it2fs_array:
        List of IT2FSs
        
        List of final IT2FSs achieved by evaluating rule base.
    
    alg_func:
        Function
        
        Type reduction algorithm to be used, which is one of these:
            
            KM_algorithm, EKM_algorithm, WEKM_algorithm, TWEKM_algorithm, 
            EIASC_algorithm, WM_algorithm, BMM_algorithm, LBMM_algorithm, and 
            NT_algorithm
    
    domain:
        numpy (n,) shaped array
        
        Indicates the universe of discourse dedicated to the IT2FS.
    
    alg_params:
        List
        
        List of parameters of type reduction algorithm if it is needed.
    
    Returns
    -------
    Based on selected type reduction algorithm tuple (l, r) or float
    
    Returns Center of sum type reduction of the input IT2FS.
    """
    lower_sum = zeros_like(domain)
    upper_sum = zeros_like(domain)
    for it2fs in it2fs_array:
        add(lower_sum, it2fs.lower, out=lower_sum)
        add(upper_sum, it2fs.lower, out=upper_sum)
    intervals = c_[domain, domain, lower_sum, upper_sum]
    return alg_func(intervals, alg_params)


def Height(it2fs_array, alg_func, domain, alg_params=None):
    """
    Height type reduction for an interval type 2 fuzzy set.
    
    Parameters
    ----------
    
    it2fs_array:
        List of IT2FSs
        
        List of final IT2FSs achieved by evaluating rule base.
    
    alg_func:
        Function
        
        Type reduction algorithm to be used, which is one of these:
            
            KM_algorithm, EKM_algorithm, WEKM_algorithm, TWEKM_algorithm, 
            EIASC_algorithm, WM_algorithm, BMM_algorithm, LBMM_algorithm, and 
            NT_algorithm
    
    domain:
        numpy (n,) shaped array
        
        Indicates the universe of discourse dedicated to the IT2FS.
    
    alg_params:
        List
        
        List of parameters of type reduction algorithm if it is needed.
    
    Returns
    -------
    Based on selected type reduction algorithm tuple (l, r) or float
    
    Returns Height type reduction of the input IT2FS.
    """
    intervals = []
    for it2fs in it2fs_array:
        index = argmax(it2fs.upper)
        intervals.append([domain[index], domain[index], it2fs.lower[index], it2fs.upper[index]])
    return alg_func(array(intervals), alg_params)


def ModiHe(it2fs_array, spread_array, alg_func, domain, alg_params=None):
    """
    Modified height type reduction for an interval type 2 fuzzy set.
    
    Parameters
    ----------
    
    it2fs_array:
        List of IT2FSs
        
        List of final IT2FSs achieved by evaluating rule base.
    
    spread_array:
        List of spread values corresponding with IT2FSs in it2fs_array. 
    
    alg_func:
        Function
        
        Type reduction algorithm to be used, which is one of these:
            
            KM_algorithm, EKM_algorithm, WEKM_algorithm, TWEKM_algorithm, 
            EIASC_algorithm, WM_algorithm, BMM_algorithm, LBMM_algorithm, and 
            NT_algorithm
    
    domain:
        numpy (n,) shaped array
        
        Indicates the universe of discourse dedicated to the IT2FS.
    
    alg_params:
        List
        
        List of parameters of type reduction algorithm if it is needed.
    
    Returns
    -------
    Based on selected type reduction algorithm tuple (l, r) or float
    
    Returns Modified height type reduction of the input IT2FS.
    """
    intervals = []
    j = 0
    for it2fs in it2fs_array:
        index = argmax(it2fs.upper)
        intervals.append([domain[index], domain[index],
                          it2fs.lower[index]/(spread_array[j] ** 2),
                          it2fs.upper[index]/(spread_array[j] ** 2)])
        j += 1
    return alg_func(array(intervals), alg_params)


class Type2Model():
    """Interval type 2 fuzzy logic system.
    
    No construction parameter is needed.
    
    Members
    -------
    inputs:
        List of str
        
        List of names of inputs as str
    
    output:
        List of str
        
        List of names ot outputs as str
        
    rules:
        List of tuples (antecedent, consequent)
        
        List of rules which each rule is defined as a 
        tuple (antecedent, consequent)
        
        Both antacedent and consequent are lists of tuples. Each tuple 
        of this list shows 
        assignement of a variable to an IT2FS. First element of the tuple 
        must be variable name (input or output) as a str and the second 
        element must be an IT2FS. 
    
    Functions
    ---------
    
    add_input_variable:
        
        Adds an input variable to the inputs list of the IT2FLS.
    
    add_output_variable:
        
        Adds an output variable to the outputs list of the IT2FLS.
    
    add_rule:
        
        Adds a rule to the rules list of the IT2FLS.
    
    copy:
        
        Returns a copy of the IT2FLS.
    
    evaluate:
        
        Evaluates the IT2FLS's output for a specified crisp input.
    
    Examples
    --------
    
    Assume that we are going to simulate an IT2FLS with two inputs and 
    two outputs. Each input is defined by three IT2FSs, Small, Medium, 
    and Large. These IT2FSs are from Gaussian type with uncertain 
    standard deviation value. The universe of discourse of the fuzzy 
    system is defined as the interval [0, 1]. Also, the rule base of 
    the system is defined as below:
        
        * IF x1 is Small and x2 is Small THEN y1 is Small and y2 is Large
        * IF x1 is Medium and x2 is Medium THEN y1 is Medium and y2 is Small
        * IF x1 is Large and x2 is Large THEN y1 is Large and y2 is Small
    
    The codes to simulate the aforementioned system using the PyIT2FLS 
    would be as below:
    
    >>> domain = linspace(0., 1., 100)
    >>> 
    >>> Small = IT2FS_Gaussian_UncertStd(domain, [0, 0.15, 0.1])
    >>> Medium = IT2FS_Gaussian_UncertStd(domain, [0.5, 0.15, 0.1])
    >>> Large = IT2FS_Gaussian_UncertStd(domain, [1., 0.15, 0.1])
    >>> IT2FS_plot(Small, Medium, Large, legends=["Small", "Medium", "large"])
    >>> 
    >>> myIT2FLS = Type2Model()
    >>> myIT2FLS.add_input_variable("x1")
    >>> myIT2FLS.add_input_variable("x2")
    >>> myIT2FLS.add_output_variable("y1")
    >>> myIT2FLS.add_output_variable("y2")
    >>> 
    >>> myIT2FLS.add_rule([("x1", Small), ("x2", Small)], [("y1", Small), ("y2", Large)])
    >>> myIT2FLS.add_rule([("x1", Medium), ("x2", Medium)], [("y1", Medium), ("y2", Small)])
    >>> myIT2FLS.add_rule([("x1", Large), ("x2", Large)], [("y1", Large), ("y2", Small)])
    >>> 
    >>> it2out, tr = myIT2FLS.evaluate({"x1":0.9, "x2":0.9}, min_t_norm, max_s_norm, domain)
    >>> it2out["y1"].plot()
    >>> TR_plot(domain, tr["y1"])
    >>> print(crisp(tr["y1"]))
    >>> 
    >>> it2out["y2"].plot()
    >>> TR_plot(domain, tr["y2"])
    >>> print(crisp(tr["y2"]))
    >>> 
    
    Notes
    -----
    
    While using the PyIT2FLS some cares must be taken by the user 
    himself which are listed as below:
        
        * The UMF defined for an IT2FS must be greater than or equal with the LMF at all points of the discrete universe of discourse. 
        * The inputs and outputs defined must be compatible while adding the rules and evluating the IT2FLS.
    """
    def __init__(self,treino,order):
        self.inputs = []
        self.outputs = []
        self.rules = []
        
        self.training_data = treino
        self.order = order


        self.number_rules = 0
        
    def __repr__(self):
        pass
    
    def config_inicial(self,data,n_sets):
        """
        Faz os calculos iniciais dos paramentros para o resto das operações'
        """
        
        self.data=data
  
        'Prepara o universo de discurso (dominio) para a criação dos sets fuzzy'
        
        minimo = np.nanmin(data)
        maximo = np.nanmax(data)
        
        'Verifica o sinal dos valores em data.'         
                    
        self.dominio_inf = np.float64(minimo * 1.1 if minimo < 0 else minimo * 0.9)
        self.dominio_sup = np.float64(maximo * 1.1 if maximo > 0 else maximo * 0.9)
        
        pontos = self.dominio_sup - self.dominio_inf
        domain_length = pontos
                
        'universo de discurso'

        if domain_length > 1000: 
            self.domain = linspace(self.dominio_inf, self.dominio_sup, int(domain_length))
        else:
            self.domain = linspace(self.dominio_inf, self.dominio_sup, 1000)
           

        self.numero_de_sets = n_sets
        
        self.intervalo_entre_set = pontos/(self.numero_de_sets+1) #Define os pontos em que cada set se inicia        
        self.largura_set = pontos/(self.numero_de_sets) #Define os pontos em que cada set se inicia


        

    def chen_model_sobreposto(self,numero_de_sets): 
        """ 
        Cria os conjuntos fuzzy SOBREPOSTOS 
        
        :data: dados da serie temporal analisada
        """
        
        'configurações inciais'
        Type2Model.config_inicial(self,self.training_data,numero_de_sets)
        
        pontos_conj = []
        for x in range(self.numero_de_sets+2):  # descobre os pontos de inicio de cada conjunto
            pontos_conj.append(self.dominio_inf + (self.intervalo_entre_set * x))
            

        intervalos_conjuntos = []
        'Constroi os intervalos de cada set'
        for x in range(1,self.numero_de_sets+1):
            aux = [pontos_conj[x-1],pontos_conj[x],pontos_conj[x+1]]
            intervalos_conjuntos.append(aux)
                                        
        dict_sets = {}   #Dicionário contendo os sets
        
        'Constroi cada set a medida que avança na lista de intervalos de conjuntos'
        for x in range(1,self.numero_de_sets+1):
      
            r,t,y = intervalos_conjuntos[x-1]
            b_esq = r        #Base esquerda
            topo_tri = t     #Topo do triangulo
            b_dir = y  
            
            #print(b_esq,topo_tri,b_dir)
            
            fou_right = (b_dir-topo_tri)*0.4        #A mancha nao pode ser maior dos que os vertices do triangulo
            fou_left = (topo_tri-b_esq)*0.4        #Calcula a mancha da esquerda e direita e pega a menor para valer para os dois
            #fou = min(fou_left,fou_right)
            
            nome = 'A%d'%x  #manda junto o nome do set para usar se precisar
            dict_sets['A%d' %x] = FuzzySet(self.domain, tri_mf, [b_esq, topo_tri, b_dir, 1],tri_mf, [b_esq+fou_left, topo_tri, b_dir-fou_right, 0.9],nome = nome)

        self.dict_sets = dict_sets
    
    
    
    def chen_model_sequencial(self,numero_de_sets):
        """ 
        Cria os conjuntos fuzzy SEQUENCIAIS
        
        :data: dados da serie temporal analisada
        """
        
        'configurações inciais'
        Type2Model.config_inicial(self,self.training_data,numero_de_sets)
        
        pontos_conj = []
        for x in range(self.numero_de_sets):  # descobre os pontos de inicio de cada conjunto
            pontos_conj.append(self.dominio_inf + (self.largura_set * x))
        
        intervalos_conjuntos = []
        'Constroi os intervalos de cada set'
        for x in range(self.numero_de_sets):
            aux = [pontos_conj[x],pontos_conj[x]+self.largura_set/2,pontos_conj[x]+self.largura_set]
            intervalos_conjuntos.append(aux)
                
        dict_sets = {}   #Dicionário contendo os sets
        
        'Constroi cada set a medida que avança na lista de intervalos de conjuntos'
        for x in range(1,self.numero_de_sets+1):
      
            r,t,y = intervalos_conjuntos[x-1]
            b_esq = r        #Base esquerda
            topo_tri = t     #Topo do triangulo
            b_dir = y  
            
            fou_right = (b_dir-topo_tri)*0.3        #A mancha nao pode ser maior dos que os vertices do triangulo
            fou_left = (topo_tri-b_esq)*0.3         #Calcula a mancha da esquerda e direita e pega a menor para valer para os dois
            fou = min(fou_left,fou_right)
            
            dict_sets['A%d' %x] = FuzzySet(self.domain,tri_mf, [b_esq, topo_tri, b_dir, 1],tri_mf, [b_esq+fou, topo_tri, b_dir-fou, 0.6])

        self.dict_sets = dict_sets
    
    
    def treinamento(self):      
        """
        Fuzzyfica os valores da serie de treino para encontrar os conjuntos fuzzy correspondentes
        a cada um deles.  Pode ser usada para conjuntos sobrepostos e sequenciais
        
        Parametros:
        :dict_sets: dictionary containing all the sets created
        :order: order of the model ( n-th order)
        treino - list of values to be trained
            
        Return
        
        :lista_regras: List containing the rules for the model according to its order
            For order = 1, lista_regras = (antec,conseq)
            For order = 2, lista_regras = (antec2,antec,conseq)
            For order = 3, lista_regras = (antec3,antec2,antec,conseq)

        :pert_val: List with the following structure for each value of the time series: (sample,value,sets_activated (upper membership)(lower membeship))
        :dict_conj: dictionary where the keys are the samples and the values are the activated sets
        
        OBS: uma outra alternativa é pegar os limites dos conjuntos criados e analisar
        onde o valor de treinamento está baseado nisso.ex: a1 = [2,4,6], a2=[4,6,8].
        lembre-se de criar um array que cobre todos os valores do triangulo e ordene do
        menor para o maior.
        
        """
        treino = self.training_data
        order = self.order

        pert_val = []        #lista com valores de pertinencia superior e inferior de cada valor de treinamento
        conj_ativados = []  #lista sem valores de pertinencia
        dict_conj = {}
        
        amostra = 1 #representa a amostra  que esta sendo analisada do TAIEX (data de analise M/D/ANO)
        
        'alpha-cut é o valor minimo de pertinencia que um valor de treinamento deve ter para ser considerado'
        'como uma regra. Deve-se escolher pois se for zero, podemos ter regras insignificantes sendo adicionadas, gerando ruido'
        
        alpha_cut = 0.1
        ultimo = list(self.dict_sets.values())
        ultimo = ultimo[-1]   #Ultimo conjunto da lista de conjuntos
    
        'Objetivo aqui é descobrir quais conjuntos foram ativados a cada valor de treinamento'
        for x in treino:   #itera sobre todos os valores de treinamento
            conj = 1
            conj_aux = []
            pert_val.append(amostra)
            pert_val.append(x)
            conj_temp = []
            for it2fs in self.dict_sets.values():                         #itera sobre todos os sets
                try:
                    u = it2fs.umf(x, it2fs.umf_params)  
                except Exception as ex:
                    print("Valor de entrada: ", x)
                    print("Parâmetros", it2fs.umf_params)
                    
                #u = it2fs.umf(x, it2fs.umf_params)     #calcula pertinencia do valor x na func. pert. superior
                l = it2fs.lmf(x, it2fs.lmf_params)     #calcula pertinencia do valor x na func. pert. inferior               
                if u >= alpha_cut:                           #pertinencia = 0 : nao toca o set  
                    conj_temp.append(conj)            #conjunto sendo ativado eh adicionado a lista temporaria
                    conj_aux.append(conj)
                    tuple_aux = [str('A%d'%conj),u,l]    #para cada valor de treino(x) tem os seguintes valores: [conjunto,pert.sup.,pert.inf.]
                    pert_val.append(tuple_aux)
                elif u > 0.0:                                 #Se for o ultimo conjunto, define este como o ativado ( o ultimo conjunto possui uma parte que nao é sobreposta)
                    if it2fs == ultimo:
                        conj_temp.append(conj)
                conj = conj+1                             #Faz rodar por todos os conjuntos existentes
            conj_ativados.append(conj_temp)                   #Adiciona a lista a uma lista de lista contendo conjuntos ativados a cada valor de treino
            pert_val.append('f')                            #Utilizando 'f' para indicar a passagem de um valor de treino para outro
            dict_conj['%d'%amostra] = conj_aux
            amostra = amostra +1     #atualiza a amostra 
                        

        'conj_ativados: mostra os conj. ativados a cada valor de treinamento'
        
        associations = conj_ativados
        
        lista_regras = []
       
        'Agora precisamos analisar e pegar os valores de conjuntos ativados para cada valor de x'
        'Dependendo da ordem de sistema escolhida: Ex. ordem= 2 ->  f(t-1)+f(t) -> f(t+1)'
        for i in range(order-1, len(associations)-1):             #Começa no n° de ordem porque se for antes nao tem como pegar valores anteriores
            relations = []
            for x in range(order-1,-1,-1):    # Vai de ordem até 0, de 1 em 1
                relations.append(associations[i-x])
            relations.append(associations[i+1])
            temp = list(product(*relations))  #faz a distributiva entre os valores
            for x in temp:
                if x not in lista_regras:  #Só adiciona regra se ela for NOVA
                    lista_regras.append(x)   #lista de 3-tuples contendo as regras
                    
                    
        print("Número de regras: ", len(lista_regras))
        
        self.number_rules = len(lista_regras)
        
        
        'Adiciona a lista de regras ao modelo'
        
        for regra in lista_regras:
            
            regras_antecedente =[]   #Lista que terá as regras do antecedente
            regras_consequente = []  #Lista que terá as regras do consequente
            aux = len(regra) - 1    #analisa o tamanho do tuple de regra. Dependendo do tamanho corresponde a n-ordem
            cont = 1
            indice = -1   #variavel que sera usada como indice do tuple de regra para pegar os valores de forma decrescente
            
            for x in regra[:aux]:   #itera sobre todas as regras           
                indice = indice - 1
                tup = ("x%d"%cont, self.dict_sets["A%d"%(regra[indice])])
                regras_antecedente.append(tup)   #produz um tuple com tamanho igual ao numero de antecedentes
                cont = cont + 1
                     
            tup = ("y1", self.dict_sets['A%d'%(regra[-1])])  # constroi o consequente ao final dos antecedentes
            regras_consequente.append(tup)
            self.add_rule(regras_antecedente,regras_consequente)  #Adiciona as regras ao modelo de FLS 

  
        #return lista_regras,pert_val,dict_conj
            
            
    def add_variables(self):
        
        """Função que adiciona as variaveis de entrada e saida ao sistema 
        fuzzy de acordo com a ordem do sistema.
        
        O usuário final não precisa utilizar essa função. Ela já é chamada
        automaticamente pela função 'predict'.
        """
        
        
        'Adiciona variaveis de entrada de acordo com a ordem do sistema'
        for x in range(1,self.order+1):
            self.add_input_variable("x%d"%x)

        self.add_output_variable("y1")
        
        
        
    def predict(self,teste_func):     
        """
        Função de previsão. Avalia cada valor de teste e pode ser aplicada para n-th order
        Funciona da seguinte forma: o for abaixo (do y) associa:
            x1 - atual
            x2 - anterior
            x3 - anterior do anterior, etc.
        """
        
        'Adiciona variaveis de entrada de acordo com a ordem do sistema'
        Type2Model.add_variables(self)

        '-------------------------------------------- Fuzzyficar e fazer previsão -----------------------------------------'
               
        inputs = {}   #Dicionário de entradas a serem treinadas
        previsao = []
        
        'O loop comeca no item relativo a ordem do modelo (para ter instancias passadas para usar na previsao)'
                    
        for x in range(self.order-1,len(teste_func)):
            #print('xxxxx',x,'',end = '')
            aux = x    
            for y in range(1,self.order+1):   #Itera entre as entradas "x" dependendo da ordem
                       
                inputs["x%d"%y] = teste_func[aux]
                #print(teste_func[aux])
                aux = aux-1
                

            it2out, tr = Type2Model.evaluate(self,inputs, min_t_norm, max_s_norm, self.domain, algorithm="KM")
            res = crisp(tr["y1"])    #'Encontra o valor defuzificado' 
            
            'Caso o conjunto ativado pela amostra nao tenha nenhuma regra, a previsao sera zero. Nesse caso usa a previsao naive (repete o ultimo valor)'
            if(res == 0.0):     
                res = teste_func[x-1]
                #print("Previsao naive - x:",x,"teste:",teste_func[x],"teste_func:",teste_func[x-1],"res:",res)
            
            previsao.append(res) #'preenche o vetor previsao com o novo valor defuzificado' 
                    
        return previsao


    
    def treinamento_multiplas_entradas(self,dict_sets, order, data_close,data_lower,data_high):      
        """
        Fuzzyfica os valores das series de treino para encontrar os conjuntos fuzzy correspondentes
        a cada um deles.  Pode ser usada para conjuntos sobrepostos e sequenciais
        
        Parametros:
        :dict_sets: dictionary containing all the sets created
        :order: order of the model ( n-th order)
        *data - lists of values to be trained ( usually 3)
            
        Return
        
        :lista_regras: List containing the rules for the model according to its order
            lista_regras = (antec_close,antec_lower,antec_high,conseq_close)
            

        :pert_val: List with the following structure for each value of the time series: (sample,value,sets_activated (upper membership)(lower membeship))
        :dict_conj: dictionary where the keys are the samples and the values are the activated sets
        
        OBS: uma outra alternativa é pegar os limites dos conjuntos criados e analisar
        onde o valor de treinamento está baseado nisso.ex: a1 = [2,4,6], a2=[4,6,8].
        lembre-se de criar um array que cobre todos os valores do triangulo e ordene do
        menor para o maior.
        
        """
        
        'alpha-cut é o valor minimo de pertinencia que um valor de treinamento deve ter para ser considerado'
        'como uma regra. Deve-se escolher pois se for zero, podemos ter regras insignificantes sendo adicionadas, gerando ruido'
        
        alpha_cut = 0.1


        pert_val = []        #lista com valores de pertinencia superior e inferior de cada valor de treinamento
        conj_ativados = []  #lista sem valores de pertinencia
        dict_conj = {}
        
        amostra = 1 #representa a amostra  que esta sendo analisada do TAIEX (data de analise M/D/ANO)
        
        'Objetivo aqui é descobrir quais conjuntos foram ativados a cada valor de treinamento'
        for x in data_close:   #itera sobre todos os valores de treinamento
            conj = 1
            conj_aux = []
            pert_val.append(amostra)
            pert_val.append(x)
            conj_temp = []
            for it2fs in dict_sets.values():                         #itera sobre todos os sets
                u = it2fs.umf(x, it2fs.umf_params)     #calcula pertinencia do valor x na func. pert. superior
                l = it2fs.lmf(x, it2fs.lmf_params)     #calcula pertinencia do valor x na func. pert. inferior               
                if u >= alpha_cut:                           #pertinencia = 0 : nao toca o set  
                    conj_temp.append(conj)            #conjunto sendo ativado eh adicionado a lista temporaria
                    conj_aux.append(conj)
                    tuple_aux = [str('A%d'%conj),u,l]    #para cada valor de treino(x) tem os seguintes valores: [conjunto,pert.sup.,pert.inf.]
                    pert_val.append(tuple_aux)                 
                conj = conj+1                             #Faz rodar por todos os conjuntos existentes
            conj_ativados.append(conj_temp)                   #Adiciona a lista a uma lista de lista contendo conjuntos ativados a cada valor de treino
            pert_val.append('f')                            #Utilizando 'f' para indicar a passagem de um valor de treino para outro
            dict_conj['%d'%amostra] = conj_aux
            amostra = amostra +1     #atualiza a amostra 
                        

        'conj_ativados: mostra os conj. ativados a cada valor de treinamento'
        
        associations = conj_ativados
               
        regras_close = []

        'Agora precisamos analisar e pegar os valores de conjuntos ativados para cada valor de x'
        'Dependendo da ordem de sistema escolhida: Ex. ordem= 2 ->  f(t-1)+f(t) -> f(t+1)'
        for i in range(order-1, len(associations)-1):             #Começa no n° de ordem porque se for antes nao tem como pegar valores anteriores
            relations = []
            for x in range(order-1,-1,-1):    # Vai de ordem até 0, de 1 em 1
                relations.append(associations[i-x])
            relations.append(associations[i+1])
            temp = list(product(*relations))  #faz a distributiva entre os valores
            for x in temp:
                regras_close.append(x)   #lista de 3-tuples contendo as regras
		    





        pert_val = []        #lista com valores de pertinencia superior e inferior de cada valor de treinamento
        conj_ativados = []  #lista sem valores de pertinencia
        dict_conj = {}
        
        amostra = 1 #representa a amostra  que esta sendo analisada do TAIEX (data de analise M/D/ANO)
    
        'Objetivo aqui é descobrir quais conjuntos foram ativados a cada valor de treinamento'
        for x in data_lower:   #itera sobre todos os valores de treinamento
            conj = 1
            conj_aux = []
            pert_val.append(amostra)
            pert_val.append(x)
            conj_temp = []
            for it2fs in dict_sets.values():                         #itera sobre todos os sets
                u = it2fs.umf(x, it2fs.umf_params)     #calcula pertinencia do valor x na func. pert. superior
                l = it2fs.lmf(x, it2fs.lmf_params)     #calcula pertinencia do valor x na func. pert. inferior               
                if u >= alpha_cut:                           #pertinencia = 0 : nao toca o set  
                    conj_temp.append(conj)            #conjunto sendo ativado eh adicionado a lista temporaria
                    conj_aux.append(conj)
                    tuple_aux = [str('A%d'%conj),u,l]    #para cada valor de treino(x) tem os seguintes valores: [conjunto,pert.sup.,pert.inf.]
                    pert_val.append(tuple_aux)                 
                conj = conj+1                             #Faz rodar por todos os conjuntos existentes
            conj_ativados.append(conj_temp)                   #Adiciona a lista a uma lista de lista contendo conjuntos ativados a cada valor de treino
            pert_val.append('f')                            #Utilizando 'f' para indicar a passagem de um valor de treino para outro
            dict_conj['%d'%amostra] = conj_aux
            amostra = amostra +1     #atualiza a amostra 
                        

        'conj_ativados: mostra os conj. ativados a cada valor de treinamento'
        associations = conj_ativados
        
        regras_lower = []

                
        'Agora precisamos analisar e pegar os valores de conjuntos ativados para cada valor de x'
        'Dependendo da ordem de sistema escolhida: Ex. ordem= 2 ->  f(t-1)+f(t) -> f(t+1)'
        for i in range(order-1, len(associations)-1):             #Começa no n° de ordem porque se for antes nao tem como pegar valores anteriores
            relations = []
            for x in range(order-1,-1,-1):    # Vai de ordem até 0, de 1 em 1
                relations.append(associations[i-x])
            relations.append(associations[i+1])
            temp = list(product(*relations))  #faz a distributiva entre os valores
            for x in temp:
                regras_lower.append(x)   #lista de 3-tuples contendo as regras


        pert_val = []        #lista com valores de pertinencia superior e inferior de cada valor de treinamento
        conj_ativados = []  #lista sem valores de pertinencia
        dict_conj = {}
        
        amostra = 1 #representa a amostra  que esta sendo analisada do TAIEX (data de analise M/D/ANO)
    
        'Objetivo aqui é descobrir quais conjuntos foram ativados a cada valor de treinamento'
        for x in data_high:   #itera sobre todos os valores de treinamento
            conj = 1
            conj_aux = []
            pert_val.append(amostra)
            pert_val.append(x)
            conj_temp = []
            for it2fs in dict_sets.values():                         #itera sobre todos os sets
                u = it2fs.umf(x, it2fs.umf_params)     #calcula pertinencia do valor x na func. pert. superior
                l = it2fs.lmf(x, it2fs.lmf_params)     #calcula pertinencia do valor x na func. pert. inferior               
                if u >= alpha_cut:                           #pertinencia = 0 : nao toca o set  
                    conj_temp.append(conj)            #conjunto sendo ativado eh adicionado a lista temporaria
                    conj_aux.append(conj)
                    tuple_aux = [str('A%d'%conj),u,l]    #para cada valor de treino(x) tem os seguintes valores: [conjunto,pert.sup.,pert.inf.]
                    pert_val.append(tuple_aux)                 
                conj = conj+1                             #Faz rodar por todos os conjuntos existentes
            conj_ativados.append(conj_temp)                   #Adiciona a lista a uma lista de lista contendo conjuntos ativados a cada valor de treino
            pert_val.append('f')                            #Utilizando 'f' para indicar a passagem de um valor de treino para outro
            dict_conj['%d'%amostra] = conj_aux
            amostra = amostra +1     #atualiza a amostra 
                        

        'conj_ativados: mostra os conj. ativados a cada valor de treinamento'
        associations = conj_ativados

        regras_high = []
        
        'Agora precisamos analisar e pegar os valores de conjuntos ativados para cada valor de x'
        'Dependendo da ordem de sistema escolhida: Ex. ordem= 2 ->  f(t-1)+f(t) -> f(t+1)'
        for i in range(order-1, len(associations)-1):             #Começa no n° de ordem porque se for antes nao tem como pegar valores anteriores
            relations = []
            for x in range(order-1,-1,-1):    # Vai de ordem até 0, de 1 em 1
                relations.append(associations[i-x])
            relations.append(associations[i+1])
            temp = list(product(*relations))  #faz a distributiva entre os valores
            for x in temp:
                regras_high.append(x)   #lista de 3-tuples contendo as regras

        '-------------------------------------------------------------------------------------------------------'
        
        
    	#aqui tenho 3 listas de regras para juntar
                
                
        """
        vamos construir a lista de regras . Ex. para order = 2:
        regra = [2,3,5,4,6,8,5]
        
        nesse casso 2 e 3 sao antecedentes de high
        5 e 4 antecedentes de lower
        6 e 8 antecedentes de close
        e o ultimo numero 5 é consequente de close          
        """
        
        lista_regras = []
        
        'passa para lista para concatenar depois'                
        for x in range(len(regras_close)):
            regra1 = list(regras_close[x])
            consequente = [regra1[-1]]
            regra1 = regra1[:-1]
            
            regra2 = list(regras_lower[x])
            regra2 = regra2[:-1]
                        
            regra3 = list(regras_high[x])
            regra3 = regra3[:-1]
            
            regra = regra3 + regra2 + regra1 + consequente #concatena os valores
            
            regra = tuple(regra)      #transforma em tuple para ser inserido depois na função de add regras
                
            if regra not in lista_regras:    #so adiciona a regra se ela for NOVA
                lista_regras.append(regra)
            
        print("Número de regras: ", len(lista_regras))
        
        
        'Adiciona a lista de regras ao modelo'
        
        for regra in lista_regras:
            
   
            regras_antecedente =[]   #Lista que terá as regras do antecedente
            regras_consequente = []  #Lista que terá as regras do consequente
            aux = len(regra) - 1    #analisa o tamanho do tuple de regra. Dependendo do tamanho corresponde a n-ordem
            cont = 1
            indice = -1   #variavel que sera usada como indice do tuple de regra para pegar os valores de forma decrescente
            
            for x in regra[:aux]:   #itera sobre todas as regras           
                indice = indice - 1
                tup = ("x%d"%cont, dict_sets["A%d"%(regra[indice])])
                regras_antecedente.append(tup)   #produz um tuple com tamanho igual ao numero de antecedentes
                cont = cont + 1
                     
            tup = ("y1", dict_sets['A%d'%(regra[-1])])  # constroi o consequente ao final dos antecedentes
            regras_consequente.append(tup)
            self.add_rule(regras_antecedente,regras_consequente)  #Adiciona as regras ao modelo de FLS (myIT2FLS)
            
            

        return lista_regras,pert_val,dict_conj
    
    
    
    def add_input_variable(self, name):
        """
        Adds new input variable name.
        
        Parameters
        ----------
        
        name:
            str
            
            Name of the new input variable as a str.
        """
        self.inputs.append(name)

    def add_output_variable(self, name):
        """
        Adds new output variable name.
        
        Parameters
        ----------
        
        name:
            str
            
            Name of the new output variable as a str.
        """
        self.outputs.append(name)

    def add_rule(self, antecedent, consequent):
        """
        Adds new rule to the rule base of the IT2FLS.
        
        Parameters
        ----------
        
        antecedent:
            List of tuples
            
            Antecedent is a list of tuples in which each tuple indicates 
            assignement of a variable to an IT2FS. First element of the 
            tuple must be input variable name as str, and the second 
            element of the tuple must be an IT2FS.
            
        consequent:
            List of tuples
            
            Consequent is a list of tuples in which each tuple indicates 
            assignement of a variable to an IT2FS. First element of the 
            tuple must be output variable name as str, and the second 
            element of the tuple must be an IT2FS.
            
        """
        
        self.rules.append((antecedent, consequent))

    def copy(self):
        """
        Returns a copy of the IT2FLS.
        """
        o = Type2Model()
        o.inputs = self.inputs.copy()
        o.outputs = self.outputs.copy()
        o.rules = self.rules.copy()
        return o
    
    def evaluate_list(self, inputs, t_norm, s_norm, domain, 
                      method="Centroid", method_params=None, 
                      algorithm="EIASC", algorithm_params=None):
        """
        Evaluates the IT2FLS based on list of crisp inputs given by user.
        
        Parameters
        ----------
        
        inputs:
            dictionary
            
            Inputs is a dictionary in which the keys are input variable 
            names as str and the values are the list of crisp values 
            corresponded with the inputs to be evaluated.
            
        t_norm:
            function
            
            Indicates the t-norm operator to be used, and should be chosen 
            between min_t_norm, product_t_norm, or other user defined 
            t-norms.
        
        s_norm:
            function
            
            Indicates the s-norm operator to be used, and should be chosen 
            between max_s_norm or other user defined s-norms.
        
        domain:
            numpy (n,) shaped array
            
            Indicates the universe of discourse dedicated to the IT2FS.
        
        method="Centroid":
            str
            
            Indicates the type reduction method name and should be one 
            of the methods listed below:
                Centroid, CoSet, CoSum, Height, and ModiHe.
        
        method_params=None:
            List
            
            Parameters of the type reduction method, if needed.
        
        algorithm="EIASC":
            Indicates the type reduction algorithm name and should be 
            one of the algorithms listed below:
                KM, EKM, WEKM, TWEKM, EIASC, WM, BMM, LBMM, and NT.
        
        algorithm_params=None:
            List
            
            Parameters of the type reduction algorithm, if needed.
        
        Returns
        -------
        It depends on which method and algorithm for type reduction is 
        chosen. If Centroid type reduction method is chosen the output 
        is a tuple with two elements. First element is the overall IT2FS 
        outputs of the system as a list of dictionaries with output names as keys 
        and sets as values. The second output is outputs of the 
        selected type reduction algorithm as a list of dictionaries with 
        output names as keys and type reduction algorithm function output 
        as value. For other type reduction methods the only output is a list of  
        dictionaries of the type reduction algorithm function outputs for 
        each output variable name as a key.
        
        Notes
        -----
        
        While using the evaluate function some cares must be taken by the user 
        himself which are listed as below:
            * The inputs must be lay in the defined universe of discourse.
            * The type reduction method and the type reduction algorithm must be selected from the lists provided in docstrings.
        """
        inputs_list = []
        
        l = len(inputs[self.inputs[0]])
        for i in inputs.values():
            if len(i) != l:
                raise ValueError("All input lists must contain same number of values.")
        
        for i in range(l):
            inputs_list.append({})
            for j in self.inputs:
                inputs_list[-1][j] = inputs[j][i]
        
        if algorithm == "KM":
            alg_func = KM_algorithm
        elif algorithm == "EKM":
            alg_func = EKM_algorithm
        elif algorithm == "WEKM":
            alg_func = WEKM_algorithm
        elif algorithm == "TWEKM":
            alg_func = TWEKM_algorithm
        elif algorithm == "WM":
            alg_func = WM_algorithm
        elif algorithm == "LBMM":
            alg_func = LBMM_algorithm
        elif algorithm == "BMM":
            alg_func = BMM_algorithm        
        elif algorithm == "NT":
            alg_func = NT_algorithm       
        elif algorithm == "EIASC":
            alg_func = EIASC_algorithm
        else:
            raise ValueError("The " + algorithm + " algorithm is not implemented, yet!")
        
        outputs = []
        
        if method == "Centroid":
            Cs = []
            TRs = []
            for inputs in inputs_list:
                B = {out: [] for out in self.outputs}
                for rule in self.rules:
                    u = 1
                    l = 1
                    for input_statement in rule[0]:
                        u = t_norm(u, input_statement[1].umf(inputs[input_statement[0]], input_statement[1].umf_params))
                        l = t_norm(l, input_statement[1].lmf(inputs[input_statement[0]], input_statement[1].lmf_params))
                    for consequent in rule[1]:
                        B_l = meet(domain, FuzzySet(domain, const_mf, [u], const_mf, [l]), consequent[1], t_norm)
                        B[consequent[0]].append(B_l)
                C = {out: FuzzySet(domain) for out in self.outputs}
                TR = {}
                for out in self.outputs:
                    for B_l in B[out]:
                        C[out] = join(domain, C[out], B_l, s_norm)
                    TR[out] = Centroid(C[out], alg_func, domain, alg_params=algorithm_params)
                Cs.append(C)
                TRs.append(TR)
            return Cs, TRs
        elif method == "CoSet":
            for inputs in inputs_list:
                F = []
                G = {out: [] for out in self.outputs}
                for rule in self.rules:
                    u = 1
                    l = 1
                    for input_statement in rule[0]:
                        u = t_norm(u, input_statement[1].umf(inputs[input_statement[0]], input_statement[1].umf_params))
                        l = t_norm(l, input_statement[1].lmf(inputs[input_statement[0]], input_statement[1].lmf_params))
                    F.append([l, u])
                    for consequent in rule[1]:
                        G[consequent[0]].append(consequent[1])
                TR = {}
                for out in self.outputs:
                    TR[out] = CoSet(array(F), G[out], alg_func, domain, alg_params=algorithm_params)
                outputs.append(TR)
            return outputs
        elif method == "CoSum":
            for inputs in inputs_list:
                B = {out: [] for out in self.outputs}
                for rule in self.rules:
                    u = 1
                    l = 1
                    for input_statement in rule[0]:
                        u = t_norm(u, input_statement[1].umf(inputs[input_statement[0]], input_statement[1].umf_params))
                        l = t_norm(l, input_statement[1].lmf(inputs[input_statement[0]], input_statement[1].lmf_params))
                    for consequent in rule[1]:
                        B_l = meet(domain, FuzzySet(domain, const_mf, [u], const_mf, [l]), consequent[1], t_norm)
                        B[consequent[0]].append(B_l)
                TR = {}
                for out in self.outputs:
                    TR[out] = CoSum(B[out], alg_func, domain, alg_params=algorithm_params)
                outputs.append(TR)
            return outputs
        elif method == "Height":
            for inputs in inputs_list:
                B = {out: [] for out in self.outputs}
                for rule in self.rules:
                    u = 1
                    l = 1
                    for input_statement in rule[0]:
                        u = t_norm(u, input_statement[1].umf(inputs[input_statement[0]], input_statement[1].umf_params))
                        l = t_norm(l, input_statement[1].lmf(inputs[input_statement[0]], input_statement[1].lmf_params))
                    for consequent in rule[1]:
                        B_l = meet(domain, FuzzySet(domain, const_mf, [u], const_mf, [l]), consequent[1], t_norm)
                        B[consequent[0]].append(B_l)
                TR = {}
                for out in self.outputs:
                    TR[out] = Height(B[out], alg_func, domain, alg_params=algorithm_params)
                outputs.append(TR)
            return outputs
        elif method == "ModiHe":
            for inputs in inputs_list:
                B = {out: [] for out in self.outputs}
                for rule in self.rules:
                    u = 1
                    l = 1
                    for input_statement in rule[0]:
                        u = t_norm(u, input_statement[1].umf(inputs[input_statement[0]], input_statement[1].umf_params))
                        l = t_norm(l, input_statement[1].lmf(inputs[input_statement[0]], input_statement[1].lmf_params))
                    for consequent in rule[1]:
                        B_l = meet(domain, FuzzySet(domain, const_mf, [u], const_mf, [l]), consequent[1], t_norm)
                        B[consequent[0]].append(B_l)
                TR = {}
                for out in self.outputs:
                    TR[out] = ModiHe(B[out], method_params, alg_func, domain, alg_params=algorithm_params)
                outputs.append(TR)
            return outputs
        else:
            raise ValueError("The method " + method + " is not implemented yet!")
    
    
    def evaluate(self, inputs, t_norm, s_norm, domain, method="Centroid", 
                 method_params=None, algorithm="EIASC", algorithm_params=None):
        """
        Evaluates the IT2FLS based on crisp inputs given by user.
        
        Parameters
        ----------
        
        inputs:
            dictionary
            
            Inputs is a dictionary in which the keys are input variable 
            names as str and the values are the crisp value of inputs to 
            be evaluated.
            
        t_norm:
            function
            
            Indicates the t-norm operator to be used, and should be chosen 
            between min_t_norm, product_t_norm, or other user defined 
            t-norms.
        
        s_norm:
            function
            
            Indicates the s-norm operator to be used, and should be chosen 
            between max_s_norm or other user defined s-norms.
        
        domain:
            numpy (n,) shaped array
            
            Indicates the universe of discourse dedicated to the IT2FS.
        
        method="Centroid":
            str
            
            Indicates the type reduction method name and should be one 
            of the methods listed below:
                Centroid, CoSet, CoSum, Height, and ModiHe.
        
        method_params=None:
            List
            
            Parameters of the type reduction method, if needed.
        
        algorithm="EIASC":
            Indicates the type reduction algorithm name and should be 
            one of the algorithms listed below:
                KM, EKM, WEKM, TWEKM, EIASC, WM, BMM, LBMM, and NT.
        
        algorithm_params=None:
            List
            
            Parameters of the type reduction algorithm, if needed.
        
        Returns
        -------
        It depends on which method and algorithm for type reduction is 
        chosen. If Centroid type reduction method is chosen the output 
        is a tuple with two elements. First element is the overall IT2FS 
        outputs of the system as a dictionary with output names as keys 
        and sets as values. The second output is outputs of the 
        selected type reduction algorithm as a dictionary with 
        output names as keys and type reduction algorithm function output 
        as value. For other type reduction methods the only output is the 
        dictionary of the type reduction algorithm function outputs for 
        each output variable name as a key.
        
        Notes
        -----
        
        While using the evaluate function some cares must be taken by the user 
        himself which are listed as below:
            * The inputs must be lay in the defined universe of discourse.
            * The type reduction method and the type reduction algorithm must be selected from the lists provided in docstrings.
        """
        if algorithm == "KM":
            alg_func = KM_algorithm
        elif algorithm == "EKM":
            alg_func = EKM_algorithm
        elif algorithm == "WEKM":
            alg_func = WEKM_algorithm
        elif algorithm == "TWEKM":
            alg_func = TWEKM_algorithm
        elif algorithm == "WM":
            alg_func = WM_algorithm
        elif algorithm == "LBMM":
            alg_func = LBMM_algorithm
        elif algorithm == "BMM":
            alg_func = BMM_algorithm        
        elif algorithm == "NT":
            alg_func = NT_algorithm       
        elif algorithm == "EIASC":
            alg_func = EIASC_algorithm
        else:
            raise ValueError("The " + algorithm + " algorithm is not implemented, yet!")
            
        if method == "Centroid":
            B = {out: [] for out in self.outputs}
            for rule in self.rules:
                u = 1
                l = 1
                for input_statement in rule[0]:  #cada regra eh: [[(x1,antecedente)],[(y1,consequente)]] entao input_statement e uma lista de tuplas de antecedentes
                    u = t_norm(u, input_statement[1].umf(inputs[input_statement[0]], input_statement[1].umf_params)) #calcula a pertinencia de cada input no set da regra
                    l = t_norm(l, input_statement[1].lmf(inputs[input_statement[0]], input_statement[1].lmf_params))
                for consequent in rule[1]:
                    B_l = meet(domain, FuzzySet(domain, const_mf, [u], const_mf, [l]), consequent[1], t_norm)
                    B[consequent[0]].append(B_l)
  
            C = {out: FuzzySet(domain) for out in self.outputs}
            TR = {}
            for out in self.outputs:
                for B_l in B[out]:
                    C[out] = join(domain, C[out], B_l, s_norm)
                TR[out] = Centroid(C[out], alg_func, domain, alg_params=algorithm_params)

            return C, TR
        elif method == "CoSet":
            F = []
            G = {out: [] for out in self.outputs}
            for rule in self.rules:
                u = 1
                l = 1
                for input_statement in rule[0]:
                    u = t_norm(u, input_statement[1].umf(inputs[input_statement[0]], input_statement[1].umf_params))
                    l = t_norm(l, input_statement[1].lmf(inputs[input_statement[0]], input_statement[1].lmf_params))
                F.append([l, u])
                for consequent in rule[1]:
                    G[consequent[0]].append(consequent[1])
            TR = {}
            for out in self.outputs:
                TR[out] = CoSet(array(F), G[out], alg_func, domain, alg_params=algorithm_params)
            return TR
        elif method == "CoSum":
            B = {out: [] for out in self.outputs}
            for rule in self.rules:
                u = 1
                l = 1
                for input_statement in rule[0]:
                    u = t_norm(u, input_statement[1].umf(inputs[input_statement[0]], input_statement[1].umf_params))
                    l = t_norm(l, input_statement[1].lmf(inputs[input_statement[0]], input_statement[1].lmf_params))
                for consequent in rule[1]:
                    B_l = meet(domain, FuzzySet(domain, const_mf, [u], const_mf, [l]), consequent[1], t_norm)
                    B[consequent[0]].append(B_l)
            TR = {}
            for out in self.outputs:
                TR[out] = CoSum(B[out], alg_func, domain, alg_params=algorithm_params)
            return TR
        elif method == "Height":
            B = {out: [] for out in self.outputs}
            for rule in self.rules:
                u = 1
                l = 1
                for input_statement in rule[0]:
                    u = t_norm(u, input_statement[1].umf(inputs[input_statement[0]], input_statement[1].umf_params))
                    l = t_norm(l, input_statement[1].lmf(inputs[input_statement[0]], input_statement[1].lmf_params))
                for consequent in rule[1]:
                    B_l = meet(domain, FuzzySet(domain, const_mf, [u], const_mf, [l]), consequent[1], t_norm)
                    B[consequent[0]].append(B_l)
            TR = {}
            for out in self.outputs:
                TR[out] = Height(B[out], alg_func, domain, alg_params=algorithm_params)
            return TR
        elif method == "ModiHe":
            B = {out: [] for out in self.outputs}
            for rule in self.rules:
                u = 1
                l = 1
                for input_statement in rule[0]:
                    u = t_norm(u, input_statement[1].umf(inputs[input_statement[0]], input_statement[1].umf_params))
                    l = t_norm(l, input_statement[1].lmf(inputs[input_statement[0]], input_statement[1].lmf_params))
                for consequent in rule[1]:
                    B_l = meet(domain, FuzzySet(domain, const_mf, [u], const_mf, [l]), consequent[1], t_norm)
                    B[consequent[0]].append(B_l)
            TR = {}
            for out in self.outputs:
                TR[out] = ModiHe(B[out], method_params, alg_func, domain, alg_params=algorithm_params)
            return TR
        else:
            raise ValueError("The method " + method + " is not implemented yet!")

























