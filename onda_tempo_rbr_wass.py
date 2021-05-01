# Calculo das distribuicoes de curto-termo

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from scipy.stats import norm,rayleigh
import scipy.stats as st

plt.close('all')


def time_domain(n1):
    """
    Calcula parametros de onda no dominio do tempo
    Input: data frame da serie de heave
    """

    # acha zeros ascendentes e descrescentest
    # valor de x no nivel medio de cada onda individual
    nmza = []
    nmzd = []
    for i in range(len(n1)-1):
        if n1.values[i] < 0 and n1.values[i+1] > 0:
            x = np.array([[n1.index[i], n1.index[i+1]]]).T
            y = np.array([[n1.values[i][0], n1.values[i+1][0]]]).T
            regr = linear_model.LinearRegression().fit(x, y)
            
            # valor de x no nivel medio de cada onda individual
            nmza.append(float(-regr.intercept_[0]/regr.coef_[0]))

        if n1.values[i] > 0 and n1.values[i+1] < 0:
            x = np.array([[n1.index[i], n1.index[i+1]]]).T
            y = np.array([[n1.values[i][0], n1.values[i+1][0]]]).T
            regr = linear_model.LinearRegression().fit(x, y)

            # valor de x no nivel medio de cada onda individual
            nmzd.append(float(-regr.intercept_[0]/regr.coef_[0]))

    # acha os indices correspondentes aos cruzamentos dos zeros
    # ascendentes e descendentes
    inmza = [n1.index.get_loc(i, method='nearest') for i in nmza]
    inmzd = [n1.index.get_loc(i, method='nearest') for i in nmzd]

    # acha periodo de ZA e ZD
    tza_ind = np.diff(nmza)
    tzd_ind = np.diff(nmzd)

    # acha altura individual de cada onda de ZA
    alt_za_ind = []
    for i in range(len(inmza)-1):
        o = n1.iloc[inmza[i]:inmza[i+1]]
        alt_za_ind.append(float(o.max() - o.min()))

    # acha altura individual de cada onda de ZD
    alt_zd_ind = []
    for i in range(len(inmzd)-1):
        o = n1.iloc[inmzd[i]:inmzd[i+1]]
        alt_zd_ind.append(float(o.max() - o.min()))

    # altura significativa
    hs_za = np.sort(alt_za_ind)[-round(len(alt_za_ind)/3):].mean()
    hs_zd = np.sort(alt_zd_ind)[-round(len(alt_zd_ind)/3):].mean()

    # altura de 1/10 das maiores
    h10_za = np.sort(alt_za_ind)[-round(len(alt_za_ind)/10):].mean()
    h10_zd = np.sort(alt_zd_ind)[-round(len(alt_zd_ind)/10):].mean()

    # altura de 1/100 das maiores
    h100_za = np.sort(alt_za_ind)[-round(len(alt_za_ind)/100):].mean()
    h100_zd = np.sort(alt_zd_ind)[-round(len(alt_zd_ind)/100):].mean()

    # acha as alturas maximas
    hmax_za = np.max(alt_za_ind)
    hmax_zd = np.max(alt_zd_ind)

    return np.array(alt_za_ind), hs_za, h10_za, h100_za, hmax_za


def get_best_distribution(data):
    dist_names = ["norm", "exponweib", "weibull_max", "weibull_min", "pareto", "genextreme"]
    dist_results = []
    params = {}
    for dist_name in dist_names:
        dist = getattr(st, dist_name)
        param = dist.fit(data)

        params[dist_name] = param
        # Applying the Kolmogorov-Smirnov test
        D, p = st.kstest(data, dist_name, args=param)
        print("p value for "+dist_name+" = "+str(p))
        dist_results.append((dist_name, p))

    # select the best fitted distribution
    best_dist, best_p = (max(dist_results, key=lambda item: item[1]))
    # store the name of the best fit and its p value

    print("Best fitting distribution: "+str(best_dist))
    print("Best p value: "+ str(best_p))
    print("Parameters for the best fit: "+ str(params[best_dist]))

    return best_dist, best_p, params[best_dist]

if __name__ == '__main__':

    path_out = os.environ['HOME'] + '/gdrive/coppe/doutorado/output/'

    heave = pd.read_csv(path_out + 'bmop_wave_heave.csv', parse_dates=True, index_col='date')
    heave = pd.read_csv('wassDF_12_5hz.csv', parse_dates=True, index_col='date_rbr')

    # frequencia de amostragem
    fs_wav = 2.0
    dt_wav = 1.0/fs_wav
    #fs_wav = 1.28
    #dt_wav = 1.0/fs_wav

    # vetores de tempo de vento e onda
    tt_wav = np.arange(0, len(heave)*dt_wav, dt_wav)

    #n1 = pd.DataFrame(heave.loc['2015-04-06 23:00:00'].values, index=tt_wav)
    n1 = pd.DataFrame(heave.p02.values, index=tt_wav)

    alt_za_ind, hs_za, h10_za, h100_za, hmax_za = time_domain(n1)
