#%%
# Import Dependencies
from matplotlib.pyplot import figure
from numpy import (absolute, asarray, concatenate, divide, exp, linspace, log,
                   sqrt)
from scipy.stats import anderson, anderson_ksamp, goodness_of_fit, probplot
from scipy.stats.distributions import nct, weibull_min

from pymechlab.tools.stats import k_factor_weibull, mnr_crit

#%%
# Sample Input
xtg = asarray([239, 270, 254, 241, 276, 261, 263, 300, 267, 318,
               259, 286, 284, 238, 237, 317, 313, 271, 264, 309,
               232, 228, 262, 244, 316, 290, 283, 292, 254, 243],
               dtype=float)

xcg = asarray([103, 117, 98.7, 98.0, 80.8, 100.0, 73.8, 69.8,
               116.0, 74.4, 85.9, 107.0, 104.0, 105.0, 97.2, 70.0,
               91.0, 84.3, 93.3, 91.4, 112.0, 112.0, 109.0, 96.8,
               106.0, 105.0, 103.0, 109.0, 112.0, 82.6],
               dtype=float)

xcf = asarray([254, 267, 312, 324, 274, 284, 337, 337, 255, 300,
               327, 318, 314, 312, 239, 328, 285, 253, 347, 284,
               306, 270, 331, 262, 322, 327, 253, 239, 257, 295],
               dtype=float)

xtu = asarray([1305, 1376, 1438, 1280, 1475, 1135, 1398, 1231,
               1335, 1202, 1113, 1344, 1354, 1472, 1177, 1391,
               1212, 1142, 1277, 1226, 1431, 1402, 1127, 1228,
               1120, 1440, 1241, 1276, 1447, 1265],
               dtype=float)

cmh2 = asarray([99.024, 103.341, 100.302, 98.463, 92.265,
                103.488, 113.735, 108.173, 108.427, 116.26,
                121.05, 111.223, 104.575, 103.223, 99.392,
                87.342, 102.731, 96.369, 99.595, 97.071],
                dtype=float)

mp1_inc = asarray([125.9, 136.6, 1444.5])
mp1_cor = asarray([125.9, 136.6, 144.45])

mp1_all = (asarray([136.64, 125.91, 144.45]),
           asarray([107.79, 114.58, 110.70]),
           asarray([125.50, 118.79, 131.24]),
           asarray([125.91, 127.86, 125.91]),
           asarray([134.41, 124.60, 127.54]),
           asarray([139.35, 119.03, 125.81]),
           asarray([120.00, 121.94, 132.58]),
           asarray([119.28, 118.30, 126.12]),
           asarray([109.50, 121.23, 130.03]),
           asarray([118.71, 126.56, 124.60]))

rcc = (asarray([37.33, 41.99, 41.67, 39.94, 42.07, 41.28, 42.43, 39.12]),
       asarray([35.32, 42.56, 36.59, 38.27, 41.73, 39.96, 34.6, 36.31]),
       asarray([31.4, 34.11, 32.01, 34.08]))

agt = (asarray([7.57175043327556, 8.02464454976303, 7.09530374838432, 8.20991629104958,
                7.65175925227142, 7.26404325725209, 7.94167921379379]),
       asarray([7.14563145292614, 7.66803586686844, 7.91177395680436, 7.51831152366928,
                7.22009369511806, 7.55744177396152, 7.73382839180175]),
       asarray([9.25280318292245, 8.8149405454068, 8.50316031746587, 9.28872045342467,
                9.37224686521014, 9.4475868355739, 8.82602104512217]))

x = concatenate(agt)

# x = concatenate(mp1_all)

# x = cmh2

xnum = x.size
xmin = x.min()
xmax = x.max()

#%%
# k-sample Anderson Darling Test
res = anderson_ksamp(agt, midrank=True)
print(res)

#%%
# MNR Test - MIL HBK 17 Problem 1
mp1_mnr_crit = mnr_crit(mp1_inc.size)
print(f'mp1_mnr_crit = {mp1_mnr_crit}')

print(f'mp1_inc = {mp1_inc}')
ad = anderson(mp1_inc, dist='norm')
mean = ad.fit_result.params.loc
stdev = ad.fit_result.params.scale
print(f'mean = {mean}')
print(f'stdev = {stdev}')
mp1_inc_res = absolute(divide(mp1_inc - mean, stdev))
print(f'mp1_inc_res = {mp1_inc_res}')

print(f'mp1_cor = {mp1_cor}')
ad = anderson(mp1_cor, dist='norm')
mean = ad.fit_result.params.loc
stdev = ad.fit_result.params.scale
print(f'mean = {mean}')
print(f'stdev = {stdev}')
mp1_cor_res = absolute(divide(mp1_cor - mean, stdev))
print(f'mp1_cor_res = {mp1_cor_res}')

#%%
# Sample Fit
params = weibull_min.fit(x, floc=0.0)
pdf_fitted = weibull_min(*params)
c, loc, scale = params
print(f'c = {c:g}')
print(f'loc = {loc:g}')
print(f'scale = {scale:g}')
mean, var, skew, kurt = pdf_fitted.stats(moments='mvsk')
stdev = sqrt(var)
cvar = stdev/mean
print(f'mean = {mean:g}')
print(f'var = {var:g}')
print(f'stdev = {stdev:g}')
print(f'cvar = {cvar:g}')
print(f'skew = {skew:g}')
print(f'kurt = {kurt:g}')
print(f'xnum = {xnum:d}')
print(f'xmin = {xmin:g}')
print(f'xmax = {xmax:g}')

b10 = weibull_min.ppf(0.10, c, scale=scale, loc=0.0)
print(f'b10 = {b10}')

b5 = weibull_min.ppf(0.05, c, scale=scale, loc=0.0)
print(f'b5 = {b5}')

b1 = weibull_min.ppf(0.01, c, scale=scale, loc=0.0)
print(f'b1 = {b1}')

#%%
# Goodness of Fit
gof = goodness_of_fit(weibull_min, x,
                      known_params={"c": c, "loc": loc, "scale": scale},
                      statistic="ad")
print(f'gof = {gof}\n')
print(f'gof.statistic = {gof.statistic}')

#%%
# Anderson Darling Test
ads = (1 + 0.2/sqrt(xnum))*gof.statistic # CMH-17 8.3.6.5.2.2(d)
print(f'ads = {ads}\n')

osl = 1/(1 + exp(-0.1 + 1.24*log(ads) + 4.48*ads)) # CMH-17 8.3.6.5.2.2(c)
print(f'osl = {osl}\n')

#%%
# Distribution Plot
domain = linspace(pdf_fitted.ppf(0.001), pdf_fitted.ppf(0.999), 100)

fig = figure(figsize=(10, 8))
ax = fig.gca()
_ = ax.plot(domain, pdf_fitted.pdf(domain), 'r-', lw=5, alpha=0.7, label='weibull pdf')
_ = ax.hist(x, density=True, bins='auto', histtype='stepfilled', alpha=0.3)

#%%
# Probability Plot
quantiles, fit_metrics = probplot(x, dist=weibull_min, sparams=params)
fig = figure(figsize=(10, 8))
ax = fig.gca()
ax.plot(quantiles[0], quantiles[0], label=f'Best Fit Rsq {fit_metrics[2]:.3f}', color='b')
ax.scatter(quantiles[0], quantiles[1], color='y', label='Data', edgecolor='r')
ax.set_title('Weibull Probability Plot')
ax.set_xlabel('Theoretical Quantiles')
ax.set_ylabel('Ordered Values')
_ = ax.legend()

#%%
# Basis Values for Weibull
def k_factor_weibull(n: int, p: float, conf: float=0.95, r: int=1) -> float:
    lamda = log(-log(p))
    nct_pdf = nct(df=n-r, nc=-sqrt(n)*lamda)
    t = nct_pdf.ppf(conf)
    return t / sqrt(n-1)

def V_factor_weibull(n: int, p: float, conf: float=0.95, r: int=1) -> float:
    lamda = log(-log(p))
    nct_pdf = nct(df=n-r, nc=-sqrt(n)*lamda)
    t = nct_pdf.ppf(conf)
    return t

ka = k_factor_weibull(xnum, 0.99)
kb = k_factor_weibull(xnum, 0.90)

lwa = scale*exp(-ka/c)
lwb = scale*exp(-kb/c)

print(f'ka = {ka}')
print(f'kb = {kb}')
print(f'lwa = {lwa}')
print(f'lwb = {lwb}')
print()

Va = V_factor_weibull(xnum, 0.99)
Vb = V_factor_weibull(xnum, 0.90)

lwa = scale*exp(-Va/c/sqrt(xnum))
lwb = scale*exp(-Vb/c/sqrt(xnum))

print(f'Va = {Va}')
print(f'Vb = {Vb}')
print(f'lwa = {lwa}')
print(f'lwb = {lwb}')
print()

vB_chk = 3.803 + exp(1.79 - 0.516*log(xnum) + 5.1/(xnum-1))
vA_chk = 6.649 + exp(2.55 - 0.526*log(xnum) + 4.76/xnum)

scale_a = scale*(0.01005)**(1/c)
scale_b = scale*(0.10536)**(1/c)

lwa = scale_a*exp(-vA_chk/c/sqrt(xnum))
lwb = scale_b*exp(-vB_chk/c/sqrt(xnum))

print(f'vA_chk = {vA_chk}')
print(f'vB_chk = {vB_chk}')
print(f'lwa = {lwa}')
print(f'lwb = {lwb}')
print()
