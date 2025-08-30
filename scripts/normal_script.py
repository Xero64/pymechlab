#%%
# Import Dependencies
from matplotlib.pyplot import figure
from numpy import asarray, concatenate, exp, linspace, log, sqrt
from scipy.stats import anderson, anderson_ksamp, f_oneway, probplot
from scipy.stats.distributions import norm

from pymechlab.tools.stats import a_value_normal, b_value_normal

#%%
# Sample Input
xtf = asarray([340, 490, 412, 375, 454, 354, 408, 353, 491, 417,
               337, 459, 373, 391, 432, 346, 357, 508, 357, 384,
               356, 363, 505, 469, 487, 434, 383, 415, 427, 426],
               dtype=float)

scf = asarray([62.5, 60.2, 60.7, 54.4, 59.4, 68.1, 50.0, 54.9, 57.0, 62.5,
               48.1, 56.6, 48.3, 52.4, 55.1, 49.0, 48.3, 69.6, 54.8, 59.4,
               49.3, 48.9, 68.6, 68.6, 57.6, 52.5, 63.7, 49.8, 52.8, 56.4],
               dtype=float)

sgf = asarray([33.8, 36.5, 35.2, 41.2, 36.7, 36.1, 37.9, 44.8, 42.9, 39.5,
               30.7, 31.5, 44.8, 32.4, 35.0, 42.9, 42.0, 41.2, 33.6, 34.4,
               37.5, 35.2, 35.5, 37.4, 34.6, 34.0, 44.8, 36.5, 44.0, 37.8],
               dtype=float)

cmh1 = asarray([85.39, 97.12, 92.66, 96.43, 90.72, 95.84,
                97.30, 109.47, 101.35, 98.01, 86.18, 100.91,
                96.05, 92.20, 90.86, 101.27, 101.23, 93.15,
                114.32, 100.14, 91.24, 86.11, 93.42, 92.65,
                97.58, 97.75, 97.95, 112.49, 95.75, 110.53])

cmh2 = asarray([99.024, 103.341, 100.302, 98.463, 92.265,
                103.488, 113.735, 108.173, 108.427, 116.26,
                121.05, 111.223, 104.575, 103.223, 99.392,
                87.342, 102.731, 96.369, 99.595, 97.071])

rcc = (asarray([37.33, 41.99, 41.67, 39.94, 42.07, 41.28, 42.43, 39.12]),
       asarray([35.32, 42.56, 36.59, 38.27, 41.73, 39.96, 34.6, 36.31]),
       asarray([31.4, 34.11, 32.01, 34.08]))

agt = (asarray([7.57175043327556, 8.02464454976303, 7.09530374838432, 8.20991629104958,
                7.65175925227142, 7.26404325725209, 7.94167921379379]),
       asarray([7.14563145292614, 7.66803586686844, 7.91177395680436, 7.51831152366928,
                7.22009369511806, 7.55744177396152, 7.73382839180175]),
       asarray([9.25280318292245, 8.81494054540680, 8.50316031746587, 9.28872045342467,
                9.37224686521014, 9.44758683557390, 8.82602104512217]))

x = concatenate(agt)

# x = cmh2

xnum = x.size
xmin = x.min()
xmax = x.max()

#%%
# k-sample Anderson Darling Test
res = anderson_ksamp(agt, midrank=True)
print(res)

#%%
# Anderson Darling Test
ad = anderson(x, dist='norm')
print(f'ad = {ad}\n')

ads = (1 + 4/xnum - 25/xnum**2)*ad.statistic # CMH-17 8.3.6.5.1.2(d)
print(f'ads = {ads}\n')

osl = 1/(1 + exp(-0.48 + 0.78*log(ads) + 4.58*ads)) # CMH-17 8.3.6.5.1.2(c)
print(f'osl = {osl}\n')

mean = ad.fit_result.params.loc
stdev = ad.fit_result.params.scale

#%%
# Sample Fit
# params = norm.fit(x)
# mean, stdev = params
# var = stdev**2*xnum/(xnum-1) # Apply Bessel's Correction
# stdev = sqrt(var)
pdf_fitted = norm(mean, stdev)
var = stdev**2
cvar = stdev/mean
print(f'osl = {osl}')
print(f'mean = {mean:g}')
print(f'var = {var:g}')
print(f'stdev = {stdev:g}')
print(f'cvar = {cvar:g}')
print(f'xnum = {xnum:d}')
print(f'xmin = {xmin:g}')
print(f'xmax = {xmax:g}')

print(pdf_fitted.interval(0.95))

B = b_value_normal(mean, stdev, xnum)
A = a_value_normal(mean, stdev, xnum)
print(f'B = {B:g}')
print(f'A = {A:g}')

#%%
# One Way ANOVA
anova = f_oneway(*rcc)
print(f'anova = {anova}\n')

#%%
# Plots
domain = linspace(pdf_fitted.ppf(0.001), pdf_fitted.ppf(0.999), 100)

fig = figure(figsize=(10, 8))
ax = fig.gca()
_ = ax.plot(domain, pdf_fitted.pdf(domain), 'r-', lw=5, alpha=0.7, label='normal pdf')
_ = ax.hist(x, density=True, bins='auto', histtype='stepfilled', alpha=0.3)

#%%
# Probability Plot
quantiles, fit_metrics = probplot(x, dist=norm, sparams=(mean, stdev))
fig = figure(figsize=(10, 8))
ax = fig.gca()
ax.plot(quantiles[0], quantiles[0], label=f'Best Fit Rsq {fit_metrics[2]:.3f}', color='b')
ax.scatter(quantiles[0], quantiles[1], color='y', label='Data', edgecolor='r')
ax.set_title('Normal Probability Plot')
ax.set_xlabel('Theoretical Quantiles')
ax.set_ylabel('Ordered Values')
_ = ax.legend()
