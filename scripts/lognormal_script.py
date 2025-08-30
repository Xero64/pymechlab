#%%
# Import Dependencies
from matplotlib.pyplot import figure
from numpy import asarray, concatenate, exp, linspace, log, sqrt
from scipy.stats import anderson, f_oneway, goodness_of_fit, probplot
from scipy.stats.distributions import lognorm

from pymechlab.tools.stats import a_value_lognormal, b_value_lognormal

#%%
# Sample Input
xcu = asarray([406, 393, 549, 429, 373, 537, 426, 600, 397, 428,
               405, 504, 480, 503, 520, 411, 417, 444, 483, 466,
               411, 441, 441, 556, 537, 495, 461, 431, 415, 422],
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

x = concatenate(rcc)

# x = cmh2

xnum = x.size
xmin = x.min()
xmax = x.max()

#%%
# Anderson Darling Test
ad = anderson(log(x), dist='norm')
print(f'ad = {ad}\n')

ads = (1 + 4/xnum - 25/xnum**2)*ad.statistic # CMH-17 8.3.6.5.1.2(d)
print(f'ads = {ads}\n')

osl = 1/(1 + exp(-0.48 + 0.78*log(ads) + 4.58*ads)) # CMH-17 8.3.6.5.1.2(c)
print(f'osl = {osl}\n')

logmean = ad.fit_result.params.loc
logstdev = ad.fit_result.params.scale
mean = exp(logmean)
stdev = exp(logstdev)

# #%%
# # Goodness of Fit
# params = lognorm.fit(x, fs=logstdev, floc=0, fscale=logmean)
# s, loc, scale = params
pdf_fitted = lognorm(logstdev, loc=0.0, scale=mean)
print(f's = {logstdev:g}')
print(f'loc = {0.0:g}')
print(f'scale = {logmean:g}')

# gof = goodness_of_fit(lognorm, x,
#                       known_params={"s": s, "loc": loc, "scale": scale},
#                       statistic="ad")

# print(f'gof = {gof}\n')
# print(f'gof.statistic = {gof.statistic}')

#%%
# Sample Fit
# mean, var, skew, kurt = pdf_fitted.stats(moments='mvsk')
# stdev = sqrt(var)
# cvar = stdev/mean
# print(f'logmean = {logmean:g}')
# print(f'logstdev = {logstdev:g}')
# print(f'mean = {mean:g}')
# print(f'var = {var:g}')
# print(f'stdev = {stdev:g}')
# print(f'cvar = {cvar:g}')
# print(f'skew = {skew:g}')
# print(f'kurt = {kurt:g}')
# print(f'xnum = {xnum:d}')
# print(f'xmin = {xmin:g}')
# print(f'xmax = {xmax:g}')

B = b_value_lognormal(mean, stdev, xnum)
A = a_value_lognormal(mean, stdev, xnum)
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
_ = ax.plot(domain, pdf_fitted.pdf(domain), 'r-', lw=5, alpha=0.7, label='lognormal pdf')
_ = ax.hist(x, density=True, bins='auto', histtype='stepfilled', alpha=0.3)

#%%
# Probability Plot
quantiles, fit_metrics = probplot(x, dist=lognorm, sparams=(logstdev, 0.0, mean))
fig = figure(figsize=(10, 8))
ax = fig.gca()
ax.plot(quantiles[0], quantiles[0], label=f'Best Fit Rsq {fit_metrics[2]:.3f}', color='b')
ax.scatter(quantiles[0], quantiles[1], color='y', label='Data', edgecolor='r')
ax.set_title('Lognorm Probability Plot')
ax.set_xlabel('Theoretical Quantiles')
ax.set_ylabel('Ordered Values')
_ = ax.legend()
