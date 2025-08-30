#%%
# Import Dependencies
from numpy import asarray
from scipy.stats import anderson_ksamp

from pymechlab.classes.cmh17statistics import pool_from_json
from pymechlab.tools.stats import adc_ksample, adk_ksample

#%%
# Problem Data
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

k = len(mp1_all)

adk = adk_ksample(mp1_all, display=True)
print(f'adk = {adk:.2f}\n')
print(f'adk*(k-1) = {adk*(k-1):.2f}\n')
print(f'adk*(k-1) - (k-1) = {adk*(k-1) - (k-1)}\n')

adc_2_5, adc_5_0, adc_10_0, adc_25_0 = adc_ksample(mp1_all, display=True)
print(f'adc_2_5 = {adc_2_5:.2f}\n')
print(f'adc_5_0 = {adc_5_0:.2f}\n')
print(f'adc_10_0 = {adc_10_0:.2f}\n')
print(f'adc_25_0 = {adc_25_0:.2f}\n')

print(2.1907134348483748/1.8173185234230869)

adk2 = anderson_ksamp(mp1_all)
print(f'adk2 = {adk2}\n')

#%%
# Import JSON File
jsonfilepath  = '../files/cmh17_pg_8-87.json'
pool = pool_from_json(jsonfilepath)

for ds in pool.datasets:
    ds.mnr_test()

pool.mnr_test()
