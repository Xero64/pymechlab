#%%
# Import Dependencies
from numpy import asarray, mean, sqrt, square, std
from scipy.stats import t

from pymechlab.classes.cmh17statistics import DataSet


#%%
# Implement Grubbs Test Function
def grubbs_test(x):
   n = len(x)
   mean_x = mean(x)
   sd_x = std(x, ddof=1)
   numerator = max(abs(x-mean_x))
   g_calculated = numerator/sd_x
   print("Grubbs Calculated Value:", g_calculated)
   t_value_1 = t.ppf(1 - 0.05 / (2 * n), n - 2)
   g_critical = ((n - 1) * sqrt(square(t_value_1))) / (sqrt(n) * sqrt(n - 2 + square(t_value_1)))
   print("Grubbs Critical Value:", g_critical)
   if g_critical > g_calculated:
      print("We can see from the Grubbs test that the calculated value is less than the crucial value. Recognize the null hypothesis and draw the conclusion that there are no outliers\n")
   else:
      print("We see from the Grubbs test that the estimated value exceeds the critical value. Reject the null theory and draw the conclusion that there are outliers\n")

#%%
# First Example Data and Results
x = asarray([12, 13, 14, 19, 21, 23])
y = asarray([12, 13, 14, 19, 21, 23, 45])

grubbs_test(x)
grubbs_test(y)

#%%
# Second Example Data and Results
mp1_inc = asarray([125.9, 136.6, 1444.5])
mp1_cor = asarray([125.9, 136.6, 144.45])

grubbs_test(mp1_inc)
grubbs_test(mp1_cor)

#%%
# Second Example Data and Results
mp1_inc_ds = DataSet('Incorrect Data', mp1_inc)
mp1_cor_ds = DataSet('Correct Data', mp1_cor)

print(f'mp1_inc_ds.data = {mp1_inc_ds.data}\n')
print(f'mp1_inc_ds.valid = {mp1_inc_ds.valid}\n')

print(f'mp1_cor_ds.data = {mp1_cor_ds.data}\n')
print(f'mp1_cor_ds.valid = {mp1_cor_ds.valid}\n')
