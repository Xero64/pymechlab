#%%
# Import Dependencies
import numpy as np
from scipy import stats

#%%
# First Test
rng = np.random.default_rng()
res = stats.anderson_ksamp([rng.normal(size=50),
                            rng.normal(loc=0.5, size=30)])
print(res.statistic, res.pvalue)
#(1.974403288713695, 0.04991293614572478)
print(res.critical_values)
# asarray([0.325, 1.226, 1.961, 2.718, 3.752, 4.592, 6.546])

#%%
# Second Test
res = stats.anderson_ksamp([rng.normal(size=50),
                            rng.normal(size=30),
                            rng.normal(size=20)],
                           method=stats.PermutationMethod())
print(res.statistic, res.pvalue)
# (-0.29103725200789504, 0.25)
print(res.critical_values)
# asarray([ 0.44925884,  1.3052767 ,  1.9434184 ,  2.57696569,  3.41634856,
#   4.07210043, 5.56419101])

#%%
# Third Test
sample = rng.normal(size=50)

res = stats.anderson_ksamp([sample, sample, sample],
                           method=stats.PermutationMethod())
print(res.statistic, res.pvalue)
print(res.critical_values)
