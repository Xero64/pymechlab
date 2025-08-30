#%%
# Import Dependencies
from math import exp, log

from IPython.display import display_markdown
from py2md.classes import MDTable

from pymechlab.tools.stats import V_factor_weibull

#%%
# Display V Factor Table
table = MDTable()
table.add_column('n', 'd')
table.add_column('vB', '.3f')
table.add_column('vB check', '.3f')
table.add_column('vA', '.3f')
table.add_column('vA check', '.3f')

for n in range(10, 45):
    vB = V_factor_weibull(n, 0.90)
    vA = V_factor_weibull(n, 0.99)
    vB_chk = 3.803 + exp(1.79 - 0.516*log(n) + 5.1/(n-1))
    vA_chk = 6.649 + exp(2.55 - 0.526*log(n) + 4.76/n)
    table.add_row([n, vB, vB_chk, vA, vA_chk])

display_markdown(table)
