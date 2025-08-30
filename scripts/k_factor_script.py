#%%
# Import Dependencies
from IPython.display import display_markdown
from py2md.classes import MDTable

from pymechlab.tools.stats import k_factor_normal

#%%
# Display K Factor Table
table = MDTable()
table.add_column('n', 'd')
table.add_column('kB', '.3f')
table.add_column('kA', '.3f')

for n in range(2, 61):
    kB = k_factor_normal(n, 0.90)
    kA = k_factor_normal(n, 0.99)
    table.add_row([n, kB, kA])

display_markdown(table)
