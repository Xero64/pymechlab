#%%
# Import Dependencies
from IPython.display import display_markdown
from py2md.classes import MDTable

from pymechlab.tools.stats import mnr_crit

#%%
# Display MNR CV Table
table = MDTable()
table.add_column('n', 'd')
table.add_column('CV', '.3f')

for n in range(3, 41):
    CV = mnr_crit(n)
    table.add_row([n, CV])

display_markdown(table)
