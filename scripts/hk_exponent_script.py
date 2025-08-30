#%%
# Import Dependencies
from IPython.display import display_markdown
from py2md.classes import MDTable

from pymechlab.classes.cmh17statistics import NonParametricStatistics

#%%
# Display HK Exponent Table
table = MDTable()
table.add_column('n', 'd')
table.add_column('rB', 'd')
table.add_column('kB', '.3f')
table.add_column('kA', '.5f')

for n in range(2, 30):
    npb = NonParametricStatistics()
    npb.set_data_size(n)
    rB = npb.brank
    kB = npb.bfactor
    kA = npb.afactor
    table.add_row([n, rB, kB, kA])

display_markdown(table)
