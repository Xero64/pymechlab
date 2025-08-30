#%%
# Import Dependencies
from IPython.display import display_markdown

from pymechlab.classes.cmh17statistics import pool_from_json

#%%
# Import JSON File
jsonfilepath  = '../files/cmh17_pg_8-87.json'
pool = pool_from_json(jsonfilepath)
display_markdown(pool)

#%%
# Import JSON File
jsonfilepath  = '../files/cmh17_pg_8-92.json'
pool = pool_from_json(jsonfilepath)
display_markdown(pool)
