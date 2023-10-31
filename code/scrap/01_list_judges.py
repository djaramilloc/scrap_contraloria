"""
The script generates a list of all judges and prosecutors in the sample
"""

import pandas as pd
from pyprojroot import here
import numpy as np

# Load base data
fiscales = pd.read_parquet(r"C:\Users\DanielJaramillo\Documents\3Research\juicios\data\proc\transparencia\datos_fiscales.parquet", columns=['nombre', 'cedula'])
jueces = pd.read_parquet(r"C:\Users\DanielJaramillo\Documents\3Research\juicios\data\proc\transparencia\datos_jueces.parquet", columns=['nombre', 'cedula'])

# Concatenate data
officials = pd.concat([fiscales, jueces])
officials = (officials
             .drop_duplicates(subset='cedula')
             .sort_values('cedula', ignore_index=True)
             )

# Store file
officials.to_parquet(here()/"data/raw/list_officials.parquet", index=False)


# Store 7 different files
for idx, df in enumerate(np.array_split(officials, 7)):
  df.to_parquet(here()/f"data/temp/officials{idx+1}.parquet", index=False)

