
import numpy as np
import re
import pandas as pd
from pyprojroot import here

# Get data on first year of data ============================================

# Clasificacion de jueces o fiscales
registros = pd.read_parquet(here()/"data/wscrap/registros_juez_fiscal.parquet")

# Remove observciones sin cargo
registros = registros.loc[~registros['cargo'].isna()]

# Minimo anio de cada uno
registros.loc[registros['year'] == '', 'year'] = np.nan
registros['year'] = registros['year'].astype(float)
registros['min_year'] = registros.groupby(['cedula', 'cargo'])['year'].transform('min')

# Find all zeros
registros['zero'] = 1*(registros['iddoc'] == '0')
registros['zero'] = registros.groupby(['cedula', 'cargo'])['zero'].transform('min')
registros = registros.loc[~((registros['cargo'] == 'otro') & (registros['zero'] == 1))]
registros = registros.drop_duplicates(['cedula', 'cargo', 'iddoc'], ignore_index=True)

# Match to imagenes ================================================================

# Datos Imagenes
datosimgs = pd.read_parquet(here()/"data/txt_extraction/datos_imgs.parquet")

# Match info
datosimgs = pd.merge(
  datosimgs.drop(columns='cargo'),
  registros.loc[(registros['iddoc'] != 0) | (registros['zero'] == 1),
    ['cedula', 'iddoc', 'year', 'min_year', 'cargo', 'zero']],
  how='outer',
  on=['cedula', 'iddoc'],
  validate='1:m',
  indicator=True
)

# REmove non-fiscales o jueces
datosimgs = (datosimgs
             .loc[(datosimgs['cargo'].isin(['juez', 'fiscal'])) & ((datosimgs['iddoc']!='0') | (datosimgs['zero']!=0))]
             .sort_values(['cedula', 'year'], ignore_index=True)
             .drop(columns=['_merge', 'zero'])
             )


# Define Useful functions ====================================================


def any_mode(var):
  var = var.dropna()
  if not var.empty:
    mode_value = var.mode().iat[0]
    return mode_value
  else:
    return None


def single_mode(var):
  var = var.dropna()
  if not var.empty:
    mode_value = var.mode()
    if len(mode_value) == 1:
      return mode_value[0]
    else:
      return None
  else:
    return None


# Extract date ================================================================

# Tratar en base a patron XXXX-XX-XX -------------------------------------------------------------------
patron = re.compile(r'(\w{4})(~|-|\.|:)(\w{2})(~|-|\.|:)(\w{2})')
fechas = (datosimgs['desde'].str.extract(patron)[[0, 2, 4]])
datosimgs = pd.concat([datosimgs, fechas.rename(columns={0: 'anio_str', 2: 'mes_str', 4: 'dia_str'})], axis=1)

# Manual changes to year strings
for col in ['anio_str', 'mes_str', 'dia_str']:
  datosimgs[col] = datosimgs[col].str.replace(r'o', '0', regex=True)
  datosimgs[col] = datosimgs[col].str.replace(r'r|t', '1', regex=True)
  datosimgs[col] = datosimgs[col].str.replace(r'z', '2', regex=True)

datosimgs['anio_str'] = datosimgs['anio_str'].str.replace(r'(?<!^)o', '0', regex=True)
datosimgs['anio_str'] = datosimgs['anio_str'].str.replace(r'^a', '2', regex=True)
datosimgs['anio_str'] = datosimgs['anio_str'].str.replace(r'^o0', '20', regex=True)
datosimgs['anio_str'] = datosimgs['anio_str'].str.replace(r'^y99', '199', regex=True)

cambiar = {'201e': '2018', '20v4': '2014', 'n913': '2013', 'y000': '1990', 'y904': '1994', 'y00e': '1998',
           'y080': '1989'}
for old, new in cambiar.items():
  datosimgs['anio_str'] = datosimgs['anio_str'].str.replace(old, new)

cambiar = {'1102772280': '2014', '1306750959': '2014', '1710993534': '2013', '1001784824': '2013',
           '0301012449': '2013', '0400698221': '2013', '0800404923': '2015', '1102636493': '2013'}
for cedula, new in cambiar.items():
  datosimgs.loc[(datosimgs['cedula'] == cedula) & (datosimgs['cargo'] == 'juez'), 'anio_str'] = new

datosimgs.loc[datosimgs['mes_str'].isna(), 'anio_str'] = np.nan


# Extraer all digitos juntos ------------------------------------------------------------------------
datosimgs['desde'] = datosimgs['desde'].str.replace(r'(\d{4})(~|-|\.|:|\s)(\d{2})(~|-|\.|:|\s)(\d{2})', r'\1\3\5', regex=True)
datosimgs['desde'] = datosimgs['desde'].str.replace(r'(\d{4})(-|\.|:|\s)(\d{4})', r'\1\3', regex=True)
datosimgs['desde'] = datosimgs['desde'].str.replace(r'(\d{6})(-|\.|:|\s)(\d{2})', r'\1\3', regex=True)

datosimgs['from_nodash'] = datosimgs['desde'].fillna('0').str.extract(r'(\d{8})')

# Complete con los extraidos en la primera parte
datosimgs.loc[datosimgs['from_nodash'].isna(), 'from_nodash'] = (datosimgs[['anio_str', 'mes_str', 'dia_str']]
                                                                 .sum(skipna=False, axis=1)
                                                                 )

# Manual Changes to whole string
datosimgs['from_nodash'] = datosimgs['from_nodash'].str.replace('^79', '19', regex=True)
datosimgs['from_nodash'] = datosimgs['from_nodash'].str.replace('^30', '20', regex=True)

# Keep only year-month
datosimgs['from_nodash'] = datosimgs['from_nodash'].str[:-2]

# Convert to year-mes
datosimgs.loc[~datosimgs['from_nodash'].isna(), 'anio_str'] = datosimgs['from_nodash'].str[:4]
datosimgs.loc[~datosimgs['from_nodash'].isna(), 'mes_str'] = datosimgs['from_nodash'].str[4:6]
datosimgs.drop(columns=['dia_str'], inplace=True)


# Correct dates =========================================================================================

# Move from string to integer year
datosimgs.loc[datosimgs['anio_str'].fillna('a').str.isdigit(), 'anio'] = datosimgs['anio_str']
datosimgs['anio'] = datosimgs['anio'].astype(float)
datosimgs.loc[(datosimgs['anio'] > datosimgs['min_year']) | (datosimgs['anio'] < 1950), 'anio'] = np.nan

# Convert mes to float
datosimgs['mes'] = datosimgs['mes_str'].astype(float)
datosimgs.loc[(datosimgs['mes'] > 12) | (datosimgs['mes'] < 1), 'mes'] = np.nan

# Create fecha string only if values work
datosimgs.loc[~((datosimgs['anio'].isna()) | (datosimgs['mes'].isna())), 'fecha_str'] = datosimgs['from_nodash']


# 1. If it has a single mode, we keep that date ---------------------------------------------
datosimgs['mode_fecha'] = datosimgs.groupby(['cedula', 'cargo'])['fecha_str'].transform(single_mode)
datosimgs['fecha_final'] = datosimgs['mode_fecha']


# 2. Si no hay un solo valido nos quedamos con el min_year y el mes que este -----------------------------------
datosimgs['anio_nan'] = 1*(datosimgs['anio'].isna())
datosimgs['anio_nan'] = datosimgs.groupby(['cedula', 'cargo'])['anio_nan'].transform('min')

# Encontrar el mes para completar con el anio
datosimgs.loc[datosimgs['anio_nan'] == 1, 'mes_mode'] = datosimgs.groupby(['cedula', 'cargo'])['mes'].transform(any_mode)

# Completar fecha final en los que tienen mes
new_fecha = datosimgs['mes_mode'].fillna(0).astype('int').astype('str')
new_fecha = new_fecha.str.pad(width=2, side='left', fillchar='0')
new_fecha = datosimgs['min_year'].astype(int).astype(str) + new_fecha
datosimgs.loc[(datosimgs['anio_nan'] == 1) & (~datosimgs['mes_mode'].isna()), 'fecha_final'] = new_fecha

# Completar los que no tienen mes con mitad de anio (Junio)
new_fecha = datosimgs['min_year'].astype(int).astype(str) + '08'
datosimgs.loc[(datosimgs['anio_nan'] == 1) & (datosimgs['fecha_final'].isna()), 'fecha_final'] = new_fecha
datosimgs.drop(columns=['anio_nan', 'mes_mode'], inplace=True)


# 3. Si no hay single mode, me quedo con los que tengan el mismo anio que el min_year -------------------------------

# Localizar los que tienen mismo anio
datosimgs['same_y0'] = 1*(datosimgs['anio'] == datosimgs['min_year'])
datosimgs['any_same_y0'] = datosimgs.groupby(['cedula', 'cargo'])['same_y0'].transform('max')

# Cmabiar fecha_str y update fecha_final
datosimgs.loc[(datosimgs['same_y0'] == 0) & (datosimgs['any_same_y0'] == 1), 'fecha_str'] = np.nan
datosimgs['mode_fecha'] = datosimgs.groupby(['cedula', 'cargo'])['fecha_str'].transform(single_mode)
datosimgs.loc[datosimgs['fecha_final'].isna(), 'fecha_final'] = datosimgs['mode_fecha']


# 4. Try to pick any mode -------------------------------------------------------------------
datosimgs['mode_fecha'] = datosimgs.groupby(['cedula', 'cargo'])['fecha_str'].transform(any_mode)
datosimgs.loc[datosimgs['fecha_final'].isna(), 'fecha_final'] = datosimgs['mode_fecha']


# 5. There are missing in year, but with moth and viceversa -------------------------------------------------------
datosimgs.loc[datosimgs['fecha_final'].isna(), 'mode_anio'] = datosimgs.groupby(['cedula', 'cargo'])['anio'].transform(single_mode)
datosimgs['mode_anio'] = datosimgs['mode_anio'].fillna(0).astype(int).astype(str)

datosimgs.loc[datosimgs['fecha_final'].isna(), 'mode_mes'] = datosimgs.groupby(['cedula', 'cargo'])['mes'].transform(single_mode)
datosimgs.loc[datosimgs['fecha_final'].isna() & datosimgs['mode_mes'].isna(), 'mode_mes'] = 8
datosimgs['mode_mes'] = datosimgs['mode_mes'].fillna(0).astype(int).astype(str).str.pad(width=2, side='left', fillchar='0')

datosimgs.loc[datosimgs['fecha_final'].isna(), 'fecha_final'] = datosimgs['mode_anio'] + datosimgs['mode_mes']


# FINAL CHANGES ==============================================================================================

start_officials = datosimgs[['cedula', 'cargo', 'fecha_final']].drop_duplicates(ignore_index=True)
(
  start_officials
  .rename(columns={'fecha_final': 'start_date'})
  .to_parquet(here()/"data/date_inicio_officials.parquet", index = False)
)

