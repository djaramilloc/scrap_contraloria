
import re
import numpy as np
from unidecode import unidecode
from rapidfuzz import process, fuzz

import pandas as pd
from pyprojroot import here

# Load Necesary data ==============================================================
registros = pd.read_parquet(here()/"data/wscrap/registros_raw.parquet")
clasificacion = pd.read_parquet(here()/"data/raw/clasif_cargos.parquet")

# Useful functions ================================================================


def limpiar(s):
  return re.sub(r'\s+', ' ', unidecode(s)).upper().lstrip().rstrip()


def get_fuzzy_score(word, choices):
  r = process.extract(word, choices, scorer=fuzz.token_sort_ratio, limit=1)
  if r:
    return r[0][:-1]
  else:
    return ('No match', 0)


# Clean Data ===============================================================

# Remove unnecesary no data
registros['con_data'] = 1*(registros['iddoc'] != 'No data')
registros['any_data'] = registros.groupby('cedula')['con_data'].transform(max)

registros = (registros
             .loc[~((registros['any_data'] == 1) & (registros['con_data'] == 0))]
             .drop_duplicates()
             .drop(columns=['any_data', 'con_data'])
             .sort_values(['cedula', 'year'])
             .reset_index(drop=True)
             )

# Clean cargos
registros[['cargo', 'institucion']] = registros[['cargo', 'institucion']].applymap(limpiar)
registros.to_parquet(here()/"data/registros_all.parquet", index=False)

# Keep only records with documents
registros = registros.loc[registros['iddoc'] != 'No data'].reset_index(drop=True)

# Identify jueces and fiscales ======================================================

# Lista de cargos
clasificacion['cargo'] = clasificacion['cargo'].map(limpiar)
juez = set(clasificacion.query("cargo_dummy=='juez'")['cargo'])
fiscal = set(clasificacion.query("cargo_dummy=='fiscal'")['cargo'])
fiscal = fiscal.union({'FISCAL DISTRITO', 'FISCAL DEL JUZGADO II SUBTENIENTE'})

# Get scores
fjuez = lambda x: get_fuzzy_score(x, juez)
ffiscal = lambda x: get_fuzzy_score(x, fiscal)
registros['match_juez'] = registros['cargo'].map(fjuez)
registros['match_fiscal'] = registros['cargo'].map(ffiscal)

# Split into columns
registros[['match_juez', 'score_juez']] = registros['match_juez'].apply(pd.Series)
registros[['match_fiscal', 'score_fiscal']] = registros['match_fiscal'].apply(pd.Series)

# Keep record with highest value
registros['match'] = np.where(
  registros['score_juez'] > registros['score_fiscal'],
  'juez',
  np.where(
    registros['score_juez'] < registros['score_fiscal'],
    'fiscal',
    'otro'
  )
)

registros['score'] = np.where(
  registros['score_juez'] > registros['score_fiscal'],
  registros['score_juez'],
  np.where(
    registros['score_juez'] < registros['score_fiscal'],
    registros['score_fiscal'],
    np.nan
  )
)

# Save temp data
#registros.to_parquet(here()/"data/temp/reg_categorias.parquet", index=False)
registros = pd.read_parquet(here()/"data/temp/reg_categorias.parquet")

# Cahnge to 100 if starts with juez -----------------------------------------------
strpatron = r"^magi|^jue|^conjue|^con jue|^ex jue|^juaza|^jez\b|^juz\b|^juuz\b|^jeuza\b|^jeueza\b|^jeuz\b|^juaz\b"
patron = re.compile(strpatron, flags=re.IGNORECASE)
registros.loc[registros['cargo'].str.contains(patron, regex=True), 'match'] = 'juez'
registros.loc[registros['cargo'].str.contains(patron, regex=True), 'score'] = 100

# Juez en cualquier palabra, menos secretario
patron = re.compile('juez', flags=re.IGNORECASE)
registros.loc[registros['cargo'].str.contains(patron) & (registros['score']<100), 'match'] = 'juez'
registros.loc[registros['cargo'].str.contains(patron) & (registros['score']<100), 'score'] = 100

patron = re.compile(r'^secre', flags=re.IGNORECASE)
registros.loc[registros['cargo'].str.contains(patron), 'match'] = 'otro'
registros.loc[registros['cargo'].str.contains(patron), 'score'] = np.nan

# Algunos tienen juez en institucion
patron = re.compile(r'^jue|^conjue|^con jue', flags=re.IGNORECASE)
registros.loc[registros['institucion'].str.contains(patron, regex=True), 'match'] = 'juez'
registros.loc[registros['institucion'].str.contains(patron, regex=True), 'score'] = 100

# Presidente de tribunales
indices = (
    registros['cargo'].str.contains(r'corte|tribunal|tribyunal|sala', regex=True, flags=re.IGNORECASE) &
    registros['cargo'].str.contains(r'^presi|^presdiente|\bminis', regex=True, flags=re.IGNORECASE)
)
registros.loc[indices, 'match'] = 'juez'
registros.loc[indices, 'score'] = 100


# 100 for fiscales ----------------------------------------------------------------
patron = re.compile(r'^fiscal|^ag|^ganete|agente fiscal|^egente|^ex- fiscal|^ex-ministro fiscal',
                    flags=re.IGNORECASE)
registros.loc[registros['cargo'].str.contains(patron, regex=True), 'match'] = 'fiscal'
registros.loc[registros['cargo'].str.contains(patron, regex=True), 'score'] = 100

# Define fiscalia
registros.loc[registros['institucion'].str.contains('MINISTERIO')
              & registros['institucion'].str.contains('FISCAL'), 'fiscalia'] = 1
registros.loc[registros['institucion'].str.contains('MINISTERIO')
              & registros['institucion'].str.contains('PUBLICO'), 'fiscalia'] = 1

patron = re.compile(r'^fi', flags=re.IGNORECASE)
registros.loc[registros['institucion'].str.contains(patron, regex=True), 'fiscalia'] = 1

patron = re.compile(r'f(.|)g(.|)e', flags=re.IGNORECASE)
registros.loc[registros['institucion'].str.contains(patron, regex=True), 'fiscalia'] = 1

patron = re.compile(r'judicial|tribunal', flags=re.IGNORECASE)
registros.loc[registros['institucion'].str.contains(patron, regex=True), 'fiscalia'] = 0

# Add ministro fiscal to fiscales
patron = re.compile(r'^minist', flags=re.IGNORECASE)
registros.loc[registros['cargo'].str.contains(patron, regex=True) & (registros['fiscalia'] == 1), 'match'] = 'fiscal'
registros.loc[registros['cargo'].str.contains(patron, regex=True) & (registros['fiscalia'] == 1), 'score'] = 100


# Move to others the ones we are not sure -------------------------------------------

# Literal values
registros.loc[registros['cargo'] == 'AGOGADO', 'match'] = 'otro'
registros.loc[registros['cargo'] == 'AGOGADO', 'score'] = np.nan

# Change to missing values with no cargo
registros.loc[registros['cargo'] == '', 'match'] = np.nan

# Puntaje general menor a 55
registros.loc[registros['score'] < 55, 'match'] = 'otro'
registros.loc[registros['score'] < 55, 'score'] = np.nan

# Botar dentro de fiscalia
registros.loc[(registros['score'] < 90) & (registros['fiscalia'] == 1), 'match'] = 'otro'
registros.loc[(registros['score'] < 90) & (registros['fiscalia'] == 1), 'score'] = np.nan

# Botar puestos clasificados como fiscales que no son
registros.loc[(registros['score'] < 90) & (registros['match'] == 'fiscal'), 'match'] = 'otro'
registros.loc[(registros['score'] < 90) & (registros['match'] == 'fiscal'), 'score'] = np.nan

# Botar puestos clasificados como jueces que no son
registros.loc[(registros['score'] < 95) & (registros['match'] == 'juez'), 'match'] = 'otro'
registros.loc[(registros['score'] < 95) & (registros['match'] == 'juez'), 'score'] = np.nan


# Save data =======================================================
(
  registros[['cedula', 'iddoc', 'year', 'match']]
  .rename(columns={'match': 'cargo'})
  .to_parquet(here()/"data/wscrap/registros_juez_fiscal.parquet", index=False)
)


"""


# Identify jueces and fiscales ======================================================





# Limpiar nombres
registros[['cargo', 'institucion']] = registros[['cargo', 'institucion']].applymap(limpiar)

# Casos seguros de fiscales o jueces -----------------------------------------------
# Los que empiezan con fiscal o agente fiscal
patron = re.compile(r'^(fiscal|agente)', flags=re.IGNORECASE)
registros['fiscal'] = 1*(registros['cargo'].str.contains(patron, regex=True))

# Los que empiezan con juez
patron = re.compile(r'^(juez|magis|conjuez)', flags=re.IGNORECASE)
registros['juez'] = 1*(registros['cargo'].str.contains(patron, regex=True))

# Fiscales ------------------------------------------------------------------------
# Los que tienen la palabra fiscal
patron = re.compile(r'\bfiscal\b', flags=re.IGNORECASE)
registros['posiblefiscal'] = 1*(registros['cargo'].str.contains(patron, regex=True))

# Sacar los que no son
patron = re.compile(r'^(aist|asias|asiss|asist|asit|aux|secret|direct|servid|fedata)', flags=re.IGNORECASE)
registros.loc[registros['cargo'].str.contains(patron, regex=True), 'posiblefiscal'] = 0

# Sacar de posibles fiscales a los jueces
registros.loc[(registros['juez'] == 1) & (registros['posiblefiscal'] == 1), 'posiblefiscal'] = 0

# Input the rest of fiscales
registros.loc[(registros['posiblefiscal']==1) & (registros['fiscal']==0), 'fiscal'] = 1
registros = registros.drop(columns='posiblefiscal')

# Jueces ------------------------------------------------------------------------
# Los que tienen la palabra fiscal
patron = re.compile('juez', flags=re.IGNORECASE)
registros['posiblejuez'] = 1*(registros['cargo'].str.contains(patron, regex=True))

# Input the rest of fiscales
registros.loc[(registros['posiblejuez'] == 1) & (registros['juez'] == 0), 'juez'] = 1
registros = registros.drop(columns='posiblejuez')


# Save data ----------------------------------------------------------------
registros.to_parquet(here()/"data/registros_juez_fiscal.parquet", index=False)
"""