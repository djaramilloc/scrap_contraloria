import pandas as pd
import numpy as np
import base64
import requests
from bs4 import BeautifulSoup
import warnings

import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract'

from PIL import Image
import io
from pyprojroot import here


# RELEVANT FUNCTIONS ======================================================================


def get_imgs(cedula:str, declaracion:str) -> dict:
  """
  Return a list containing the binary information of the images
  """

  # Get data
  url = f"https://www.contraloria.gob.ec/sistema/WFDeclaracionTemporal.aspx?xx=99&id={cedula}&td={declaracion}"
  with warnings.catch_warnings():
    warnings.filterwarnings('ignore')
    r = requests.get(url, verify=False)

  # Read data
  if r.ok:
    soup = BeautifulSoup(r.content, 'html.parser')
    results = {}
    for idx, val in enumerate(soup.find_all('img')):
      imgdata = val['src']
      imgdata = imgdata[imgdata.find(',') + 1:]
      imgdata = base64.b64decode(imgdata)
      results[f"img{idx}"] = np.frombuffer(imgdata, np.uint8)

    return results

  else:
    return {'img0': "No data"}


def txt_box(pos:list, data:pd.DataFrame) -> str:
  vals = data.query(f"x0>={pos[0]} and y0>={pos[1]} and x1<={pos[2]} and y1<={pos[3]}").text
  return ' '.join(vals)


# Define function to get data
def img_to_data(imagen, positions) -> dict:
  """
  Convert image to data based on the positions
  """
  # Read image
  datos = pytesseract.image_to_data(imagen)

  # Convert encoding to pandas
  lines = datos.split('\n')
  data = []
  header = lines[0].split('\t')
  for line in lines[1:]:
    values = line.split('\t')
    data.append(dict(zip(header, values)))
  data = pd.DataFrame(data)

  # Clean data
  data = data.loc[(data['conf'].astype(float) > 0)].reset_index(drop=True)
  data[['left', 'top', 'width', 'height']] = data[['left', 'top', 'width', 'height']].astype(int)
  data['x1'] = data['left'] + data['width']
  data['y1'] = data['top'] + data['height']
  data = data.rename(columns={'left': 'x0', 'top': 'y0'})[['text', 'x0', 'y0', 'x1', 'y1']]

  # Return data if 1st image
  res = {}
  for cargo, pos in positions.items():
    res[cargo] = txt_box(pos, data)

  return res


# LOAD POSITIONS =====================================================================
positions = pd.read_excel(here()/"data/raw/positions.xlsx")
positions = positions.groupby('field').agg(x0=('x0', 'min'), y0=('y0', 'min'), x1=('x1', 'max'), y1=('y1', 'max'))

# Convert ot dictionary
positions['pos'] = positions[['x0', 'y0', 'x1', 'y1']].apply(lambda row: list(row), axis=1)
positions = positions[['pos']].to_dict()['pos']

# LOAD DATA =================================================================

# Define process
cpu = "4"
registros = pd.read_parquet(here()/f"data/temp/registros{cpu}.parquet")
registros = registros.loc[(registros['iddoc'] != '0') & (registros['iddoc'] != 'No data')].reset_index(drop=True)

if 'estado' in registros.columns:
  faltantes = registros.loc[registros['estado'] == 0].reset_index(drop=True).copy()
else:
  faltantes = registros.copy()

# RUN IMAGE EXTRACTION =================================================================

res = []
for row in faltantes.iterrows():
  # Define cedula and doc to look for
  cedula = row[1]['cedula']
  iddoc = row[1]['iddoc']

  # Get First Image
  imagen_bytes = get_imgs(cedula, iddoc)['img0']

  # Check if image exist
  if type(imagen_bytes) == str:
    r = {'cedula': cedula, 'iddoc': iddoc}
    print(f"No doc for cedula: {cedula}, and doc {iddoc} ---- CPU: {cpu}")

  # If exists extract data
  else:
    imagen = Image.open(io.BytesIO(imagen_bytes))
    r = img_to_data(imagen, positions)
    r['cedula'] = cedula
    r['iddoc'] = iddoc
    print(f"Done with cedula: {cedula}, and doc {iddoc} ---- CPU: {cpu}")

  # Append to list
  res.append(r)

  # Update estado
  faltantes.loc[faltantes.index == row[0], 'estado'] = 1


# Save data
pd.DataFrame(res).to_parquet(here()/f"data/temp/datos_contraloria.parquet", index=False)

