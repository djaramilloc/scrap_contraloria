import pandas as pd
import warnings

from pyprojroot import here

from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.common.exceptions import UnexpectedAlertPresentException
from webdriver_manager.firefox import GeckoDriverManager


from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Load webscrapper
import sys
sys.path.append((here()/'code').as_posix())
from scrap_funcs import registros_contraloria

# Load Data ===========================================================================

# Define cpus: Each entry is of the form `cpu: (x0, y0)`
positions = {'1': (0, 0), '2': (0, 200), '3': (200, 0), '4': (200, 200), '5': (0, 0), '6': (0, 100), '7': (400, 0)}
cpu = "1"
x0, y0 = [value for key, value in positions.items() if key == cpu][0]

# Load data
print(f"Begin CPU:{cpu} ---------------------------")
officials = pd.read_parquet(here()/f"data/temp/officials{cpu}.parquet")
if 'estado' in officials.columns:
    faltantes = officials.loc[officials['estado'].isna()]
else:
    faltantes = officials.copy()


# Set-up Model and Driver ============================================================================

# Load models to detect captcha
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-printed')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-printed')

# Start Webdriver
s = (Service(GeckoDriverManager().install()))
options = webdriver.FirefoxOptions()
driver = webdriver.Firefox(service=s, options=options)
driver.set_window_rect(x=x0, y=y0)


def get_data(cedula, antes15, driver, processor, model):
    try:
        return registros_contraloria(cedula, antes15, driver, processor, model)
    except UnexpectedAlertPresentException:
        print("Error de alerta no presente!")
        return {'estado': False}

# Call function =======================================================================================

for row in faltantes.iterrows():
    cedula = row[1]['cedula']
    print(f"Working on {cedula} ---------------------------------------------------------")

    # Look for cedula, antes de 2015
    r = get_data(cedula, True, driver, processor, model)
    while not r['estado']:
        print(f"Try again with {cedula}")
        r = get_data(cedula, True, driver, processor, model)

    before15 = pd.DataFrame(
        r['res'],
        columns=['cedula', 'nombre', 'cargo', 'institucion', 'na1', 'year', 'iddoc', 'na2']
    )

    # Look for cedula after 2015
    r = get_data(cedula, False, driver, processor, model)
    while not r['estado']:
        print(f"Try again with {cedula}")
        r = get_data(cedula, False, driver, processor, model)
    after15 = pd.DataFrame(
        r['res'],
        columns=['cedula', 'nombre', 'cargo', 'institucion', 'na1', 'year', 'iddoc', 'na2']
    )

    # Store results of registros
    try:
        registros = pd.read_parquet(here() / f"data/temp/registro{cpu}.parquet")
    except FileNotFoundError:
        registros = pd.DataFrame()

    registros = pd.concat([registros, pd.concat([before15, after15])], ignore_index=True)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        registros.to_parquet(here() / f"data/temp/registro{cpu}.parquet", index=False)
    print("Saved registros data")

    # Store results of estado
    officials.loc[officials.index == row[0], 'estado'] = 1
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        officials.to_parquet(here()/f"data/temp/officials{cpu}.parquet", index=False)
    print("Saved estado data")


# Close driver
print(f"Done with CPU {cpu}!!!")
driver.close()

