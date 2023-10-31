
import json

from PIL import Image
import io
import time

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoAlertPresentException


# Relevant Functions
def solve_captcha(imagen, processor, model):
    pixel_values = processor(images=imagen, return_tensors='pt').pixel_values
    generated_ids = model.generate(pixel_values, max_new_tokens=6)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0].replace('&', '8')


def get_table_response(driver, comando:str, cedula:str) -> list:
    network_data = driver.execute_script(comando)
    urls = [dct['name'] for dct in network_data if 'WFResultados' in dct['name']]

    response = []
    for url in urls:
        driver.get(url)
        textoel = WebDriverWait(driver, 5).until(EC.visibility_of_element_located((By.TAG_NAME, 'body')))
        values = json.loads(textoel.text)['data']

        if len(values) > 0:
            response += values

    if len(response) > 0:
        return response
    else:
        return [[cedula] + ['No data' for _ in range(7)]]

js_script = """
var networkData = performance.getEntries() || {};
return networkData;
"""


def registros_contraloria(cedula:str, antes15:bool, driver, processor, model) -> dict:
    """
    Return a list with all the entries found for cedula
    """

    # Open start page
    driver.get('https://www.contraloria.gob.ec/Consultas/DeclaracionesJuradas')
    time.sleep(1)

    # Send cedula
    ced_element = WebDriverWait(driver, 5).until(EC.visibility_of_element_located((By.ID, 'txtCedula')))
    ced_element.send_keys(cedula)

    # Try to decode captcha
    image_binary = driver.find_element(By.ID, 'captcha').screenshot_as_png
    img = Image.open(io.BytesIO(image_binary)).convert('RGB')
    driver.find_element(By.ID, 'x').send_keys(solve_captcha(img, processor, model))

    # Look elements before 2015
    if antes15:
        driver.find_element(By.ID, 'rdoHistorico_1').click()

    # Send `buscar` click
    driver.find_element(By.ID, 'btnBuscar_in').click()
    time.sleep(1)

    # Check if I got correctly the code
    try:
        alerta = driver.switch_to.alert.text
        driver.switch_to.alert.accept()
        if 'incorrecto' in alerta:
            print("Wrong captcha!")
            return {'estado': False}
    except NoAlertPresentException:
        print("Correct Captcha")
        pass

    # Check if there is info before 2015
    tabla = driver.find_element(By.ID, 'tblBusquedaResultados').find_element(By.TAG_NAME, 'tbody').text
    if tabla == 'Sin resultados':
        print(f"Sin resultados {'antes 2015' if antes15 else 'after 2015'}")
        return {'estado': True, 'res': [[cedula] + ['No data' for _ in range(7)]]}
    else:
        print(f"Collected results {cedula}")
        return {'res': get_table_response(driver, js_script, cedula), 'estado': True}

