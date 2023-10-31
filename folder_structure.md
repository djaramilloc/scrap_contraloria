## Scrap Contraloria

This project downloads the records of `declaraciones juradas` from the webpage of contraloria

### Code Description

#### Scrap Data Contraloria
- Obtain a list of ids from judges and prosecutors to extract
  - Script: `scrap/01_list_judges.py`
  - Outcome: `data/raw/list_officials.parquet`
- Scrap cases
  - Script: `scrap/02_scrap_contraloria.py`
  - Output: `data/wscrap/registros_raw.parquet`
- Clean downloaded cases
  - Script: `scrap/03_clean_registros.py`
  - Output:
    - Registros con juez y fiscal: `data/wscrap/registros_juez_fiscal.parquet`
    - Todos los registros: `data/wscrap/registros_all.parquet`

#### Extract Information from text
- Get raw fields from images
  - Script `txt_extraction/01_process_imgs.py`
  - Output: `data/txt_extraction/datos_imgs.parquet`
- Clean fecha inicio
  - Script `txt_extraction/02_clean_fechas.py`
  - Output: `data/txt_extraction/docs_con_fechas.parquet`