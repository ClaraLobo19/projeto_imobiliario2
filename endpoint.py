from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
from pydantic import BaseModel, Field
import os
os.chdir(os.path.abspath(os.curdir))
import pickle
import pandas as pd

# Instanciando o app
app = FastAPI(title="API de Previsão de Preço de Imóvel")

# Carregando o modelo treinado
modelo = joblib.load('models/modelo3.pkl')


colunas = [
    'aream2', 'Quartos', 'banheiros', 'vagas', 'condominio', 'latitude',
    'longitude', 'idh_longevidade', 'area_renda', 'distancia_centro', 'cluster_geo'
]

# Definindo o formato dos dados de entrada
class InputData(BaseModel):
    aream2: float 
    Quartos: int
    banheiros: int
    vagas: int
    condominio: float
    latitude: float
    longitude: float
    idh_longevidade: float 
    area_renda: float
    distancia_centro: float
    cluster_geo: int

@app.post("/predict")
def predict(data: InputData):
    try:
        # Convertendo os dados para DataFrame
        df = pd.DataFrame([[
            data.aream2, data.Quartos, data.banheiros, data.vagas,
            data.condominio, data.latitude, data.longitude, data.idh_longevidade,
            data.area_renda, data.distancia_centro, data.cluster_geo
        ]], columns=colunas)
        
        #df1 = pd.DataFrame([[80.0, 2, 1, 1, 500.0, -3.75, -38.5, 0.82, 3500.0, 5.2, 3]], columns=colunas)
        # Fazendo a predição
        prediction = modelo.predict(df)

        return {"predicted_house_value": round(float(prediction[0]), 2)}

    except Exception as e:
        return {"error": str(e)}