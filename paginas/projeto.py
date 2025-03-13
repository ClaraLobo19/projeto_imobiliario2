import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
import os

print("Diretório atual:", os.getcwd()) 
# Função para carregar o modelo treinado
def load_model():
    # Definir seu modelo e pipelines
    scaler = StandardScaler()
     # Mostra o diretório onde o script está rodando

    # Criar um exemplo de DataFrame para suas variáveis numéricas
    df = pd.read_csv('../arquivos/teste.csv')
    
    df = df.drop_duplicates()
    df = df[df['bairro'] != 'Siqueira']
    df = df.dropna(subset=['preço'])
    df = df[df['condominio'] < 10000]
    df.drop(columns=['Unnamed: 0','Unnamed: 0.1','IDH_x','IDH-Renda_x', 'IDH-Longevidade_x','IDH-Educação_x', 'Regional_x','Regional_y','numero','Regional','preco p/ m²','IDH-Educação','IDH'], inplace=True)

    numericas = [col for col in df.columns if df[col].dtype in ['float64', 'int64'] and col not in ['preço', 'preco p/ m²']]
    
    X = df[numericas]
    y = df['preço']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    best_params = {'colsample_bytree': 1.0, 'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 500, 'subsample': 0.9}
    preprocessor = ColumnTransformer([('num', StandardScaler(), numericas)])
    
    # Definir seu modelo (XGBoost, RandomForest ou outro)
    xgb_pipeline = make_pipeline(
        preprocessor,
        XGBRegressor(**best_params)
    )

    # Treinar o modelo
    xgb_pipeline.fit(X_train, y_train)

    return xgb_pipeline, numericas

# Carregar o modelo
model, numericas = load_model()

# Título da página
st.title("Previsão de Preço de Imóveis")

# Inputs do usuário para as características
st.sidebar.header("Informações do Imóvel")

# Inputs interativos
inputs = {}
for feature in numericas:
    if feature == 'preço_cond_ratio':
        
        inputs[feature] = st.sidebar.number_input(f"Valor de {feature}", min_value=0.0, value=0.0, step=1.0)

    else:
        inputs[feature] = st.sidebar.number_input(f"Valor de {feature}", min_value=0.0, value=0.0, step=1.0)

# Converter inputs em DataFrame
input_data = pd.DataFrame([inputs])

# Previsão com o modelo
if st.sidebar.button("Fazer Previsão"):
    prediction = model.predict(input_data)
    st.write(f"O preço estimado do imóvel é: R$ {prediction[0]:,.2f}")
