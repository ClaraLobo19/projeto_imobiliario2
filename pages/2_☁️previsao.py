import streamlit as st
import pandas as pd
#import pickle  # ou joblib, se preferir
#from modules.model import load_and_train_model
import pydeck as pdk
import sys
import os
from sklearn.cluster import KMeans
from modules.model import data_frame
from haversine import haversine
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
from endpoint import app  

st.set_page_config(layout="wide")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "modules")))

#--------------------------------------------------------------------------------------------------------------------------
modelo_treinado_path = 'models/modelo3.pkl'
kmeans_path = 'models/kmeans3.pkl'

model = joblib.load(modelo_treinado_path)
kmeans_model = joblib.load(kmeans_path)

df = data_frame()
#st.write(df)
numericas = [
    "aream2", "Quartos", "banheiros", "vagas", "condominio", 
    "latitude", "longitude", "idh_longevidade", "area_renda", 
    "distancia_centro", "cluster_geo"
]
# ------------------------------------------SELECIONAR BAIRROS E RETORNAR VALORE PARA PREDIÇÃO-----------------------------------
def selecionar_bairro(df):
    bairro_selecionado = st.sidebar.selectbox("Selecione um bairro:", df["bairro"].sort_values().unique())
    df_filtrado = df[df["bairro"] == bairro_selecionado]
    #lat, lon = df_filtrado["latitude"].mean() , df_filtrado["longitude"].mean()
    
    # Aplicando K-Means para encontrar um ponto representativo dentro do bairro
    kmeans_bairro= KMeans(n_clusters=1, random_state=42, n_init=10)
    kmeans_bairro.fit(df_filtrado[["latitude", "longitude"]])
    
    # Obter o centro do cluster
    lat, lon = kmeans_bairro.cluster_centers_[0]

    # Cálculo do IDH médio
    idh_longevidade = df_filtrado["idh_longevidade"].mean()
    idh_renda = df_filtrado["idh_renda"].mean()
    idh_longevidade = df_filtrado["idh_educacao"].mean()
    idh_renda = df_filtrado["idh_renda"].mean()
    return lat, lon, idh_longevidade, idh_renda, df_filtrado
#-----------------------------------------------------------------------------------------------------------------------------------



st.sidebar.header("Informações do Imóvel")
#---------------------------------------- SEPARAR AS VARIÁVEIS DE ENTRADA COM OS COLETADOS DE ENTRADAS DO USUÁRIO---------------------------------------------------------
def input_variaveis(numericas):
    inputs = {}
    numericas = [col for col in numericas if col not in [ 'latitude', 'longitude', 'idh_longevidade', 'area_renda', 'distancia_centro', 'cluster_geo','Unnamed: 0']]
    numericas_extra = ['latitude', 'longitude', 'idh_longevidade', 'idh_renda','cluster_geo', 'area_renda','distancia_centro']
    #,'quartos_por_m2','banheiros_por_quarto'
    
    lat, lon, idh_longevidade, idh_renda, df_filtrado = selecionar_bairro(df)
     
    for feature in numericas:
        if (feature == 'condominio') :
            
            inputs[feature] = st.sidebar.number_input(f"Valor do condomínio", min_value = 0.0, step = 50.0)
        
        elif (feature == 'aream2'):
            inputs[feature] = st.sidebar.number_input(f"Tamanho da area m²", min_value = 10, step = 20)
        
        elif (feature == 'Quartos') or (feature == 'banheiros'):
           
            inputs[feature] = st.sidebar.number_input(f"Quantidade de {feature}", min_value = 1, step = 1)
        elif (feature == 'vagas'):
            inputs[feature] = st.sidebar.number_input(f"Número de {feature} na garagem ", min_value = 0, step = 1)
        #else:
        #    # Para outras variáveis, o valor mínimo é 0.1
        #    st.write(f"Valor de {feature} ")
        #    inputs[feature] = st.sidebar.number_input(f"Quantidade de {feature}", min_value = 0.0,  step = 10.0)

    for var in numericas_extra:
        if var == 'latitude':
            inputs[var] = lat
        elif var == 'longitude':
            inputs[var] = lon
        elif var == 'idh_longevidade':
            inputs[var] = idh_longevidade
        elif var == 'idh_renda':
            inputs[var] = idh_renda
        #elif var == 'quartos_por_m2':
        #    inputs[var] = float(inputs['Quartos']) / inputs['aream2']
        #elif var == 'banheiros_por_quarto':
        #    inputs[var] = inputs['banheiros'] / inputs['Quartos']
        elif var == 'cluster_geo':
        #if 'kmeans_model' not in globals():
            #kmeans_model = joblib.load('modelo_kmeans.pkl')
            scaler = StandardScaler()
            coords = df_filtrado[['latitude', 'idh_renda']]
            coords_scaled = scaler.fit_transform(coords)  # Ajusta o scaler aos dados do bairro

            # Aplica a transformação nos dados do usuário
            coords_usuario = scaler.transform([[lat, idh_renda]])
            inputs[var] =  kmeans_model.predict(coords_usuario)
            # st.write( kmeans_model.predict(coords_usuario))
        
        elif var == 'area_renda':
            # st.write(df)
            inputs[var] = inputs['aream2'] * idh_renda  

        elif var == 'distancia_centro':
            centro_fortaleza = (-3.730451, -38.521798)
            inputs[var] = haversine(centro_fortaleza, (lat, lon))
    
    return inputs, df_filtrado, numericas, numericas_extra

inputs, df_filtrado, numericas, numericas_extra = input_variaveis(numericas)

# st.write(f'numericas:', numericas)

# st.write(f'numericas_extra: ', numericas_extra)
# st.write(f'inputs:', inputs)
# st.write(df)
st.title("🏡Previsão de Preço de Imóveis")
st.write(
    '**Este é um simulador de preços de imóveis da cidade de Fortaleza- CE. '
    'Estamos continuamente melhorando este simulador para melhor experiência do usuário**')

#Input usuário
input_data = pd.DataFrame([inputs])
#st.write(input_data)
#st.write(input_data.info())
# st.write(f'Inputs:{inputs}')
if st.sidebar.button("Fazer Previsão"):
    prediction = model.predict(input_data)
    st.write(f"## O preço estimado do imóvel é: R$ {prediction[0]:,.2f}")

#if st.sidebar.button("Simular Investimento"):
#    st.session_state.input_data = input_data
#    st.switch_page('simulador')  


col1, col2 = st.columns(2)

def exibir_mapa_scater(df_filtrado):
    
    if df_filtrado.empty:
        st.warning("Nenhum imóvel encontrado para o bairro selecionado.")
        return

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=df_filtrado,
        get_position=["longitude", "latitude"],
        get_color=[255, 0, 0, 160],  # Vermelho semi-transparente
        get_radius=30,  # Tamanho do ponto
    )

    view_state = pdk.ViewState(
        latitude=df['latitude'].mean(),
        longitude=df['longitude'].mean(),
        zoom=13,  # Nível de zoom inicial
        pitch=15,
    )

    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, map_style="mapbox://styles/mapbox/light-v10"))
    

def mostrar_estatisticas(df_filtrado):
    if df_filtrado.empty:
        return
    
    # st.write(f"## 📊 Estatísticas do Bairro {df_filtrado['bairro'].unique()[0]}")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        #st.metric("🏠 Preço Médio", f"R$ {df_filtrado['preço'].mean():,.2f}")
        st.metric("🏠 Faixa Mediana de Preço", f"R$ {df_filtrado['preco'].median():,.2f}")
        st.metric("📏 Área Média", f"{df_filtrado['aream2'].mean():,.2f} m²")
    
    with col2:
        st.metric("🛏️ Média de Quartos", f"{int(df_filtrado['Quartos'].mean())}")
        st.metric("🚿 Média de Banheiros ", f"{int(df_filtrado['banheiros'].mean())}")
    
    with col3:
        df_filtrado['preço p/m'] = df_filtrado['preco']/ df_filtrado['aream2']
        qntd_amostra = df_filtrado.shape[0]
        st.metric("Média de preço por m²", f"R$ {df_filtrado['preco p/m2'].mean():.2f} ")
        st.metric("Número de Casas disponíveis ", f"{qntd_amostra}")
    
    with col4:
        st.metric("idh_renda", f"{df_filtrado['idh_renda'].mean():.2f}")
        st.metric('idh_longevidade', f"{df_filtrado['idh_longevidade'].mean():.2f}")    

mostrar_estatisticas(df_filtrado)

st.write("## 📍 Mapa de alguns Imóveis no Bairro")

#lat, lon, idh_longevidade, idh_renda, df_filtrado = selecionar_bairro(df_filtrado)
exibir_mapa_scater(df_filtrado)




