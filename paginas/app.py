import streamlit as st
import pandas as pd
import pickle  # ou joblib, se preferir
from streamlit.paginas.model import load_and_train_model
import matplotlib.pyplot as plt
import pydeck as pdk
import streamlit as st
import shap



# Carregar o modelo treinado
model, numericas, df = load_and_train_model()

def selecionar_bairro(df):
    bairro_selecionado = st.sidebar.selectbox("Selecione um bairro:", df["bairro"].sort_values().unique())
    df_filtrado = df[df["bairro"] == bairro_selecionado]
    lat, lon = df_filtrado["latitude"].mean() , df_filtrado["longitude"].mean()
    
    #st.write(f"Imóveis no bairro **{bairro_selecionado}**:", df_filtrado)
    idh_longevidade, idh_renda = df_filtrado["IDH-Longevidade"].mean() , df_filtrado["IDH-Renda"].mean()
    
    return lat, lon, idh_longevidade, idh_renda, df_filtrado



st.sidebar.header("Informações do Imóvel")
# Coletar entradas numéricas do usuário
def input_variaveis(numericas):
    inputs = {}
    numericas = [col for col in numericas if col not in ['quartos_por_m²', 'banheiros_por_quarto', 'latitude', 'longitude', 'IDH-Longevidade', 'IDH-Renda']]
    numericas_extra = ['quartos_por_m²', 'banheiros_por_quarto', 'latitude', 'longitude', 'IDH-Longevidade', 'IDH-Renda']

    lat, lon, idh_longevidade, idh_renda, df_filtrado = selecionar_bairro(df)    
    
    for feature in numericas:
        if feature == 'condominio':
            # Valor mínimo do condomínio é 0
            inputs[feature] = st.sidebar.number_input(f"Valor de {feature}", min_value=0.0, value=0.0, step=10.0)
        else:
            # Para outras variáveis, o valor mínimo é 0.1
            inputs[feature] = st.sidebar.number_input(f"Valor de {feature}", min_value=0.1, value=0.1, step=10.0)

    for var in numericas_extra:
        if var == 'latitude':
            inputs[var] = lat
        elif var == 'longitude':
            inputs[var] = lon
        elif var == 'IDH-Longevidade':
            inputs[var] = idh_longevidade
        elif var == 'IDH-Renda':
            inputs[var] = idh_renda
        elif var == 'quartos_por_m²':
            inputs[var] = inputs['Quartos'] / inputs['area m²']
        elif var == 'banheiros_por_quarto':
            inputs[var] = inputs['banheiros'] / inputs['Quartos']
    
    return inputs, df_filtrado, numericas, numericas_extra

inputs, df_filtrado, numericas, numericas_extra = input_variaveis(numericas)


#Input usuário
input_data = pd.DataFrame([inputs])
st.title("Previsão de Preço de Imóveis")
st.write(
    '**Este é um simulador de preços de imóveis da cidade de Fortaleza- CE. '
    'Estamos continuamente melhorando este simulador para melhor experiência do usuário**')




# Realizar a previsão quando o botão for pressionado
if st.sidebar.button("Fazer Previsão"):
    prediction = model.predict(input_data)
    st.write(f"O preço estimado do imóvel é: R$ {prediction[0]:,.2f}")
    
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
        latitude=df_filtrado["latitude"].mean(),
        longitude=df_filtrado["longitude"].mean(),
        zoom=13,  # Nível de zoom inicial
        pitch=15,
    )

    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state))

def exibir_scater(df):

    bins = [0, 100000, 250000, 500000, 1000000, float('inf')]
    labels = ['0-100k', '100k-250k', '250k-500k', '500k-1M', 'Acima de 1M']

    df['preco_bin'] = pd.cut(df['preço'], bins=bins, labels=labels)

    # Mapear os labels de bins para valores numéricos para usar no mapa
    bin_values = {
        '0-100k': 100000,
        '100k-250k': 250000,
        '250k-500k': 500000,
        '500k-1M': 750000,
        'Acima de 1M': 1500000
    }

    # Substituir os bins por valores numéricos
    df['preco_bin_numeric'] = df['preco_bin'].map(bin_values)

    # Preparar os dados para o mapa
    df_filtrado = df.dropna(subset=['longitude', 'latitude'])  

    # Gerar o mapa de calor
    heatmap_layer = pdk.Layer(
        "HeatmapLayer",  
        data=df_filtrado,  
        get_position=["longitude", "latitude"], 
        get_weight="preco_bin_numeric", 
        opacity=0.7, 
        threshold=0.1  
    )

    # Definir o estado de visualização do mapa
    view_state = pdk.ViewState(
        latitude=df_filtrado["latitude"].mean(),
        longitude=df_filtrado["longitude"].mean(),
        zoom=12,
        pitch=0
    )
    st.pydeck_chart(pdk.Deck(layers=[heatmap_layer], initial_view_state=view_state))    

def mostrar_estatisticas(df_filtrado):
    if df_filtrado.empty:
        return
    
    st.write("## 📊 Estatísticas do Bairro")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("🏠 Preço Médio", f"R$ {df_filtrado['preço'].mean():,.2f}")
        st.metric("📏 Área Média", f"{df_filtrado['area m²'].mean():,.2f} m²")
    
    with col2:
        st.metric("🛏️ Quartos Médios", f"{df_filtrado['Quartos'].mean():.1f}")
        st.metric("🚿 Banheiros Médios", f"{df_filtrado['banheiros'].mean():.1f}")

mostrar_estatisticas(df_filtrado)

st.write("## 📍 Mapa dos Imóveis no Bairro")
col1, col2 = st.columns(2)

with col1:
    exibir_mapa_scater(df_filtrado)
with col2:
    exibir_scater(df)

st.write('### Dados de Entrada:', input_data)

import streamlit as st
import pandas as pd

# Carregar os dados do seu dataset (assumindo que df seja seu dataframe)
# df = pd.read_csv("seu_arquivo.csv")  # Carregue os dados do arquivo

# Função para filtrar os dados do bairro
def comparar_bairros(df, bairro1, bairro2):
    df_bairro1 = df[df['bairro'] == bairro1]
    df_bairro2 = df[df['bairro'] == bairro2]
    
    return df_bairro1, df_bairro2


# Quando o botão for pressionado, comparar os bairros
# if st.sidebar.button("Comparar bairros"):
#     df_bairro1, df_bairro2 = comparar_bairros(df, bairro1, bairro2)
    
#     # Exibir as estatísticas de ambos os bairros lado a lado
#     col1, col2 = st.columns(2)
    
#     with col1:
#         st.subheader(f"📍 {bairro1}")
#         st.write(f"**Preço Médio**: R$ {df_bairro1['preço'].mean():,.2f}")
#         st.write(f"**Área Média**: {df_bairro1['area m²'].mean():,.2f} m²")
#         st.write(f"**Quartos Médios**: {df_bairro1['Quartos'].mean():,.1f}")
#         st.write(f"**Banheiros Médios**: {df_bairro1['banheiros'].mean():,.1f}")
    
#     with col2:
#         st.subheader(f"📍 {bairro2}")
#         st.write(f"**Preço Médio**: R$ {df_bairro2['preço'].mean():,.2f}")
#         st.write(f"**Área Média**: {df_bairro2['area m²'].mean():,.2f} m²")
#         st.write(f"**Quartos Médios**: {df_bairro2['Quartos'].mean():,.1f}")
#         st.write(f"**Banheiros Médios**: {df_bairro2['banheiros'].mean():,.1f}")


    #mostrar_importancia(model, input_data)


st.title("Página Inicial")

if st.sidebar.button("Comparar bairros"):
    st.switch_page('app1.py')

