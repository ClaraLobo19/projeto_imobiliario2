import streamlit as st
import streamlit as st
import pandas as pd
import os
from modules.model import load_and_train_model
import pydeck as pdk

# Configuração da página
st.set_page_config(page_title="Simulador de Imóveis", layout="wide")
#sst.sidebar.title("Menu")
# Título principal
st.title("🏡 Bem-vindo ao Simulador de Imóveis")
st.write("#### Escolha uma opção abaixo para explorar os dados:")

# Carregar o modelo treinado
model, numericas, df = load_and_train_model()
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
        opacity=0.8, 
        threshold=0.2  
    )

    # Definir o estado de visualização do mapa
    view_state = pdk.ViewState(
        latitude=df_filtrado["latitude"].mean(),
        longitude=df_filtrado["longitude"].mean(),
        zoom=12,
        pitch=0
    )
    st.pydeck_chart(pdk.Deck(layers=[heatmap_layer], initial_view_state=view_state, map_style="mapbox://styles/mapbox/light-v10"))

st.write("## 📍 Mapa de calor por preço Fortaleza")
st.write('Este mapa representa a distribuição de preços dos imóveis em Fortaleza. As áreas em vermelho são as areas com imóveis mais caros. As áreas amarelas são as áreas com imóveis mais baratos.')
exibir_scater(df)

# class MultiApp:
#     def __init__(self):
#         self.apps = {}

#     def add_app(self, title, func):
#         """Adiciona uma nova página ao app"""
#         self.apps[title] = func

#     def run(self):
#         """Executa a página selecionada no menu lateral"""
#         with st.sidebar:
#             selected = option_menu(
#                 menu_title="Menu",  # Nome do menu na barra lateral
#                 options=list(self.apps.keys()),  # Opções disponíveis
#                 icons=['cloud', 'calculator'],  # Ícones para cada página
#                 menu_icon="cast",
#                 default_index=0,
#                 styles={
#                     "container": {"padding": "5px"},
#                     "nav-link": {"color": "black", "font-weight": "bold"},
#                     "nav-link-selected": {"color": "white", "background-color": "green"},
#                 }
#             )

#         # Chama a função correspondente apenas uma vez
#         if selected in self.apps:
#             self.apps[selected]()

# # Funções das páginas
# def previsao():
#     st.write("### Página de Previsão de Preços")
#     st.write("Aqui você poderá prever os preços dos imóveis.")

# def simulador():
#     st.write("### Página do Simulador de Investimentos")
#     st.write("Aqui você pode simular investimentos em imóveis.")

# # Criando a aplicação
# app = MultiApp()
# app.add_app("Previsão", previsao)
# app.add_app("Simulador", simulador)
# app.run()
