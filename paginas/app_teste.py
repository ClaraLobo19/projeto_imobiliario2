import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
import pydeck as pdk
from streamlit.paginas.model import load_and_train_model

# 🔹 Carregar o modelo treinado e os dados
model, numericas, df = load_and_train_model()

# 📌 Função para selecionar o bairro e filtrar os dados
def selecionar_bairro(df):
    bairro_selecionado = st.sidebar.selectbox("🏙️ Selecione um bairro:", df["bairro"].sort_values().unique())
    df_filtrado = df[df["bairro"] == bairro_selecionado]

    if df_filtrado.empty:
        st.warning("Nenhum imóvel encontrado para este bairro.")
        return None, None, None, None, None

    return (
        df_filtrado["latitude"].mean(),
        df_filtrado["longitude"].mean(),
        df_filtrado["IDH-Longevidade"].mean(),
        df_filtrado["IDH-Renda"].mean(),
        df_filtrado
    )

# 🔹 Função para selecionar dois bairros e filtrar os dados para comparação
def selecionar_bairros_comparar(df):
    bairros = df["bairro"].sort_values().unique()
    bairro1 = st.sidebar.selectbox("🏙️ Selecione o primeiro bairro:", bairros)
    bairro2 = st.sidebar.selectbox("🏙️ Selecione o segundo bairro:", bairros)

    df_bairro1 = df[df["bairro"] == bairro1]
    df_bairro2 = df[df["bairro"] == bairro2]

    return bairro1, bairro2, df_bairro1, df_bairro2

# 🔹 Comparação entre bairros
bairro1, bairro2, df_bairro1, df_bairro2 = selecionar_bairros_comparar(df)

# 📌 Título do App
st.title("🏡 Previsão de Preço de Imóveis")

# 📌 Sidebar para inserção de dados do imóvel
st.sidebar.header("📋 Informações do Imóvel")

inputs = {}
numericas_filtradas = [col for col in numericas if col not in ['quartos_por_m²', 'banheiros_por_quarto', 'latitude', 'longitude', 'IDH-Longevidade', 'IDH-Renda']]
numericas_extra = ['quartos_por_m²', 'banheiros_por_quarto', 'latitude', 'longitude', 'IDH-Longevidade', 'IDH-Renda']

# 🔹 Coletar entradas do usuário
for feature in numericas_filtradas:
    min_value = 0.0 if feature == 'condominio' else 0.1
    inputs[feature] = st.sidebar.number_input(f"{feature.capitalize()}:", min_value=min_value, value=min_value, step=10.0)

# 🔹 Adicionar variáveis extras calculadas automaticamente
inputs.update({
    'latitude': df_bairro1['latitude'].mean(),
    'longitude': df_bairro1['longitude'].mean(),
    'IDH-Longevidade': df_bairro1['IDH-Longevidade'].mean(),
    'IDH-Renda': df_bairro1['IDH-Renda'].mean(),
    'quartos_por_m²': inputs['Quartos'] / inputs['area m²'] if 'Quartos' in inputs and 'area m²' in inputs else 0,
    'banheiros_por_quarto': inputs['banheiros'] / inputs['Quartos'] if 'banheiros' in inputs and 'Quartos' in inputs else 0
})

# Criar DataFrame com os dados de entrada
input_data = pd.DataFrame([inputs])

# 📌 Exibir os dados inseridos
st.write("### 🔎 Dados de Entrada:", input_data)

# 📌 Estatísticas do bairro
def mostrar_estatisticas(df_filtrado, bairro_nome):
    if df_filtrado is None or df_filtrado.empty:
        return
    
    st.subheader(f"📊 Estatísticas de {bairro_nome}")

    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("💰 Preço Médio", f"R$ {df_filtrado['preço'].mean():,.2f}")
        st.metric("📏 Área Média", f"{df_filtrado['area m²'].mean():,.2f} m²")
    
    with col2:
        st.metric("🛏️ Quartos Médios", f"{df_filtrado['Quartos'].mean():.1f}")
        st.metric("🚿 Banheiros Médios", f"{df_filtrado['banheiros'].mean():.1f}")

# 📌 Mostrar estatísticas dos bairros selecionados
#mostrar_estatisticas(df_bairro1, bairro1)
#mostrar_estatisticas(df_bairro2, bairro2)

# 📌 Mapa interativo dos imóveis
def exibir_mapa(df_filtrado, bairro_nome):
    if df_filtrado is None or df_filtrado.empty:
        return

    st.subheader(f"📍 Mapa dos Imóveis em {bairro_nome}")

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=df_filtrado,
        get_position=["longitude", "latitude"],
        get_color=[255, 0, 0, 160],  # Vermelho semi-transparente
        get_radius=30,
    )

    view_state = pdk.ViewState(
        latitude=df_filtrado["latitude"].mean(),
        longitude=df_filtrado["longitude"].mean(),
        zoom=13,
        pitch=0,
    )

    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state))

exibir_mapa(df_bairro1, bairro1)
exibir_mapa(df_bairro2, bairro2)

# 📌 Gráfico de Importância das Variáveis (SHAP)
def mostrar_importancia(model, input_data):
    if input_data is None or input_data.empty:
        return
    
    st.subheader("📈 Importância das Variáveis no Modelo")
    
    explainer = shap.Explainer(model.named_steps['xgbregressor'])
    shap_values = explainer(input_data)

    fig, ax = plt.subplots(figsize=(8, 4))
    shap.summary_plot(shap_values, input_data, plot_type="bar", show=False)
    st.pyplot(fig)

# 📌 Fazer Previsão e exibir gráficos
if st.sidebar.button("🚀 Fazer Previsão"):
    prediction = model.predict(input_data)
    st.write(f"💰 **O preço estimado do imóvel é:** R$ {prediction[0]:,.2f}")

    mostrar_importancia(model, input_data)

# Modo Claro/Escuro
dark_mode = st.sidebar.toggle("🌙 Modo Escuro")

# # CSS para alternar entre os temas
# def set_theme(dark_mode):
#     if dark_mode:
#         st.markdown(
#             """
#             <style>
#                 body {
#                     background-color: #121212;
#                     color: white;
#                 }
#                 .stApp {
#                     background-color: #121212;
#                 }
#                 .css-1aumxhk {
#                     color: white !important;
#                 }
#             </style>
#             """,
#             unsafe_allow_html=True
#         )
#     else:
#         st.markdown(
#             """
#             <style>
#                 body {
#                     background-color: white;
#                     color: black;
#                 }
#             </style>
#             """,
#             unsafe_allow_html=True
#         )

# set_theme(dark_mode)


############################################################################################################################

def mostrar_importancia(model, input_data):
    st.subheader("📈 Importância das Variáveis no Modelo")
    
    explainer = shap.Explainer(model.named_steps['xgbregressor'])
    shap_values = explainer(input_data)

    fig, ax = plt.subplots(figsize=(8, 4))
    shap.summary_plot(shap_values, input_data, show=False)
    st.pyplot(fig)
if st.sidebar.button("Fazer Previsão", key="previsao_button"):
    prediction = model.predict(input_data)
    st.write(f"O preço estimado do imóvel é: R$ {prediction[0]:,.2f}")




bins = [0, 100000, 250000, 500000, 1000000, float('inf')]
labels = ['0-100k', '100k-250k', '250k-500k', '500k-1M', 'Acima de 1M']

df['preco_bin'] = pd.cut(df['preço'], bins=bins, labels=labels)

# Exemplo de visualização do DataFrame com a nova coluna
print(df[['preço', 'preco_bin']])

# Preparar os dados para o mapa
df_filtrado = df.dropna(subset=['longitude', 'latitude'])  # Garantir que não haja valores nulos

# Gerar o mapa de calor
heatmap_layer = pdk.Layer(
    "HeatmapLayer",  # Tipo de camada: mapa de calor
    data=df_filtrado,  # Dados filtrados
    get_position=["longitude", "latitude"],  # Coordenadas dos imóveis
    get_weight="preco_bin",  # O peso será determinado pelos bins de preço
    opacity=0.7,  # Opacidade do mapa
    threshold=0.1  # Limite de intensidade para o mapa de calor
)

# Definir o estado de visualização do mapa
view_state = pdk.ViewState(
    latitude=df_filtrado["latitude"].mean(),
    longitude=df_filtrado["longitude"].mean(),
    zoom=12,
    pitch=45
)

# Criar o mapa e exibi-lo no Streamlit
st.pydeck_chart(pdk.Deck(layers=[heatmap_layer], initial_view_state=view_state))




# Dividir os preços em bins
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
df_filtrado = df.dropna(subset=['longitude', 'latitude'])  # Garantir que não haja valores nulos

# Gerar o mapa de calor
heatmap_layer = pdk.Layer(
    "HeatmapLayer",  # Tipo de camada: mapa de calor
    data=df_filtrado,  # Dados filtrados
    get_position=["longitude", "latitude"],  # Coordenadas dos imóveis
    get_weight="preco_bin_numeric",  # O peso será determinado pelos valores numéricos dos bins de preço
    opacity=0.7,  # Opacidade do mapa
    threshold=0.1  # Limite de intensidade para o mapa de calor
)

# Definir o estado de visualização do mapa
view_state = pdk.ViewState(
    latitude=df_filtrado["latitude"].mean(),
    longitude=df_filtrado["longitude"].mean(),
    zoom=12,
    pitch=0
)

# Criar o mapa e exibi-lo no Streamlit
st.pydeck_chart(pdk.Deck(layers=[heatmap_layer], initial_view_state=view_state))    

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar os dados (substitua pela forma como você carrega o DataFrame no seu app)
# df = pd.read_csv("seu_arquivo.csv")

st.title("📊 Análises Estatísticas")

# --- HISTOGRAMA ---
st.subheader("Distribuição de Preços dos Imóveis")
fig, ax = plt.subplots(figsize=(10, 5))
sns.histplot(df['preço'], bins=30, kde=True, color='blue', ax=ax)
ax.set_xlabel("Preço (R$)")
ax.set_ylabel("Frequência")
st.pyplot(fig)

# --- BOXPLOT ---
st.subheader("Boxplot de Preços por Bairro")
fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(x='bairro', y='preço', data=df, palette="coolwarm")
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
st.pyplot(fig)
