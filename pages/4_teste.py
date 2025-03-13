import streamlit as st
import pandas as pd

# Dividindo a tela em 3 colunas
col1, col2, col3 = st.columns(3)

# Botões nas colunas com chave exclusiva para cada um
if col1.button("Botão 1", key="botao_1"):
    st.write("Você clicou no botão 1")

if col2.button("Botão 2", key="botao_2"):
    st.write("Você clicou no botão 2")

if col3.button("Botão 3", key="botao_3"):
    st.write("Você clicou no botão 3")

# Sidebar
st.sidebar.selectbox("Selecione um item:", ["Item 1", "Item 2", "Item 3"])

# Botões adicionais com chaves exclusivas
with col1:
    if st.button('Botão com chave 1', key="botao_4"):
        st.write('Você clicou no botão com chave 1')

with col2:
    if col2.button("Botão com chave 2", key="botao_5"):
        st.write("Você clicou no botão com chave 2")

with col3:
    if col3.button("Botão com chave 3", key="botao_6"):
        st.write("Você clicou no botão com chave 3")

# Informações do imóvel na sidebar
st.sidebar.header("Informações do Imóvel")

# Suponha que você tenha um dataframe df
df = pd.DataFrame({
    "bairro": ["Bairro A", "Bairro B", "Bairro C"],
    "preço": [200000, 300000, 400000]
})

# Adicionar o menu no topo
menu = ["Home", "Previsão de Preço", "Estatísticas", "Mapa"]
opcao_selecionada = st.selectbox("Escolha uma opção", menu)

# Definindo o conteúdo baseado na seleção do menu
if opcao_selecionada == "Home":
    st.title("Bem-vindo ao Simulador de Preços de Imóveis")
    st.write("Este é o menu principal, onde você pode navegar pelas opções.")
elif opcao_selecionada == "Previsão de Preço":
    st.title("Previsão de Preço de Imóveis")
    st.write("Aqui você pode fazer previsões de preço com base nas informações do imóvel.")
elif opcao_selecionada == "Estatísticas":
    st.title("Estatísticas dos Imóveis")
    st.write("Aqui você pode ver as estatísticas dos imóveis.")
elif opcao_selecionada == "Mapa":
    st.title("Mapa dos Imóveis")
    st.write("Aqui você pode visualizar a localização dos imóveis no mapa.")
