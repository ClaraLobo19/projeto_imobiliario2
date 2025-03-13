import streamlit as st
import numpy as np
import pandas as pd


# Coletando os dados de entrada

# Configuração da página
# Recuperar os dados de entrada armazenados na sessão
#input_data = st.session_state.get('input_data', {})


input_data = {
    'valor_imovel': 300000.0,
    'entrada': 60000.0,
    'taxa_juros': 9.0,
    'prazo': 30
}

if not input_data:
    st.error("Nenhum dado encontrado. Retorne à página de previsão para inserir os valores.")
    st.stop()
st.set_page_config(page_title="Simulador de Financiamento Imobiliário", layout="centered")
st.title("🏡 Simulador de Financiamento Imobiliário")



# --- Inputs do Usuário ---
col1, col2 = st.columns(2)

with col1:
    valor_imovel = st.number_input("🏠 Valor do Imóvel (R$)", min_value=10000.0, value=input_data['valor_imovel'], step=1000.0, format="%.2f")
    entrada = st.number_input("💰 Entrada (R$)", min_value=0.0, value=input_data['entrada'], step=1000.0, format="%.2f")

with col2:
    taxa_juros = st.number_input("📈 Taxa de Juros Anual (%)", min_value=0.1, value=input_data['taxa_juros'], step=0.1, format="%.2f")
    prazo = st.slider("📆 Prazo do Financiamento (anos)", min_value=1, max_value=35, value=input_data['prazo'])

# --- Cálculo da Parcela ---
valor_financiado = valor_imovel - entrada
n_meses = prazo * 12
taxa_mensal = (taxa_juros / 100) / 12

if taxa_mensal > 0:
    parcela = valor_financiado * (taxa_mensal * (1 + taxa_mensal) ** n_meses) / ((1 + taxa_mensal) ** n_meses - 1)
else:
    parcela = valor_financiado / n_meses  # Sem juros, apenas divisão

st.write("### 📊 Resultado do Financiamento")
st.write(f"💵 **Valor Financiado:** R$ {valor_financiado:,.2f}")
st.write(f"📅 **Número de Parcelas:** {n_meses} meses")
st.write(f"📌 **Parcela Mensal:** R$ {parcela:,.2f}")

# --- Evolução da Dívida ---
saldo_devedor = valor_financiado
tabela_amortizacao = []
for mes in range(1, n_meses + 1):
    juros = saldo_devedor * taxa_mensal
    amortizacao = parcela - juros
    saldo_devedor -= amortizacao
    tabela_amortizacao.append([mes, parcela, juros, amortizacao, saldo_devedor])

df_amortizacao = pd.DataFrame(tabela_amortizacao, columns=["Mês", "Parcela", "Juros", "Amortização", "Saldo Devedor"])

# --- Exibir Tabela e Gráfico ---
st.write("### 📉 Evolução do Financiamento")
st.line_chart(df_amortizacao.set_index("Mês")[["Saldo Devedor"]])
st.dataframe(df_amortizacao)
