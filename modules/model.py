from turtle import st
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


#@st.cache_data
import os
import pandas as pd


def chamar_arquivo():
    # Definir seu modelo e pipelines
    scaler = StandardScaler()

    
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Caminho absoluto para o arquivo CSV (volta uma pasta e acessa 'arquivos/teste.csv')
    file_path = os.path.join(current_dir, '..', 'arquivos', 'teste.csv')

    # Verifica se o arquivo existe antes de tentar carregar
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Arquivo NÃO encontrado: {file_path}")

    print(f"Arquivo encontrado: {file_path}")

   
    df = pd.read_csv(file_path)

    # Removendo colunas desnecessárias
    colunas_para_remover = ['Unnamed: 0', 'Unnamed: 0.1', 'IDH_x', 'IDH-Renda_x', 
                             'IDH-Longevidade_x', 'IDH-Educação_x', 'Regional_x', 
                             'Regional_y', 'numero', 'Regional', 'preco p/ m²', 
                             'IDH-Educação', 'IDH', 'preço_cond_ratio']

    # Verifica se as colunas existem antes de tentar remover
    df = df.drop(columns=[col for col in colunas_para_remover if col in df.columns], errors='ignore')

    numericas = [col for col in df.columns if df[col].dtype in ['float64', 'int64'] and col not in ['preço', 'preco p/ m²']]

    return df, numericas


def tirar_outliers(df):
    Q1 = df['preço'].quantile(0.25)
    Q3 = df['preço'].quantile(0.75)
    IQR = Q3 - Q1

    limite_inferior = Q1 - 2 * IQR
    limite_superior = Q3 + 2 * IQR
    df = df[(df['preço'] >= limite_inferior) & (df['preço'] <= limite_superior)]
    return df


def separar_dados(df,numericas):
    #numericas = [col for col in df.columns if df[col].dtype in ['float64', 'int64'] and col not in ['preço', 'preco p/ m²']]
    X = df[numericas]
    y = df['preço']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

    
def load_and_train_model():
    df,numericas = chamar_arquivo()
   
    df = df.drop_duplicates()
    df = df[df['bairro'] != 'Siqueira']
    df = df.dropna(subset=['preço'])
    df = df[df['condominio'] < 10000]

    df.reset_index(drop=True, inplace=True)
    df = tirar_outliers(df)
    X_train, X_test, y_train, y_test = separar_dados(df,numericas)

    
    best_params = {
        'colsample_bytree': 1.0, 
        'learning_rate': 0.1, 
        'max_depth': 3, 
        'n_estimators': 500, 
        'subsample': 0.9
    }
    
    preprocessor = ColumnTransformer([('num', StandardScaler(), numericas)])
    
    xgb_pipeline = make_pipeline(
        preprocessor,
        XGBRegressor(**best_params)
    )

    xgb_pipeline.fit(X_train, y_train)
    
    return xgb_pipeline, numericas, df

#if __name__ == '__main__':
#    model, numericas, df = load_and_train_model()
    # Aqui você pode salvar o modelo usando pickle ou joblib, se desejar.
