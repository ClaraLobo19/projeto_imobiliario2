import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from haversine import haversine
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def chamar_arquivo():
    
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Caminho absoluto para o arquivo CSV (volta uma pasta e acessa 'base_consolidada.csv')
    file_path = os.path.join(current_dir, '..', 'arquivos', 'base_consolidada.csv')

    # Verifica se o arquivo existe antes de tentar carregar
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Arquivo NÃO encontrado: {file_path}")

    df = pd.read_csv(file_path)
    df = novas_colunas(df)

    # Removendo colunas desnecessárias
    colunas_para_remover = ['endereco','IDH-Educação','IDH', 'preco_bin', 'IDH-Educação','Unnamed: 0']
    df = df.drop(columns=[col for col in colunas_para_remover if col in df.columns], errors='ignore')

    #numericas = [col for col in df.columns if df[col].dtype in ['float64', 'int64','int32'] and col not in ['preço', 'preco p/ m²', 'Regional', 'IDH-Renda']]

    return df


def tirar_outliers(df):
    Q1 = df['preço'].quantile(0.25)
    Q3 = df['preço'].quantile(0.75)
    IQR = Q3 - Q1

    limite_inferior = Q1 - 2 * IQR
    limite_superior = Q3 + 2 * IQR
    df = df[(df['preço'] >= limite_inferior) & (df['preço'] <= limite_superior)]
    return df


def novas_colunas(df):
    df['area_renda'] = df['area m²'] * df['IDH-Renda'] 
    centro_fortaleza = (-3.730451, -38.521798)  # Centro de Fortaleza
    df['distancia_centro'] = df.apply(lambda row: haversine(centro_fortaleza, (row['latitude'], row['longitude'])), axis=1) 
    return df


def separar_dados(df, numericas):
    #numericas = [col for col in df.columns if df[col].dtype in ['float64', 'int64'] and col not in ['preço', 'preco p/ m²']]
    X = df[numericas]
    y = df['preço']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


def cluster(df):
    df = df.reset_index(drop=True)  

    # Selecionar colunas de localização
    coords = df[['latitude', 'IDH-Renda']]

    # Normalizar os dados
    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(coords)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['cluster_geo'] = kmeans.fit_predict(coords_scaled)
    return df, kmeans
    

def gestao_data(df):
    df = df.drop_duplicates()
    df = df[df['bairro'] != 'Siqueira']
    df = df.dropna(subset=['preço'])
    df = df[(df['condominio'] > 1) & (df['condominio'] < 5000)]
    df.reset_index(drop=True, inplace=True)
    
    return df


def load_and_train_model():
    df = chamar_arquivo()
    df = gestao_data(df)
    df = tirar_outliers(df)
    df = novas_colunas(df)
    
    df = cluster(df)[0]
    Kmeans = cluster(df)[1]
    numericas = [col for col in df.columns if df[col].dtype in ['float64', 'int64','int32'] and col not in ['preço', 'preco p/ m²', 'Regional', 'IDH-Renda']]
    
    X_train, X_test, y_train, y_test = separar_dados(df,numericas)
    
    best_params = {
        'colsample_bytree': 1.0, 'learning_rate': 0.1,
        'max_depth': 7, 'n_estimators': 400, 'subsample': 0.9
    }
    
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numericas)]
    )
    xgb_pipeline = make_pipeline(
        preprocessor,
        XGBRegressor(**best_params)
    )
    xgb_pipeline.fit(X_train, y_train)
    
    return xgb_pipeline, numericas, df, Kmeans

#if __name__ == '__main__':
#    model, numericas, df = load_and_train_model()
    # Aqui você pode salvar o modelo usando pickle ou joblib, se desejar.
