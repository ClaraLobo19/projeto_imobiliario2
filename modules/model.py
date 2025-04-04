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
import joblib

def chamar_arquivo():
    
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Caminho absoluto para o arquivo CSV (volta uma pasta e acessa 'base_consolidada.csv')
    file_path = os.path.join(current_dir, '..', 'arquivos', 'base_consolidada.csv')

    # Verificando se o arquivo existe antes de tentar carregar
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Arquivo NÃO encontrado: {file_path}")

    df = pd.read_csv(file_path)

    return df

def tirar_outliers(df):
    Q1 = df['preco'].quantile(0.25)
    Q3 = df['preco'].quantile(0.75)
    IQR = Q3 - Q1

    limite_inferior = Q1 - 2 * IQR
    limite_superior = Q3 + 2 * IQR
    df = df[(df['preco'] >= limite_inferior) & (df['preco'] <= limite_superior)]
    return df


def novas_colunas(df):
    df['area_renda'] = df['aream2'] * df['idh_renda'] 

    df['quartos_por_m2']= df['Quartos'] / df['aream2']
    df['banheiros_por_quarto']= df['banheiros'] / df['Quartos']
    centro_fortaleza = (-3.730451, -38.521798)  # Centro de Fortaleza
    df['distancia_centro'] = df.apply(lambda row: haversine(centro_fortaleza, (row['latitude'], row['longitude'])), axis=1) 
    return df


def separar_dados(df, numericas):
    X = df[numericas]
    y = df['preco']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def cluster(df):
    df = df.reset_index(drop=True)  

    # Selecionar colunas de localização
    coords = df[['latitude', 'idh_renda']]

    # Normalizar os dados
    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(coords)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['cluster_geo'] = kmeans.fit_predict(coords_scaled)
    return df, kmeans
    

def gestao_data(df):
    df = df.drop_duplicates()
    df = df[df['bairro'] != 'Siqueira']

    df = df.dropna(subset=['preço', 'latitude'])
    df = df[(df['condominio'] > 1) & (df['condominio'] < 5000)]

    colunas_para_remover = ['endereco', 'IDH', 'preco_bin','Unnamed: 0']
    df = df.drop(columns=[col for col in colunas_para_remover if col in df.columns], errors='ignore')

    df.rename(columns={'preço': 'preco', 'IDH-Renda': 'idh_renda', 'IDH-Longevidade': 'idh_longevidade','IDH-Educação':'idh_educacao',
                        'area m²': 'aream2','preco p/ m²':'preco p/m2'}, inplace=True)
    df = df.reset_index(drop=True)

    df = novas_colunas(df)
    
    return df


def data_frame():
    # Carregar o arquivo CSV
    print("Carregando arquivo...")

    df = chamar_arquivo()
    print("Após carregar:", df.shape)
    
    df = gestao_data(df)
    print("Após gestao_data:", df.shape)

    df = novas_colunas(df)
    print("Após novas_colunas:", df.shape)

    df = tirar_outliers(df)
    print("Após tirar_outliers:", df.shape)

    return df



def load_and_train_model():
    
    df = data_frame()
    df = cluster(df)[0]
    kmeans_model = cluster(df)[1]


    numericas = [col for col in df.columns if df[col].dtype in ['float64', 'int64','int32'] and col not in ['preco', 'preco p/m2', 'Regional', 'idh_renda']]
    print(f'df.columns: {df.columns}')
    
    X_train, X_test, y_train, y_test = separar_dados(df,numericas)
    
    print(f'Dados de treino: {X_train}')
    print(f'{X_train.info()}')

    best_params = {
        'colsample_bytree': 1.0, 'learning_rate': 0.1,
        'max_depth': 7, 'n_estimators': 400, 'subsample': 0.9
    }

    # Criando o modelo de xbgboost
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numericas)]
    )
    xgb_pipeline = make_pipeline(
        preprocessor,
        XGBRegressor(**best_params)
    )
    xgb_pipeline.fit(X_train, y_train)
    
    return xgb_pipeline, kmeans_model


# Definindo os caminhos para salvar os modelos
if not os.path.exists("models"):
    os.makedirs("models")
 
modelo_treinado_path = 'models/modelo4.pkl'
kmeans_path = 'models/kmeans4.pkl' 
treinar_modelo = True
 
if treinar_modelo or not (os.path.exists(modelo_treinado_path) and os.path.exists(kmeans_path)):
    print("Treinando o modelo...")
    xgb_pipeline, kmeans_model = load_and_train_model()
 
    joblib.dump(xgb_pipeline, modelo_treinado_path) # Salva o modelo  
    joblib.dump(kmeans_model, kmeans_path)            
   
    print("Modelos treinados e salvos com sucesso!")
else:
    print("Modelos já existem, pulando treinamento.")

#------------------------------------------------------------------------------------------

