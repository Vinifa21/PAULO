import pandas as pd
import ast
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

def pre_processar_dados(caminho_csv):
    print("[Pré-processamento] Carregando o dataset...")
    df = pd.read_csv(caminho_csv)
    
    print("[Pré-processamento] Convertendo strings para listas...")
    colunas_lista = ['categories', 'genres', 'tags']
    
    # O CSV salva listas como texto ("['Ação', 'RPG']"). 
    # O ast.literal_eval converte esse texto de volta para listas reais do Python.
    for col in colunas_lista:
        df[col] = df[col].fillna('[]').apply(ast.literal_eval)
        
    # Removemos jogos sem gêneros definidos, pois são a variável que queremos prever.
    df = df[df['genres'].map(len) > 0]
    
    print("[Pré-processamento] Criando a matriz de features (X) com TF-IDF...")
    # Unificamos tags e categorias numa única string para cada jogo
    df['features_texto'] = df['tags'].apply(lambda x: ' '.join(x)) + ' ' + \
                           df['categories'].apply(lambda x: ' '.join(x))
    
    # Converte o texto em uma matriz numérica considerando a relevância das palavras.
    # max_features=5000 limita o vocabulário para evitar consumo excessivo de memória RAM.
    tfidf = TfidfVectorizer(max_features=5000)
    X = tfidf.fit_transform(df['features_texto'])
    
    print("[Pré-processamento] Criando a matriz alvo (y)...")
    # Transforma a lista de gêneros em múltiplas colunas binárias (0 ou 1 para cada gênero)
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df['genres'])
    
    print("[Pré-processamento] Dividindo em dados de treino e teste...")
    # Separa 80% dos dados para treinar o modelo e 20% para testar a sua capacidade de generalização
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, mlb