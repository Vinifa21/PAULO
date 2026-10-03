import pandas as pd

def filtrar_dataset_steam(caminho_csv_original, caminho_csv_novo):
    print(f"A ler o ficheiro original: {caminho_csv_original}...")
    
    # Carregar o dataset original
    
    df_completo = pd.read_csv(caminho_csv_original, low_memory=False)
    
    # Definir as colunas que queremos manter
    colunas_desejadas = ['appid', 'name', 'categories', 'genres', 'tags']
    
    # Verificar se todas as colunas desejadas existem no dataset original
    colunas_existentes = df_completo.columns.tolist()
    colunas_em_falta = [col for col in colunas_desejadas if col not in colunas_existentes]
    
    if colunas_em_falta:
        print(f"ERRO: Faltam as seguintes colunas no CSV original: {colunas_em_falta}")
        return
    
    
    df_filtrado = df_completo[colunas_desejadas].copy()
    
    
    df_filtrado = df_filtrado.dropna(subset=['genres', 'tags'])
    
    print(f"A guardar o novo dataset em: {caminho_csv_novo}...")
    
    df_filtrado.to_csv(caminho_csv_novo, index=False)
    
    print("Sucesso!")
    print(f"O dataset original tinha {df_completo.shape[1]} colunas e {df_completo.shape[0]} linhas.")
    print(f"O novo dataset tem {df_filtrado.shape[1]} colunas e {df_filtrado.shape[0]} linhas.")



ficheiro_original = '/home/vini21/projetos/AM2/PAULO/games_march2025_cleaned.csv' 
ficheiro_novo = 'steam_filtrado.csv'
    
filtrar_dataset_steam(ficheiro_original, ficheiro_novo)