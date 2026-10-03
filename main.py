from pre_processamento import pre_processar_dados
from Relevancia_binaria import RB


def main():
    print("=== INICIANDO EXPERIMENTO STEAM ===")
    
    # Preparação dos Dados
    # Chama a função que lê o CSV, limpa e retorna as matrizes X e y prontas para uso.
    caminho_arquivo = '/home/vini21/projetos/AM2/PAULO/steam_filtrado.csv'
    X_train, X_test, y_train, y_test, mlb = pre_processar_dados(caminho_arquivo)
    
    print("\n--- Resumo dos Dados ---")
    print(f"Treino: X={X_train.shape}, y={y_train.shape}")
    print(f"Teste:  X={X_test.shape}, y={y_test.shape}")
    
    #  Execução do Modelo 1 (Baseline / Binary Relevance)
    # Testa a abordagem simples que não considera a correlação entre os gêneros
    modelo1, pred1, hl1, f11 = RB(X_train, y_train, X_test, y_test)
    
   
# Garante que a função main() só seja executada se este arquivo for rodado diretamente
if __name__ == "__main__":
    main()


