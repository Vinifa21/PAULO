import time
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import hamming_loss, f1_score, accuracy_score, classification_report

def RB(X_train, y_train, X_test, y_test):
    print("\n" + "="*40)
    print(" INICIANDO MODELO 1: BINARY RELEVANCE")
    print("="*40)
    
    # O OneVsRestClassifier implementa a estratégia Binary Relevance.
    # Ele treina um classificador independente (Regressão Logística) para cada um dos 33 gêneros.
    # Ele ignora completamente se um gênero tem relação com outro.
    modelo_br = OneVsRestClassifier(LogisticRegression(max_iter=1000, random_state=42))
    
    print("[Modelo 1] Treinando (isso pode levar alguns segundos)...")
    inicio_treino = time.time()
    
    # O ajuste (fit) treina todos os 33 classificadores internamente
    modelo_br.fit(X_train, y_train)
    
    fim_treino = time.time()
    print(f"[Modelo 1] Treinamento concluído em {fim_treino - inicio_treino:.2f} segundos.")
    
    print("[Modelo 1] Realizando previsões no conjunto de teste...")
    y_pred = modelo_br.predict(X_test)
    
    # Avaliação de Desempenho
    # Hamming Loss: Porcentagem de erros individuais (ex: previu Ação, mas não era). Mais próximo de 0 é melhor.
    hl = hamming_loss(y_test, y_pred)

    
    
    # Micro F1-Score: Média balanceada entre precisão e recall considerando todas as classes juntas. Mais próximo de 1 é melhor.
    f1 = f1_score(y_test, y_pred, average='micro')

    exact_match = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)

    
    print("\n--- Resultados: Modelo 1 ---")
    print(f"Hamming Loss: {hl:.4f}")
    print(f"Micro F1-Score: {f1:.4f}")
    print(f"Macro F1-Score: {f1_macro:.4f}")
    print(f"Exact Match Ratio: {exact_match:.4f}")
    
    
    return modelo_br, y_pred, hl, f1

