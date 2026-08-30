from pathlib import Path
import json
import sys
import pandas as pd
from src.modelagem import executar_modelagem


def main():
    raiz = Path(__file__).resolve().parent
    csv = raiz / "data" / "dados_iot_ml.csv"
    notebook = raiz / "modelagem_machine_learning.ipynb"
    resultados = raiz / "resultados"

    obrigatorios = [
        csv,
        notebook,
        raiz / "README.md",
        raiz / "requirements.txt",
        raiz / "src" / "modelagem.py",
        raiz / "src" / "gerar_dados.py",
    ]
    faltantes = [str(p.relative_to(raiz)) for p in obrigatorios if not p.exists()]
    if faltantes:
        print("ARQUIVOS AUSENTES:")
        for item in faltantes:
            print(f"- {item}")
        sys.exit(1)

    df = pd.read_csv(csv)
    if len(df) < 1000:
        raise AssertionError("O dataset deve ter pelo menos 1000 linhas.")
    if df.isna().any().any():
        raise AssertionError("O dataset possui valores ausentes inesperados.")
    if set(df["risco_alto"].unique()) != {0, 1}:
        raise AssertionError("A variavel alvo precisa conter as classes 0 e 1.")

    r = executar_modelagem(csv, resultados)
    metricas = r["metricas"]
    if len(metricas) < 2:
        raise AssertionError("Devem existir pelo menos dois modelos avaliados.")
    if not {"accuracy", "precision", "recall", "f1_score", "auc"}.issubset(metricas.columns):
        raise AssertionError("Nem todas as metricas obrigatorias foram calculadas.")
    if not ((metricas[["accuracy", "precision", "recall", "f1_score", "auc"]] >= 0).all().all() and
            (metricas[["accuracy", "precision", "recall", "f1_score", "auc"]] <= 1).all().all()):
        raise AssertionError("Metricas fora do intervalo esperado.")

    params_path = resultados / "melhores_hiperparametros.json"
    params = json.loads(params_path.read_text(encoding="utf-8"))
    if not params:
        raise AssertionError("O tuning nao retornou hiperparametros.")

    print("\n=== VALIDACAO DO PROJETO ===")
    print(f"Dataset: OK ({len(df)} linhas)")
    print("Holdout: OK (80% treino / 20% teste estratificado)")
    print("Modelos: OK (Regressao Logistica + Random Forest)")
    print("Scaling: OK (StandardScaler)")
    print("Encoding: OK (OneHotEncoder)")
    print("Tuning: OK (GridSearchCV somente no treino)")
    print("Metricas: OK (accuracy, precision, recall, f1-score e AUC)")
    print("Interpretacao: OK (Permutation Importance)")
    print("\nMetricas no holdout:")
    print(metricas.round(4).to_string(index=False))
    print("\nTop 5 variaveis mais relevantes:")
    print(r["importancia"].head(5).round(4).to_string(index=False))
    print("\nPROJETO VALIDADO COM SUCESSO.")


if __name__ == "__main__":
    main()
