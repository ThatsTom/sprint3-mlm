from pathlib import Path
import json
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def carregar_dados(caminho):
    return pd.read_csv(caminho)


def separar_dados(df, alvo="risco_alto"):
    X = df.drop(columns=[alvo])
    y = df[alvo]
    return train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)


def criar_preprocessador(X):
    numericas = X.select_dtypes(include="number").columns.tolist()
    categoricas = X.select_dtypes(exclude="number").columns.tolist()
    return ColumnTransformer([
        ("numericas", StandardScaler(), numericas),
        ("categoricas", OneHotEncoder(handle_unknown="ignore"), categoricas),
    ]), numericas, categoricas


def calcular_metricas(modelo, X_teste, y_teste):
    pred = modelo.predict(X_teste)
    proba = modelo.predict_proba(X_teste)[:, 1]
    return {
        "accuracy": accuracy_score(y_teste, pred),
        "precision": precision_score(y_teste, pred, zero_division=0),
        "recall": recall_score(y_teste, pred, zero_division=0),
        "f1_score": f1_score(y_teste, pred, zero_division=0),
        "auc": roc_auc_score(y_teste, proba),
    }


def executar_modelagem(caminho_csv, pasta_resultados):
    df = carregar_dados(caminho_csv)
    X_treino, X_teste, y_treino, y_teste = separar_dados(df)
    preprocessador, numericas, categoricas = criar_preprocessador(X_treino)

    regressao = Pipeline([
        ("preprocessamento", preprocessador),
        ("modelo", LogisticRegression(max_iter=2000, random_state=42)),
    ])
    regressao.fit(X_treino, y_treino)

    random_forest = Pipeline([
        ("preprocessamento", preprocessador),
        ("modelo", RandomForestClassifier(random_state=42, class_weight="balanced")),
    ])
    grade = {
        "modelo__n_estimators": [150, 300],
        "modelo__max_depth": [4, 7, None],
        "modelo__min_samples_leaf": [1, 3, 6],
    }
    busca = GridSearchCV(random_forest, grade, cv=5, scoring="f1", n_jobs=-1)
    busca.fit(X_treino, y_treino)
    melhor_rf = busca.best_estimator_

    metricas = pd.DataFrame([
        {"modelo": "Regressao Logistica", **calcular_metricas(regressao, X_teste, y_teste)},
        {"modelo": "Random Forest Tunada", **calcular_metricas(melhor_rf, X_teste, y_teste)},
    ])

    perm = permutation_importance(
        melhor_rf, X_teste, y_teste, n_repeats=15, random_state=42, scoring="roc_auc"
    )
    importancia = pd.DataFrame({
        "variavel": X_teste.columns,
        "importancia_media_auc": perm.importances_mean,
        "desvio_padrao": perm.importances_std,
    }).sort_values("importancia_media_auc", ascending=False)

    pasta_resultados = Path(pasta_resultados)
    pasta_resultados.mkdir(parents=True, exist_ok=True)
    metricas.to_csv(pasta_resultados / "metricas_modelos.csv", index=False)
    importancia.to_csv(pasta_resultados / "importancia_variaveis.csv", index=False)
    with open(pasta_resultados / "melhores_hiperparametros.json", "w", encoding="utf-8") as f:
        json.dump(busca.best_params_, f, indent=2, ensure_ascii=False)

    return {
        "df": df,
        "X_treino": X_treino,
        "X_teste": X_teste,
        "y_treino": y_treino,
        "y_teste": y_teste,
        "regressao": regressao,
        "random_forest": melhor_rf,
        "busca": busca,
        "metricas": metricas,
        "importancia": importancia,
        "numericas": numericas,
        "categoricas": categoricas,
    }
