# Resumo dos resultados

## Problema

Classificação binária de observações IoT em:

- `0`: normal;
- `1`: alto risco.

## Dataset

- 1.400 observações.
- 12 variáveis explicativas.
- Aproximadamente 41,8% das observações classificadas como alto risco.

## Holdout

- 80% treino: 1.120 registros.
- 20% teste: 280 registros.
- Divisão estratificada.
- `random_state=42`.

## Modelos

### Regressão Logística

- Accuracy: 0,7107
- Precision: 0,6875
- Recall: 0,5641
- F1-score: 0,6197
- AUC: 0,7558

### Random Forest tunada

- Accuracy: 0,6929
- Precision: 0,6348
- Recall: 0,6239
- F1-score: 0,6293
- AUC: 0,7319

## Leitura dos resultados

A Regressão Logística apresentou a melhor AUC e a melhor accuracy, mostrando boa capacidade geral de discriminação. A Random Forest tunada apresentou recall e F1-score ligeiramente superiores, identificando uma proporção maior dos casos realmente classificados como alto risco.

Não existe contradição nisso: métricas diferentes medem aspectos diferentes do comportamento de um classificador.

## Variáveis mais relevantes

Segundo a Permutation Importance aplicada à Random Forest no holdout:

1. perda de pacotes;
2. tentativas de login falhas;
3. temperatura;
4. criptografia ativa;
5. umidade.

Esses resultados indicam quais variáveis carregaram mais informação preditiva no conjunto analisado. Importância preditiva não deve ser interpretada automaticamente como causalidade.
