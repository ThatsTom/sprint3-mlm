# Como testar e validar o projeto

## Teste rápido

Abra o terminal na pasta do projeto e execute:

```bash
python validar_projeto.py
```

Se no Windows o comando `python` não estiver disponível:

```bash
py validar_projeto.py
```

A validação executa novamente a modelagem e confere:

- dataset com as classes 0 e 1;
- holdout 80/20 estratificado;
- dois modelos treinados;
- StandardScaler;
- OneHotEncoder;
- tuning com GridSearchCV;
- accuracy, precision, recall, F1-score e AUC;
- interpretação com Permutation Importance.

O resultado esperado termina com:

```text
PROJETO VALIDADO COM SUCESSO.
```

## Teste completo pelo notebook

1. Instale as dependências:

```bash
pip install -r requirements.txt
```

2. Abra o notebook:

```bash
jupyter notebook modelagem_machine_learning.ipynb
```

3. No Jupyter use **Kernel > Restart Kernel and Run All Cells**.

4. Confirme que nenhuma célula fica vermelha ou apresenta erro.

5. Verifique a pasta `resultados`.

## Resultados de referência do pacote entregue

### Regressão Logística

- Accuracy: aproximadamente 0,7107
- Precision: aproximadamente 0,6875
- Recall: aproximadamente 0,5641
- F1-score: aproximadamente 0,6197
- AUC: aproximadamente 0,7558

### Random Forest tunada

- Accuracy: aproximadamente 0,6929
- Precision: aproximadamente 0,6348
- Recall: aproximadamente 0,6239
- F1-score: aproximadamente 0,6293
- AUC: aproximadamente 0,7319

Pequenas diferenças podem ocorrer em versões futuras das bibliotecas, mas com o mesmo dataset, `random_state=42` e versões semelhantes os valores devem ficar próximos.

## Como validar a interpretação

Abra:

```text
resultados/importancia_variaveis.csv
```

No pacote entregue, as cinco primeiras variáveis foram:

1. `perda_pacotes_pct`
2. `tentativas_login_falhas`
3. `temperatura_c`
4. `criptografia_ativa`
5. `umidade_pct`

O arquivo `resultados/importancia_variaveis.png` apresenta o mesmo ranking em gráfico.

## Como gerar novamente o dataset

```bash
python src/gerar_dados.py
```

O script recria `data/dados_iot_ml.csv` com `seed=42`, mantendo a geração reproduzível.

Depois rode novamente:

```bash
python validar_projeto.py
```

## Conferência antes da entrega

- Preencha o nome do aluno no topo do notebook.
- Execute todas as células.
- Confirme que os gráficos aparecem.
- Confirme que `resultados/metricas_modelos.csv` existe.
- Confirme que `resultados/melhores_hiperparametros.json` existe.
- Confirme que `resultados/importancia_variaveis.csv` existe.
- Publique a pasta no GitHub.
- Copie o link público ou acessível ao professor.
