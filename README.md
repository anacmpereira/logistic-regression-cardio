# logistic-regression-cardio
# ❤️ Predição de Doenças Cardiovasculares com Regressão Logística

<p align="center">

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?style=for-the-badge&logo=scikitlearn)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-blue?style=for-the-badge)
![Seaborn](https://img.shields.io/badge/Seaborn-Statistical%20Plots-4C72B0?style=for-the-badge)

</p>

---

## 📖 Sobre o Projeto

Este projeto aplica um modelo de **Regressão Logística** para prever a ocorrência de doenças cardiovasculares utilizando dados clínicos de pacientes.

O objetivo foi desenvolver um fluxo completo de Ciência de Dados, contemplando desde o tratamento dos dados até a avaliação do modelo de Machine Learning.

Durante o desenvolvimento foram aplicadas técnicas de:

- Limpeza e tratamento dos dados;
- Remoção de registros inconsistentes;
- Análise Exploratória de Dados (EDA);
- Análise de Correlação;
- Padronização das variáveis;
- Treinamento de um modelo de Regressão Logística;
- Avaliação através de métricas de classificação;
- Curva ROC e cálculo da AUC.

---

# 📂 Estrutura do Projeto

```
regressao-logistica-cardio/

│

├── data/
│   └── CARDIO_BASE.csv

├── images/
│   ├── roc_curve.png
│   ├── correlation_matrix.png
│   ├── genero_cardio.png
│   └── idade_cardio.png

├── src/
│   └── main.py

├── requirements.txt

├── README.md

└── .gitignore
```

---

# 📊 Base de Dados

A base contém informações clínicas de pacientes adultos.

Entre as principais variáveis estão:

| Variável | Descrição |
|-----------|-----------|
| age | Idade |
| gender | Sexo |
| height | Altura |
| weight | Peso |
| cholesterol | Colesterol |
| gluc | Glicose |
| smoke | Tabagismo |
| alco | Consumo de álcool |
| active | Atividade física |
| cardio_disease | Presença de doença cardiovascular |

---

# ⚙️ Fluxo do Projeto

```text
Leitura da Base
        │
        ▼
Tratamento dos Dados
        │
        ▼
Remoção de Outliers
        │
        ▼
Análise Exploratória
        │
        ▼
Matriz de Correlação
        │
        ▼
Separação Treino/Teste
        │
        ▼
Padronização
        │
        ▼
Regressão Logística
        │
        ▼
Avaliação do Modelo
        │
        ▼
Curva ROC
```

---

# 📈 Análise Exploratória

Foram realizadas análises gráficas para compreender a distribuição das variáveis e sua relação com a presença de doenças cardiovasculares.

Os gráficos incluem:

- Distribuição da doença por gênero;
- Distribuição da idade por presença de doença;
- Relação entre atividade física e doença cardiovascular;
- Boxplots para identificação de outliers;
- Matriz de Correlação.

---

# 🤖 Modelo de Machine Learning

Foi utilizado o algoritmo de **Regressão Logística**, amplamente empregado em problemas de classificação binária.

As etapas de modelagem incluíram:

- Separação treino/teste (80/20);
- Padronização das variáveis numéricas;
- Ajuste do modelo;
- Avaliação no conjunto de treino;
- Avaliação no conjunto de teste.

---

# 📊 Resultados

O modelo apresentou desempenho moderado na classificação dos pacientes.

### Principais métricas

| Métrica | Valor |
|----------|------:|
| Acurácia | **65%** |
| AUC | **0.65** |

Os resultados indicam boa capacidade de generalização, uma vez que o desempenho observado no conjunto de teste foi semelhante ao obtido durante o treinamento.

---

# 🔍 Principais Insights

A análise exploratória mostrou que:

- Pacientes com doença cardiovascular apresentaram idade média superior.
- Colesterol apresentou uma das maiores correlações com a variável alvo.
- Peso também apresentou associação positiva com doenças cardiovasculares.
- Não foram observadas diferenças expressivas entre homens e mulheres na proporção de indivíduos doentes.
- A prática de atividade física não apresentou diferenças visuais marcantes entre os grupos.

---

# 📷 Resultados Visuais

## Matriz de Correlação

<img src="images/correlation_matrix.png" width="750">

---

## Curva ROC

<img src="images/roc_curve.png" width="650">

---

## Distribuição por Gênero

<img src="images/genero_cardio.png" width="650">

---

## Idade por Doença Cardiovascular

<img src="images/idade_cardio.png" width="650">

---

# 🚀 Como executar

## 1. Clone o repositório

```bash
git clone https://github.com/SEU-USUARIO/regressao-logistica-cardio.git
```

Entre na pasta

```bash
cd regressao-logistica-cardio
```

---

## 2. Crie um ambiente virtual

Windows

```bash
python -m venv venv

venv\Scripts\activate
```

Linux/Mac

```bash
python3 -m venv venv

source venv/bin/activate
```

---

## 3. Instale as dependências

```bash
pip install -r requirements.txt
```

---

## 4. Execute o projeto

```bash
python src/main.py
```

---

# 📦 Bibliotecas Utilizadas

- Pandas
- Matplotlib
- Seaborn
- Scikit-Learn

---

# 💡 Melhorias Futuras

O projeto pode ser expandido utilizando:

- Random Forest
- XGBoost
- LightGBM
- Validação Cruzada
- Grid Search
- Pipeline do Scikit-Learn
- Engenharia de Atributos
- Feature Selection
- Balanceamento automático das classes
- Deploy utilizando Streamlit

---

# 👩‍💻 Autora

**Ana Carolina Martins Pereira**

Bióloga | Professora | Pesquisadora

Este projeto foi desenvolvido como atividade prática de Machine Learning utilizando Python e Scikit-Learn.

---
