"""
Projeto: Predição de Doenças Cardiovasculares com Regressão Logística

Autor: Ana Carolina Martins Pereira

Descrição
----------
Este projeto utiliza um modelo de Regressão Logística para prever a presença
de doenças cardiovasculares a partir de informações clínicas dos pacientes.

Fluxo do projeto:

1. Carregamento dos dados
2. Limpeza e tratamento
3. Análise exploratória
4. Matriz de correlação
5. Preparação dos dados
6. Treinamento
7. Avaliação
8. Curva ROC
"""

# ==========================
# IMPORTAÇÃO DAS BIBLIOTECAS
# ==========================

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ==========================
# CONFIGURAÇÕES
# ==========================

pd.set_option("display.max_columns", None)

RANDOM_STATE = 42
TEST_SIZE = 0.20

NUMERIC_COLUMNS = [
    "age",
    "height",
    "weight",
]


# ==========================
# CARREGAMENTO DOS DADOS
# ==========================

def carregar_dados():

    """
    Carrega a base de dados.
    """

    df = pd.read_csv(
        "data/CARDIO_BASE.csv",
        delimiter=";"
    )

    return df


# ==========================
# TRATAMENTO DOS DADOS
# ==========================

def tratar_dados(df):

    """
    Realiza limpeza e tratamento da base.
    """

    print(df.info())

    # Conversão da coluna weight

    df["weight"] = pd.to_numeric(
        df["weight"].str.replace(",", ".", regex=False)
    )

    # Padronização da variável gênero

    gender_mapping = {
        1: 0,
        2: 1
    }

    df["gender"] = df["gender"].replace(gender_mapping)

    print(df.describe())

    return df


# ==========================
# REMOÇÃO DE OUTLIERS
# ==========================

def remover_outliers(df):

    """
    Remove registros com alturas biologicamente
    incompatíveis.
    """

    print("\nMenores alturas")
    print(df["height"].sort_values().head(20))

    print("\nMaiores alturas")
    print(df["height"].sort_values().tail(20))

    df = df[
        (df["height"] > 120)
        &
        (df["height"] <= 220)
    ]

    return df


# ==========================
# BOXPLOTS
# ==========================

def plot_boxplots(df):

    """
    Plota boxplots das variáveis numéricas.
    """

    plt.figure(figsize=(5, 5))
    plt.boxplot(df["age"])
    plt.title("Boxplot - Age")
    plt.savefig("images/box_age.png", dpi=300, bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(5, 5))
    plt.boxplot(df["height"])
    plt.title("Boxplot - Height")
    plt.savefig("images/graf_height.png", dpi=300, bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(5, 5))
    plt.boxplot(df["weight"])
    plt.title("Boxplot - Weight")
    plt.savefig("images/box_weight.png", dpi=300, bbox_inches="tight")
    plt.show()


# ==========================
# ANÁLISE EXPLORATÓRIA
# ==========================

def grafico_genero(df):

    """
    Distribuição da doença cardíaca por gênero.
    """

    contagem = pd.crosstab(
        df["gender"],
        df["cardio_disease"]
    )

    contagem.index = [
        "Homens",
        "Mulheres"
    ]

    contagem.plot(
        kind="bar",
        stacked=True
    )

    plt.title(
        "Doença Cardíaca por Gênero"
    )

    plt.xlabel("Gênero")
    plt.ylabel("Quantidade")

    plt.legend(
        title="Doença",
        labels=[
            "Não",
            "Sim"
        ]
    )
    plt.savefig("images/graf1.png", dpi=300, bbox_inches="tight")
    plt.show()


def grafico_idade(df):

    """
    Boxplot da idade por doença cardíaca.
    """

    media = df.groupby(
        "cardio_disease"
    )["age"].mean()

    print(media)

    plt.figure(figsize=(6, 5))

    df.boxplot(
        column="age",
        by="cardio_disease"
    )

    plt.title(
        "Idade x Doença Cardíaca"
    )

    plt.suptitle("")

    plt.xlabel(
        "Doença Cardíaca"
    )

    plt.ylabel("Idade")

    plt.xticks(
        [1, 2],
        ["Não", "Sim"]
    )
    plt.savefig("images/graf2.png", dpi=300, bbox_inches="tight")
    plt.show()


def grafico_atividade(df):

    """
    Distribuição da doença cardíaca
    por atividade física.
    """

    contagem = pd.crosstab(
        df["active"],
        df["cardio_disease"]
    )

    contagem.index = [
        "Sedentários",
        "Ativos"
    ]

    contagem.plot(
        kind="bar",
        stacked=True
    )

    plt.title(
        "Atividade Física x Doença Cardíaca"
    )

    plt.xlabel("Atividade Física")
    plt.ylabel("Quantidade")

    plt.legend(
        title="Doença",
        labels=[
            "Não",
            "Sim"
        ]
    )
    plt.savefig("images/graf3.png", dpi=300, bbox_inches="tight")
    plt.show()

# ==========================
# MATRIZ DE CORRELAÇÃO
# ==========================

def matriz_correlacao(df):
    """
    Exibe a matriz de correlação entre as variáveis.
    """

    plt.figure(figsize=(10, 8))

    correlacao = df.corr()

    sns.heatmap(
        correlacao,
        annot=True,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
    )

    plt.title("Matriz de Correlação")
    plt.savefig("images/graf4.png", dpi=300, bbox_inches="tight")
    plt.show()


# ==========================
# PREPARAÇÃO DOS DADOS
# ==========================

def preparar_dados(df):
    """
    Separa os dados em treino e teste e realiza
    a padronização das variáveis numéricas.
    """

    X = df.drop("cardio_disease", axis=1)
    y = df["cardio_disease"]

    print("\nDistribuição das classes:")
    print(y.value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    scaler = StandardScaler()

    X_train[NUMERIC_COLUMNS] = scaler.fit_transform(
        X_train[NUMERIC_COLUMNS]
    )

    X_test[NUMERIC_COLUMNS] = scaler.transform(
        X_test[NUMERIC_COLUMNS]
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
    )


# ==========================
# TREINAMENTO
# ==========================

def treinar_modelo(X_train, y_train):
    """
    Treina o modelo de regressão logística.
    """

    modelo = LogisticRegression(
        random_state=RANDOM_STATE
    )

    modelo.fit(
        X_train,
        y_train,
    )

    return modelo


# ==========================
# COEFICIENTES
# ==========================

def mostrar_coeficientes(modelo, X_train):
    """
    Exibe o intercepto e os coeficientes
    da regressão logística.
    """

    print("\nIntercepto")

    print(modelo.intercept_)

    coeficientes = pd.DataFrame(
        {
            "Variável": X_train.columns,
            "Coeficiente": modelo.coef_[0],
        }
    )

    coeficientes = coeficientes.sort_values(
        by="Coeficiente",
        key=abs,
        ascending=False,
    )

    print("\nCoeficientes")

    print(coeficientes)


# ==========================
# AVALIAÇÃO
# ==========================

def avaliar_modelo(modelo, X, y, conjunto="Treino"):
    """
    Avalia o desempenho do modelo.
    """

    previsoes = modelo.predict(X)

    print(f"\n===== {conjunto.upper()} =====")

    print(
        classification_report(
            y,
            previsoes,
        )
    )

    return previsoes


# ==========================
# CURVA ROC
# ==========================

def curva_roc(modelo, X_test, y_test):
    """
    Plota a curva ROC e calcula a AUC.
    """

    probabilidades = modelo.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(
        y_test,
        probabilidades,
    )

    auc = roc_auc_score(
        y_test,
        probabilidades,
    )

    print(f"\nAUC: {auc:.3f}")

    plt.figure(figsize=(7, 6))

    plt.plot(
        fpr,
        tpr,
        lw=2,
        label=f"AUC = {auc:.3f}",
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
    )

    plt.xlabel("Taxa de Falsos Positivos")
    plt.ylabel("Taxa de Verdadeiros Positivos")

    plt.title("Curva ROC")

    plt.legend()
    plt.savefig("images/roc_curve.png", dpi=300, bbox_inches="tight")
    plt.show()


# ==========================
# FUNÇÃO PRINCIPAL
# ==========================

def main():
    """
    Executa todo o fluxo do projeto.
    """

    print("=" * 60)
    print("PREDIÇÃO DE DOENÇAS CARDIOVASCULARES")
    print("=" * 60)

    # -----------------------
    # Carregamento
    # -----------------------

    df = carregar_dados()

    # -----------------------
    # Limpeza
    # -----------------------

    df = tratar_dados(df)

    df = remover_outliers(df)

    # -----------------------
    # Análise exploratória
    # -----------------------

    plot_boxplots(df)

    grafico_genero(df)

    grafico_idade(df)

    grafico_atividade(df)

    matriz_correlacao(df)

    # -----------------------
    # Preparação
    # -----------------------

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = preparar_dados(df)

    # -----------------------
    # Modelo
    # -----------------------

    modelo = treinar_modelo(
        X_train,
        y_train,
    )

    mostrar_coeficientes(
        modelo,
        X_train,
    )

    # -----------------------
    # Avaliação
    # -----------------------

    avaliar_modelo(
        modelo,
        X_train,
        y_train,
        "Treino",
    )

    avaliar_modelo(
        modelo,
        X_test,
        y_test,
        "Teste",
    )

    curva_roc(
        modelo,
        X_test,
        y_test,
    )


# ==========================
# EXECUÇÃO
# ==========================

if __name__ == "__main__":
    main()