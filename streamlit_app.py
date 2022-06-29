from logging import exception
import numpy as np
import pandas as pd
import streamlit as st
import warnings
from PIL import Image
warnings.simplefilter(action='ignore', category=FutureWarning)
from mlxtend.frequent_patterns import apriori, fpmax, fpgrowth
from mlxtend.frequent_patterns import association_rules
import matplotlib.pyplot as plt

img = Image.open(r'img/ds.png')
st.set_page_config(
     page_title="TERA | Association Rules",
     page_icon=img,
     layout="wide",
     initial_sidebar_state="expanded")

def check_password():
    """Returns `True` if the user had a correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if (
            st.session_state["username"] in st.secrets["passwords"]
            and st.session_state["password"]
            == st.secrets["passwords"][st.session_state["username"]]
        ):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store username + password
            del st.session_state["username"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show inputs for username + password.
        st.text_input("Username", on_change=password_entered, key="username")
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input("Username", on_change=password_entered, key="username")
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Usuário ou senha incorreto")
        return False
    else:
        # Password correct.
        return True

if check_password():

    st.title('Analise de Hiperparametros')
    uploaded_file = st.sidebar.file_uploader("Selecione sua base de dados no formato csv", type=".csv")
    dataset_example = st.sidebar.checkbox('Utilizar dataset de exemplo')


    def analise(df):
        st.markdown("### Data preview")
        st.dataframe(df.head())

        st.markdown("### Selecione os parametros para aplicar o modelo")
        with st.form(key="my_form"):
            id_venda = st.selectbox(
                "Coluna com ID's das vendas",
                options=df.columns,
                index = 0,
                help = "Selecione a coluna que contem o código referente aos pedidos ou vendas. Exemplo: CD_PEDIDO, COD_VENDA, ID_VENDA."
            )

            id_prod = st.selectbox(
                "Coluna com Identificação dos produtos",
                options=df.columns,
                index = 1,
                help="Selecione a coluna que contém os códigos dos produtos. Exemplo: CD_MATERIAL, COD_PRODUTO, ID_PRODUTO.",
                )
            
            mod = st.selectbox(
                "Selecione o modelo a ser aplicado",
                options = ['fpgrowth', 'apriori'],
                index = 0,
                help = "Para mais detalhes acesse: [frequent_patterns](http://rasbt.github.io/mlxtend/api_subpackages/mlxtend.frequent_patterns/) "
            )
            st.write("OBS.: O modelo apriori tem um custo computacional maior que o fpgrowth e ambos resultam no mesmo conjunto de regras")

            min_support = st.text_input('Digite os min_support que deseja testar, separe-os por virgula.', '', help = "Exemplo: 0.1,0.2,0.3,0.4")

            regras = st.selectbox(
                "Selecione a regra que deseja comparar",
                options = ['confidence', 'lift', 'leverage', 'conviction'],
                index = 0,
                help = "Para mais detalhes acesse: [association_rules](http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/association_rules/)"
            )
            #st.write(f"OBS.: Os valores de threshold utilizados serão: {min_threshold}")
            min_threshold = st.multiselect(
                'Escolha os valores de threshold',
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

            plot = st.radio(
            "Qual gráfico gostaria de visualizar?",
            ('Unificado', 'Individual'),
            help = "Ao selecionar o gráfico unificado, será plotada uma linha para cada min_support no mesmo gráfico, ao selecionar Individual será plotado um gráfico para cada min_support"
            )

            submit_button = st.form_submit_button(
                label="Submit"
                )
            if submit_button:
                with st.spinner('Executando...'):
                    df = pd.crosstab(df[id_venda], df[id_prod])
                    df[df>1] = 1

                    data = df
                    if mod == 'fpgrowth':
                        algorythm = fpgrowth
                    elif mod == 'apriori':
                        algorythm = apriori
                    else:
                        pass
                    min_supports = min_support.replace(' ', '')
                    min_supports = min_supports.split(',')
                    for i in range(len(min_supports)):
                        min_supports[i] = float(min_supports[i])
                    parameter = regras
                    if plot == 'Individual':
                        plot = 1
                    elif plot == 'Unificado':
                        plot = 2
                    else:
                        pass

                    rules_levels = []
                    coluna1 = []
                    coluna2 = []
                    coluna3 = []

                    for s in range(len(min_supports)):
                        rules_levels.append([])
                        itemsets = algorythm(data, min_support=min_supports[s], use_colnames=True)
                        if itemsets.empty == True:
                            raise exception('O valor de min_threshold é muito alto, resultando em um DataFrame vazio, altere os valores e tente novamente')
                        else:
                            pass
                        for c in range(len(min_threshold)):
                            rl = association_rules(itemsets, metric = parameter, min_threshold = min_threshold[c])
                            rules_levels[s].append(len(rl.index))
                            
                    for s in range(len(min_supports)):
                        for i in range(len(rules_levels[0])):
                            coluna1.append(min_supports[s])
                            
                    for c in range(len(min_supports)):
                        for i in range(len(min_threshold)):
                            coluna2.append(min_threshold[i])
                    
                    for r in range(len(rules_levels)):
                        for i in range(len(rules_levels[r])):
                            coluna3.append(rules_levels[r][i])
                    
                    dicionario = {
                    'min_support': coluna1,
                    'confidence': coluna2,
                    'qt_itemsets': coluna3
                    }
                    df_classe = pd.DataFrame(dicionario)
                    
                    if plot == 1:
                        for r in range(len(rules_levels)):
                            fig = plt.figure(figsize = (10, 5))
                            with plt.style.context('seaborn'):
                                plt.xlabel(f'{parameter} level')
                                plt.ylabel('quantidade de regras')
                                plt.plot(min_threshold, rules_levels[r], 'k--')
                                plt.plot(min_threshold, rules_levels[r], 'go')
                                plt.title(f'min_support = {min_supports[r]}\nEquivalente a {len(data.index)*min_supports[r]} vendas.')
                                st.pyplot(fig)
                    elif plot == 2:
                        fig2 = plt.figure(figsize = (10, 5))
                        for r in range(len(rules_levels)):
                            with plt.style.context('seaborn'):
                                plt.plot(min_threshold, rules_levels[r], label = f'min_support = {min_supports[r]}')
                        plt.xlabel(f'{parameter} level')
                        plt.ylabel('quantidade de regras')
                        plt.title('Comparação de todos os suportes adicionados.')
                        plt.legend()
                        st.pyplot(fig2)       
                    elif plot == 0:
                        pass
                    else:
                        print('O parametro plot não tem valor válido.')
                    
                    st.write(df_classe)

    if dataset_example:
        df = pd.read_csv("Groceries_dataset.csv")
        analise(df)

    if uploaded_file:
        dataset_example = False
        df = pd.read_csv(uploaded_file)
        analise(df)
