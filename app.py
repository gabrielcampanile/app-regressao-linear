import streamlit as st
import numpy as np
import pandas as pd
from pages.sem_repeticao import page_no_repetition
from pages.com_repeticao import page_with_repetition

def main():
    st.set_page_config(page_title="Visualização ANOVA", layout="wide")
    
    # Seleção da página
    page = st.sidebar.radio("Selecione o tipo de regressão:", 
                           ["Sem Repetição", "Com Repetição"])
    
    if page == "Sem Repetição":
        page_no_repetition()
    else:
        page_with_repetition()

if __name__ == '__main__':
    main()