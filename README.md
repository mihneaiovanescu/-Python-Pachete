# Analiza Activității și Extinderii RetailPlus SA

Acest proiect este o aplicație web interactivă construită cu **Streamlit** (Python). Aplicația analizează performanța unei rețele de retail și simulează oportunități de extindere folosind tehnici de data science.

## Funcționalități incluse:
1. **Preprocesare și EDA (Pandas):** Curățarea datelor (valori lipsă, outliers) și agregarea vânzărilor.
2. **Analiză Spațială (Geopandas):** Vizualizarea pe coordonate a magazinelor existente.
3. **Clusterizare (Scikit-Learn):** Segmentarea magazinelor folosind algoritmul K-Means în funcție de trafic și vânzări.
4. **Predicție (Statsmodels):** Model de regresie multiplă pentru estimarea veniturilor unui nou magazin pe baza bugetului de marketing și a concurenței.

## Cum să rulezi proiectul local
1. Asigură-te că ai instalat Python.
2. Instalează pachetele necesare: `pip install -r requirements.txt`
3. Rulează aplicația: `streamlit run app.py`
