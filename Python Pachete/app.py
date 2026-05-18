import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import statsmodels.api as sm


st.set_page_config(page_title="Analiză RetailPlus SA", layout="wide")
st.title("Analiza Activității și Extinderii RetailPlus SA")



@st.cache_data
def load_data():

    np.random.seed(42)
    data = pd.DataFrame({
        'ID_Magazin': range(1, 101),
        'Regiune': np.random.choice(['București', 'Moldova', 'Transilvania', 'Banat', 'Dobrogea'], 100),
        'Vanzari': np.random.normal(500000, 100000, 100),
        'Trafic_Clienti': np.random.normal(5000, 1000, 100),
        'Suprafata_mp': np.random.normal(200, 50, 100),
        'Buget_Marketing': np.random.normal(10000, 2000, 100),
        'Distanta_Concurenta_km': np.random.exponential(5, 100),
        'Tip_Locatie': np.random.choice(['Mall', 'Stradal'], 100),
        'Latitudine': np.random.uniform(43.5, 48.0, 100),
        'Longitudine': np.random.uniform(20.0, 29.0, 100)
    })

    data.loc[0:5, 'Vanzari'] = np.nan
    data.loc[10:15, 'Trafic_Clienti'] = np.nan
    return data


df_brut = load_data()


tab1, tab2, tab3, tab4 = st.tabs(
    ["1. Preprocesare & EDA", "2. Analiză Spațială", "3. Clusterizare", "4. Predicție Regresie"])


with tab1:
    st.header("Tratarea datelor și Agregare")


    df_clean = df_brut.copy()
    df_clean['Vanzari'] = df_clean['Vanzari'].fillna(df_clean['Vanzari'].mean())
    df_clean['Trafic_Clienti'] = df_clean['Trafic_Clienti'].fillna(df_clean['Trafic_Clienti'].median())


    Q1 = df_clean['Vanzari'].quantile(0.25)
    Q3 = df_clean['Vanzari'].quantile(0.75)
    IQR = Q3 - Q1
    df_clean = df_clean[(df_clean['Vanzari'] >= Q1 - 1.5 * IQR) & (df_clean['Vanzari'] <= Q3 + 1.5 * IQR)]

    st.subheader("Baza de date curățată (Primele 10 rânduri)")
    st.dataframe(df_clean.head(10))


    st.subheader("Vânzări Totale pe Regiuni")
    df_grouped = df_clean.groupby('Regiune')['Vanzari'].sum().reset_index()
    st.bar_chart(df_grouped.set_index('Regiune'))


with tab2:
    st.header("Analiza Spațială a Magazinelor")


    geometry = [Point(xy) for xy in zip(df_clean['Longitudine'], df_clean['Latitudine'])]
    gdf = gpd.GeoDataFrame(df_clean, geometry=geometry, crs="EPSG:4326")

    st.write("Distribuția spațială a magazinelor existente pe coordonate (Long/Lat).")


    fig, ax = plt.subplots(figsize=(8, 6))

    gdf.plot(ax=ax, markersize=gdf['Vanzari'] / 10000, alpha=0.6, color='red', edgecolor='k')
    plt.title("Amplasamentul magazinelor (mărimea indică volumul vânzărilor)")
    plt.xlabel("Longitudine")
    plt.ylabel("Latitudine")
    st.pyplot(fig)


with tab3:
    st.header("Segmentarea Magazinelor (K-Means)")

    df_cluster = df_clean[['Suprafata_mp', 'Vanzari', 'Trafic_Clienti', 'Tip_Locatie']].copy()
    df_cluster = pd.get_dummies(df_cluster, columns=['Tip_Locatie'], drop_first=True)


    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_cluster)


    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df_clean['Cluster'] = kmeans.fit_predict(scaled_features)

    st.write("Magazinele au fost împărțite în 3 clustere bazate pe performanță și dimensiune.")

    fig, ax = plt.subplots()
    scatter = ax.scatter(df_clean['Trafic_Clienti'], df_clean['Vanzari'], c=df_clean['Cluster'], cmap='viridis')
    plt.xlabel("Trafic Clienți")
    plt.ylabel("Vânzări")
    plt.title("Clustere: Vânzări vs. Trafic")
    plt.colorbar(scatter, label="Cluster")
    st.pyplot(fig)


with tab4:
    st.header("Predicția Vânzărilor")


    X = df_clean[['Trafic_Clienti', 'Buget_Marketing', 'Distanta_Concurenta_km']]
    X = sm.add_constant(X)  # Adăugare intercept
    y = df_clean['Vanzari']


    model = sm.OLS(y, X).fit()

    st.subheader("Rezumatul Modelului Statistic")
    st.text(model.summary().tables[1])

    st.subheader("Simulator de Vânzări (Oportunitate de extindere)")
    trafic_estimat = st.number_input("Trafic Clienți Estimat", value=5000)
    buget_mkt = st.number_input("Buget Marketing (RON)", value=10000)
    dist_concurenta = st.number_input("Distanța față de concurență (km)", value=5.0)

    if st.button("Calculează Venit Estimat"):

        predictie = model.params['const'] + (model.params['Trafic_Clienti'] * trafic_estimat) + \
                    (model.params['Buget_Marketing'] * buget_mkt) + \
                    (model.params['Distanta_Concurenta_km'] * dist_concurenta)
        st.success(f"Veniturile estimate pentru noul magazin sunt: {predictie:,.2f} RON")