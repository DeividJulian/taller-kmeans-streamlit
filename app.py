"""
Taller Final Minería de Datos – Parte 7
Despliegue con Streamlit: segmentación de clientes con K-Means (Mall Customers)

Ejecutar:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Segmentación de clientes K-Means", page_icon="🛍️", layout="wide")
sns.set_theme(style="whitegrid")

FEATURES = ["Annual Income (k$)", "Spending Score (1-100)"]
RANDOM_STATE = 42

ESTRATEGIAS = {
    "Clientes de alto valor": "Programa VIP de fidelización, acceso anticipado a lanzamientos, productos premium y atención personalizada.",
    "Clientes promedio": "Promociones de temporada, combos, venta cruzada y tarjeta de puntos para aumentar su gasto gradualmente.",
    "Clientes de bajo valor": "Descuentos en productos básicos, marcas propias y comunicación de bajo costo.",
    "Clientes con potencial": "Encuestas para entender por qué no compran, campañas de calidad y experiencias exclusivas, incentivos de primera compra.",
    "Clientes impulsivos (sensibles al precio)": "Promociones flash, cupones, pagos a cuotas y marketing en redes sociales.",
}


# ---------------------------------------------------------------- datos y modelo
@st.cache_data
def cargar_datos(archivo=None):
    df = pd.read_csv(archivo) if archivo is not None else pd.read_csv("Mall_Customers.csv")
    return df.rename(columns={"Genre": "Gender"})


@st.cache_data
def curvas_evaluacion(X_scaled):
    inercias, siluetas = [], []
    for k in range(1, 11):
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=RANDOM_STATE).fit(X_scaled)
        inercias.append(km.inertia_)
        siluetas.append(silhouette_score(X_scaled, km.labels_) if k > 1 else np.nan)
    return inercias, siluetas


@st.cache_resource
def entrenar(X, k):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    modelo = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=RANDOM_STATE).fit(X_scaled)
    return scaler, modelo


def asignar_perfil(ingreso, gasto, df):
    ing_m, ing_s = df[FEATURES[0]].mean(), df[FEATURES[0]].std()
    gas_m, gas_s = df[FEATURES[1]].mean(), df[FEATURES[1]].std()
    ing = "alto" if ingreso > ing_m + 0.5 * ing_s else "bajo" if ingreso < ing_m - 0.5 * ing_s else "medio"
    gas = "alto" if gasto > gas_m + 0.5 * gas_s else "bajo" if gasto < gas_m - 0.5 * gas_s else "medio"
    if ing == "alto" and gas == "alto":
        return "Clientes de alto valor"
    if ing == "alto" and gas == "bajo":
        return "Clientes con potencial"
    if ing == "bajo" and gas == "bajo":
        return "Clientes de bajo valor"
    if ing == "bajo" and gas == "alto":
        return "Clientes impulsivos (sensibles al precio)"
    return "Clientes promedio"


# ---------------------------------------------------------------- barra lateral
st.sidebar.title("⚙️ Configuración")
archivo = st.sidebar.file_uploader("Dataset (opcional)", type="csv",
                                   help="Si no subes nada se usa Mall_Customers.csv de la carpeta.")
try:
    df = cargar_datos(archivo)
except FileNotFoundError:
    st.error("No se encontró **Mall_Customers.csv**. Colócalo junto a app.py o súbelo en la barra lateral.")
    st.stop()

faltantes = [c for c in FEATURES if c not in df.columns]
if faltantes:
    st.error(f"El archivo no tiene las columnas requeridas: {faltantes}")
    st.stop()

k = st.sidebar.slider("Número de clusters (K)", 2, 10, 5,
                      help="K = 5 es el óptimo según el codo y la silueta.")

X = df[FEATURES].values
scaler, modelo = entrenar(X, k)
X_scaled = scaler.transform(X)
df["Cluster"] = modelo.labels_
centroides = pd.DataFrame(scaler.inverse_transform(modelo.cluster_centers_), columns=FEATURES)
perfiles = {i: asignar_perfil(r.iloc[0], r.iloc[1], df) for i, r in centroides.iterrows()}
df["Perfil"] = df["Cluster"].map(perfiles)

st.sidebar.metric("Clientes", len(df))
st.sidebar.metric("Silueta del modelo", f"{silhouette_score(X_scaled, modelo.labels_):.3f}")

# ---------------------------------------------------------------- encabezado
st.title("🛍️ Segmentación de clientes con K-Means")
st.caption("Taller Final de Minería de Datos · Mall Customers Dataset · Variables: ingreso anual y puntaje de gasto")

tab_codo, tab_clusters, tab_nuevo = st.tabs(
    ["📉 Método del codo", "🎯 Clusters generados", "🧑 Clasificar nuevo cliente"]
)

# ---------------------------------------------------------------- 1. codo
with tab_codo:
    inercias, siluetas = curvas_evaluacion(X_scaled)
    ks = list(range(1, 11))
    c1, c2 = st.columns(2)

    with c1:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(ks, inercias, "o-", lw=2)
        ax.axvline(k, color="red", ls="--", label=f"K seleccionado = {k}")
        ax.set(xlabel="Número de clusters (K)", ylabel="Inercia (WCSS)", title="Método del codo", xticks=ks)
        ax.legend()
        st.pyplot(fig)

    with c2:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(ks[1:], siluetas[1:], "o-", color="green", lw=2)
        ax.axvline(k, color="red", ls="--", label=f"K seleccionado = {k}")
        ax.set(xlabel="Número de clusters (K)", ylabel="Silueta", title="Coeficiente de silueta", xticks=ks[1:])
        ax.legend()
        st.pyplot(fig)

    k_best = ks[1:][int(np.nanargmax(siluetas[1:]))]
    st.info(f"El codo se forma en **K = 5** y la silueta máxima se obtiene en **K = {k_best}** "
            f"({np.nanmax(siluetas):.3f}). Ambos métodos coinciden.")
    st.dataframe(pd.DataFrame({"K": ks, "Inercia": np.round(inercias, 2), "Silueta": np.round(siluetas, 4)}),
                 hide_index=True, width="stretch")

# ---------------------------------------------------------------- 2. clusters
with tab_clusters:
    c1, c2 = st.columns([3, 2])
    with c1:
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.scatterplot(data=df, x=FEATURES[0], y=FEATURES[1], hue="Perfil", style="Cluster",
                        palette="Set2", s=70, ax=ax)
        ax.scatter(centroides[FEATURES[0]], centroides[FEATURES[1]], c="black", s=250, marker="X", label="Centroides")
        ax.set_title(f"Clusters de clientes (K = {k})")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        st.pyplot(fig)
    with c2:
        st.subheader("Centroides y perfiles")
        resumen = df.groupby("Cluster").agg(
            Clientes=(FEATURES[0], "count"),
            Edad=("Age", "mean"),
            Ingreso=(FEATURES[0], "mean"),
            Gasto=(FEATURES[1], "mean"),
        ).round(1)
        resumen.insert(0, "Perfil", resumen.index.map(perfiles))
        st.dataframe(resumen, width="stretch")

    st.subheader("Datos con su cluster asignado")
    st.dataframe(df, width="stretch", height=300)
    st.download_button("⬇️ Descargar CSV con clusters", df.to_csv(index=False).encode("utf-8"),
                       "clientes_segmentados.csv", "text/csv")

# ---------------------------------------------------------------- 3 y 4. nuevo cliente
with tab_nuevo:
    st.subheader("Ingresa los datos del nuevo cliente")
    with st.form("nuevo_cliente"):
        c1, c2, c3 = st.columns(3)
        genero = c1.selectbox("Género", ["Female", "Male"])
        edad = c2.number_input("Edad", 18, 100, 30)
        ingreso = c3.number_input("Ingreso anual (k$)", 0, 300, 60)
        gasto = st.slider("Puntaje de gasto (1-100)", 1, 100, 50)
        enviar = st.form_submit_button("Clasificar cliente", type="primary")

    if enviar:
        nuevo_esc = scaler.transform([[ingreso, gasto]])
        cluster = int(modelo.predict(nuevo_esc)[0])
        perfil = perfiles[cluster]
        distancias = np.linalg.norm(modelo.cluster_centers_ - nuevo_esc, axis=1)

        st.success(f"El cliente pertenece al **Cluster {cluster}**: **{perfil}**")
        m1, m2, m3 = st.columns(3)
        m1.metric("Cluster asignado", cluster)
        m2.metric("Clientes en este cluster", int((df["Cluster"] == cluster).sum()))
        m3.metric("Distancia al centroide", f"{distancias[cluster]:.3f}")
        st.markdown(f"**Estrategia sugerida:** {ESTRATEGIAS[perfil]}")

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(data=df, x=FEATURES[0], y=FEATURES[1], hue="Perfil", palette="Set2",
                        s=50, alpha=0.6, ax=ax)
        ax.scatter(centroides[FEATURES[0]], centroides[FEATURES[1]], c="black", s=200, marker="X", label="Centroides")
        ax.scatter(ingreso, gasto, c="red", s=400, marker="*", edgecolor="black", label="Nuevo cliente")
        ax.set_title("Ubicación del nuevo cliente")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        st.pyplot(fig)
