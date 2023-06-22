import numpy as np
import plotly.graph_objects as go
import pandas as pd
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.preprocessing import StandardScaler
import streamlit as st
import plotly.express as px


data = pd.read_csv("baseDef.csv")


cols_to_drop = ["obj_ID", "run_ID", "rerun_ID", "cam_col", "field_ID", "spec_obj_ID", "plate", "MJD", "fiber_ID"]
data = data.drop(cols_to_drop, axis=1)


st.title("Visualização dos Dados de Classificação de Estrelas")

category_mapping = {"STAR": 1, "QSO": 2, "GALAXY": 3}
data["class"] = data["class"].map(category_mapping)

k = st.sidebar.slider("Número de Clusters (k)", 2, 10, 3)


max_samples = st.sidebar.slider("Quantidade de Dados", 100, len(data), 1000)

algorithm = st.sidebar.selectbox("Algoritmo de Clustering", ["K-means", "K-medoids"])

st.image("./imagem/imagem1.png", width=600, caption="Imagem 1")
st.image("./imagem/imagem2.png", width=600, caption="Imagem 2")

sampled_data = data.sample(max_samples, random_state=42)

X = sampled_data.values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

if algorithm == "K-means":
    cluster_model = KMeans(n_clusters=k, random_state=42)
else:
    cluster_model = KMedoids(n_clusters=k, random_state=42)

cluster_model.fit(X_scaled)
labels = cluster_model.labels_

feature_names = sampled_data.columns
print(feature_names)



# 3D
fig_3d = go.Figure(data=[go.Scatter3d(x=X_scaled[:, 0], y=X_scaled[:, 1], z=X_scaled[:, 2],
mode='markers', marker=dict(color=labels, colorscale='viridis'), showlegend=False)])
fig_3d.update_layout(scene=dict(xaxis_title=feature_names[0], yaxis_title=feature_names[1], zaxis_title=feature_names[2]),
title=f'Visualização 3D dos Dados de Classificação de Estrelas - {algorithm} (k={k})',scene_aspectmode='cube')
st.plotly_chart(fig_3d)

# Contagem de estrelas em cada cluster
cluster_counts = pd.Series(labels).value_counts().reset_index()
cluster_counts.columns = ["Cluster", "Count"]
fig_bar = go.Figure(data=[go.Bar(x=cluster_counts["Cluster"], y=cluster_counts["Count"])])
fig_bar.update_layout(title="Contagem de Estrelas em cada Cluster")
st.plotly_chart(fig_bar)

#Histograma
for feature in sampled_data.columns[:-1]:
    fig_hist = go.Figure(data=[go.Histogram(x=sampled_data[feature], marker=dict(color="steelblue"))])
    fig_hist.update_layout(title=f"Histograma - {feature}")
    st.plotly_chart(fig_hist)

# Box Plot
for feature in sampled_data.columns:
    fig_box = go.Figure()
    for i in range(k):
        fig_box.add_trace(go.Box(x=sampled_data.loc[labels == i, feature], name=f"Cluster {i}"))
    fig_box.update_layout(title=f"Box Plot - {feature} por Cluster")
    st.plotly_chart(fig_box)

# Heatmap de Correlação
corr_matrix = data.drop("class", axis=1).corr()
fig_heatmap = go.Figure(data=go.Heatmap(z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.columns,
                                        colorscale='Viridis'))
fig_heatmap.update_layout(title="Heatmap de Correlação")
st.plotly_chart(fig_heatmap)

# Gráfico de dispersão 
fig_scatter = go.Figure()
fig_scatter.add_trace(go.Scatter(x=X_scaled[:, 0], y=X_scaled[:, 1], mode='markers', marker=dict(color=labels, colorscale='viridis')))
fig_scatter.update_layout(title=f'Gráfico de Dispersão das Duas Primeiras Características - {algorithm} Clustering')
st.plotly_chart(fig_scatter)


cluster_data = pd.DataFrame(X_scaled, columns=sampled_data.columns)
cluster_data["Cluster"] = labels

cluster_means = cluster_data.groupby("Cluster").mean()



# Definir uma paleta de cores personalizada
cores = px.colors.sequential.Turbo[:k]



# Gráfico de radar 
fig_radar = go.Figure()
for i in range(k):
    fig_radar.add_trace(go.Scatterpolar(
        r=cluster_means.loc[i].values,
        theta=sampled_data.columns[:-1],
        fill='toself',
        fillcolor=cores[i],
        name=f"Cluster {i}"
    ))
fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True)),
title="Gráfico de Radar - Médias das Características por Cluster")
st.plotly_chart(fig_radar)


st.subheader("Tabela de Dados")
st.write("<p> Class: STAR: 1, QSO: 2, GALAXY: 3</p>", unsafe_allow_html=True)

st.dataframe(sampled_data)


st.subheader("Informações sobre os Clusters")
cluster_info = pd.DataFrame({"Cluster": labels})
cluster_counts = cluster_info["Cluster"].value_counts().reset_index()
cluster_counts.columns = ["Cluster", "Count"]
st.dataframe(cluster_counts)


st.subheader("Informações de cada Cluster")
st.dataframe(cluster_data)

