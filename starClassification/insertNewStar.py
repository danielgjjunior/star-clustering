import numpy as np #numerico
import plotly.graph_objects as go #graficos
import pandas as pd #csv
from sklearn.cluster import KMeans #modelo
from sklearn_extra.cluster import KMedoids #extra
from sklearn.preprocessing import StandardScaler #scaler
import streamlit as st #
import plotly.express as px #express


data = pd.read_csv("baseDef.csv")




excluirColunas = ["obj_ID", "run_ID", "rerun_ID", "cam_col", "field_ID", "spec_obj_ID", "plate", "MJD", "fiber_ID"]


data = data.drop(excluirColunas, axis=1)

st.title("Inserir uma nova estrela")

#Converter os valores de string para number
converterClass = {"STAR": 1, "QSO": 2, "GALAXY": 3}
data["class"] = data["class"].map(converterClass)


# Configurar o número de clusters
k = st.sidebar.slider("Número de Clusters (k)", 2, 10, 3)

# Configurar a quantidade de dados a serem exibidos
max_dados = st.sidebar.slider("Quantidade de Dados", 100, len(data), 1000)

# Escolher o algoritmo de clustering
algoritmo = st.sidebar.selectbox("Algoritmo de Clustering", ["K-means", "K-medoids"])

st.image("./imagem/imagem3.jpg", width=600, caption="Imagem 3")


# Amostrar os dados
baseAmostrada = data.sample(max_dados, random_state=42)

# Separar as características (features) dos dados amostrados
X = baseAmostrada.values

# Padronizar as características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Realizar o agrupamento
if algoritmo == "K-means":
    modeloCluster = KMeans(n_clusters=k, random_state=42)
else:
    modeloCluster = KMedoids(n_clusters=k, random_state=42)

modeloCluster.fit(X_scaled)
labels = modeloCluster.labels_

# Restaurar os valores originais da coluna "class" para mostrar no gráfico
data["class"] = data["class"].map({1: "STAR", 2: "QSO", 3: "GALAXY"})

#Nomes das características do baseAmostrada
recursosBases = baseAmostrada.columns







#Gráfico 3D ao app
fig_3d = go.Figure(data=[go.Scatter3d(x=X_scaled[:, 0], y=X_scaled[:, 1], z=X_scaled[:, 2],
                                      mode='markers', marker=dict(color=labels, colorscale='viridis'), showlegend=False)])
fig_3d.update_layout(scene=dict(xaxis_title=recursosBases[0], yaxis_title=recursosBases[1], zaxis_title=recursosBases[2]),
title=f'Visualização 3D dos Dados de Classificação de Estrelas - {algoritmo} (k={k})',
scene_aspectmode='cube')

st.plotly_chart(fig_3d)

# Gráfico de contagem de estrelas em cada cluster
cluster_counts = pd.Series(labels).value_counts().sort_index()
fig_bar = px.bar(cluster_counts, x=cluster_counts.index, y=cluster_counts.values)
fig_bar.update_layout(xaxis_title="Cluster", yaxis_title="Contagem", title="Contagem de Estrelas em Cada Cluster")
st.plotly_chart(fig_bar)

# Dados com suas classes e rótulos de cluster
baseAmostrada["cluster"] = labels
st.write("Dados Amostrados com Classes e Rótulos de Cluster")
st.write("<p> Class: STAR: 1, QSO: 2, GALAXY: 3</p>", unsafe_allow_html=True)
st.write(baseAmostrada)

# nova estrela
st.header("Inserir nova Estrela")
new_star_df = pd.DataFrame(index=[0], columns=recursosBases)
for feature in recursosBases:
    new_star_df[feature] = st.number_input(f"{feature}:", value=0.0)

# Padronizar os valores da nova estrela
new_star_scaled = scaler.transform(new_star_df.values)

# Prever o cluster da nova estrela
new_star_cluster = modeloCluster.predict(new_star_scaled)[0]


st.write("<h3>Cluster Previsto para a Nova Estrela:</h3>", unsafe_allow_html=True)
st.markdown(f"<h1 style='font-size: 80px; text-align: center; color:green'>{new_star_cluster}</h1>", unsafe_allow_html=True)



# Calcular a média das características dos clusters
cluster_means = []
for i in range(k):
    cluster_points = X_scaled[labels == i]
    cluster_mean = np.mean(cluster_points, axis=0)
    cluster_means.append(cluster_mean)




# Calcular a média das características da nova estrela
new_star_mean = np.mean(new_star_scaled, axis=0)


st.write("Distância Euclidiana entre a Estrela e a Média dos Clusters")
for i, cluster_mean in enumerate(cluster_means):
    distance = np.linalg.norm(new_star_mean - cluster_mean)
    st.write(f"Cluster {i}: {distance}")




#Plotar a nova estrela no gráfico

new_star_coordinates = new_star_scaled[0]
X_with_new_star = np.concatenate((X_scaled, [new_star_coordinates]))

labels_with_new_star = np.concatenate((labels, [new_star_cluster]))

# Exibir 3D novamente com a estrela
fig_3d_with_new_star = go.Figure(data=[go.Scatter3d(x=X_with_new_star[:, 0], y=X_with_new_star[:, 1], z=X_with_new_star[:, 2],
mode='markers', marker=dict(color=labels_with_new_star, colorscale='viridis'), showlegend=False)])

# Cone - para nova estrela kkkkkk
fig_3d_with_new_star.add_trace(go.Cone(x=[new_star_coordinates[0]], y=[new_star_coordinates[1]], z=[new_star_coordinates[2]],
u=[0], v=[0], w=[1], sizemode="scaled", sizeref=0.1, showscale=False, colorscale='electric'))

fig_3d_with_new_star.update_layout(scene=dict(xaxis_title=recursosBases[0], yaxis_title=recursosBases[1], zaxis_title=recursosBases[2]),
title=f'Visualização 3D dos Dados de Classificação de Estrelas - {algoritmo} (k={k})',
scene_aspectmode='cube')

st.plotly_chart(fig_3d_with_new_star)



X_2d = X_scaled[:, :2]  # duas primeiras colunas (alfa,beta)

#Gráfico 2D
fig_2d = go.Figure(data=[go.Scatter(x=X_2d[:, 0], y=X_2d[:, 1], mode='markers', marker=dict(color=labels, colorscale='viridis'))])
fig_2d.update_layout(xaxis_title=recursosBases[0], yaxis_title=recursosBases[1], title=f'Visualização 2D dos Dados de Classificação de Estrelas - {algoritmo} (k={k})')

# Nova estrela
fig_2d.add_trace(go.Scatter(x=[new_star_scaled[0, 0]], y=[new_star_scaled[0, 1]], mode='markers',
marker=dict(color=new_star_cluster, size=10, symbol='star', line=dict(color='black', width=2)), name='Nova Estrela'))
st.plotly_chart(fig_2d)


st.write("Tabela de Dados Completa")
st.write(data)
