# Scatter Plot Bidimensional
feature_combinations = [("u", "g"), ("u", "r"), ("g", "r"), ("r", "i"), ("i", "z")]
for features in feature_combinations:
    fig_scatter = go.Figure(data=[go.Scatter(x=data[features[0]], y=data[features[1]], mode='markers',
 marker=dict(color=labels_kmeans, colorscale='viridis'))])
    fig_scatter.update_layout(title=f"Scatter Plot - {features[0]} vs {features[1]}")
    st.plotly_chart(fig_scatter)

# Heatmap de Correlação
corr_matrix = data.drop("class", axis=1).corr()
fig_heatmap = go.Figure(data=go.Heatmap(z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.columns,
                                        colorscale='Viridis'))
fig_heatmap.update_layout(title="Heatmap de Correlação")
st.plotly_chart(fig_heatmap)


