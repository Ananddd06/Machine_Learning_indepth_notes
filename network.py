import matplotlib.pyplot as plt
import networkx as nx

# Define categories and algorithms
categories = {
    "Supervised Learning": {
        "Regression": ["Linear Regression", "Ridge/Lasso/ElasticNet", "Decision Trees (Reg)", 
                       "Random Forest (Reg)", "Gradient Boosted Trees (Reg)", "SVR", "KNN (Reg)", "Neural Nets (Reg)"],
        "Classification": ["Perceptron", "Logistic Regression", "Softmax Regression", "SVM", "Naive Bayes", 
                           "LDA/QDA", "Decision Trees (Clf)", "Random Forest (Clf)", "Gradient Boosted Trees (Clf)", 
                           "KNN (Clf)", "Neural Nets (MLP, CNN, RNN, Transformers)"]
    },
    "Unsupervised Learning": {
        "Clustering": ["k-Means", "k-Medoids", "Hierarchical Clustering", "DBSCAN", "OPTICS", "Gaussian Mixture Models", "Spectral Clustering"],
        "Dimensionality Reduction": ["PCA", "SVD", "LDA", "t-SNE", "UMAP", "Autoencoders"],
        "Association": ["Apriori", "FP-Growth"]
    },
    "Semi-Supervised Learning": {
        "Methods": ["Self-training", "Label Propagation", "Semi-supervised Generative Models"]
    },
    "Reinforcement Learning": {
        "Value-based": ["Q-learning", "DQN"],
        "Policy-based": ["REINFORCE"],
        "Actor-Critic": ["A2C/A3C", "PPO", "DDPG", "TD3", "SAC"]
    },
    "Probabilistic Models": {
        "Methods": ["Gaussian Processes", "Hidden Markov Models", "Bayesian Networks", "Markov Random Fields"]
    },
    "Ensemble Methods": {
        "Methods": ["Bagging (Random Forests)", "Boosting (AdaBoost, XGBoost, LightGBM, CatBoost)", "Stacking"]
    }
}

# Build graph
G = nx.DiGraph()

for main_cat, subcats in categories.items():
    G.add_node(main_cat, type="main")
    for subcat, algos in subcats.items():
        G.add_node(subcat, type="sub")
        G.add_edge(main_cat, subcat)
        for algo in algos:
            G.add_node(algo, type="algo")
            G.add_edge(subcat, algo)

# Define node colors
colors = []
for node, data in G.nodes(data=True):
    if data["type"] == "main":
        colors.append("lightcoral")
    elif data["type"] == "sub":
        colors.append("skyblue")
    else:
        colors.append("lightgreen")

plt.figure(figsize=(20, 15))
pos = nx.spring_layout(G, k=0.35, seed=42)

nx.draw(G, pos, with_labels=True, node_color=colors, node_size=1500, font_size=8, font_weight="bold", edge_color="gray")
plt.title("Machine Learning Algorithms Family Tree", fontsize=16, fontweight="bold")
plt.show()
