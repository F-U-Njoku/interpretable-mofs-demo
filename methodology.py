import sys
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from scipy.stats import rankdata
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import four_objectives, six_objectives
from kneed import KneeLocator


def norm(X):
    return X / X.sum()


def entropy(X, k):
    return (X * np.log(X)).sum() * k


def rank_according_to(data, candidates):
    ranks = rankdata(data).astype(int)
    ranks -= 1
    zipped = list(zip(ranks, candidates))
    sorted_zipped = sorted(zipped, key=lambda x: x[0])
    _, sorted_list = zip(*sorted_zipped)

    return list(sorted_list)[::-1]


def fetch_shap(X, y, feat_list, classifier):
    model = classifier.fit(X.loc[:, feat_list], y)
    explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X.loc[:, feat_list], 1000))
    shap_values = explainer.shap_values(shap.sample(X.loc[:, feat_list], 1000))
    shap.summary_plot(shap_values, feature_names=feat_list, color=plt.get_cmap("tab10"), plot_size=(7, 5), show=False)
    plt.legend(ncol=1)
    plt.savefig("feature_contribution.png")


def main():
    if len(sys.argv) == 6:
        file_path, clf = six_objectives.experiment(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
        col = ['Subset size', ' Accuracy', ' F1-Score', " VIF", ' Statistical parity', ' Equalised odds']
        benefit_attributes = {1, 2, 4, 5}
    else:
        file_path, clf = four_objectives.experiment(sys.argv[1], sys.argv[2], sys.argv[3])
        col = ['Subset size', ' Accuracy', ' F1-Score', " VIF"]
        benefit_attributes = {1, 2}
    weight_method = sys.argv[-1]

    df_all = pd.read_csv(file_path)
    print(df_all.columns)
    df = df_all[col]
    m = df.shape[0]  # number of rows
    n = df.shape[1]  # number of columns
    candidates = np.array([i for i in range(1, m + 1)])

    # Get weight for objectives
    if weight_method == "equal":
        weights = [1 / len(col) for i in col]
    elif weight_method == "entropy":
        range_std = [(df[c].max() - df[c].min()) / df[c].std() for c in df.columns]
        weights = [i / sum(range_std) for i in range_std]

    elif weight_method == "rs":
        norm_df = df.apply(norm)
        k = -(1 / np.log(norm_df.shape[0]))
        ent = entropy(norm_df, k)
        dod = 1 - ent
        weights = dod / dod.sum()
        weights = weights.values
    else:
        print("Wrong weight method passed, we will use range/standard deviation.")
    df_weight = pd.DataFrame(weights, columns=["weight"])
    df_weight["rank"] = df_weight.weight.rank(ascending=False, method="min")
    df_weight["objective"] = col

    # TOPSIS ranking of solutions
    for c in col:
        df[c] = df[c].apply(abs)
        df[c] /= np.sqrt(sum(df[c] ** 2))
    raw_data = df * weights

    a_pos = np.zeros(n)
    a_neg = np.zeros(n)
    columns = ["$X_{%d}$" % j for j in range(n)]
    for j in range(n):
        column = raw_data.iloc[:, j]
        max_val = np.max(column)
        min_val = np.min(column)

        # See if we want to maximize benefit or minimize cost (for PIS)
        if j in benefit_attributes:
            a_pos[j] = max_val
            a_neg[j] = min_val
        else:
            a_pos[j] = min_val
            a_neg[j] = max_val

    pd.DataFrame(data=[a_pos, a_neg], index=["$A^*$", "$A^-$"], columns=columns)

    sp = np.zeros(m)
    sn = np.zeros(m)
    cs = np.zeros(m)

    for i in range(m):
        diff_pos = raw_data.iloc[i] - a_pos
        diff_neg = raw_data.iloc[i] - a_neg
        sp[i] = np.sqrt(diff_pos @ diff_pos)
        sn[i] = np.sqrt(diff_neg @ diff_neg)
        cs[i] = sn[i] / (sp[i] + sn[i])

    pd.DataFrame(data=zip(cs), index=candidates, columns=["$C^*$"])

    cs_order = rank_according_to(cs, candidates)

    # Get optimal number of clusters
    norm_df = normalize(df)
    sse = []
    k = range(1, min(11, m + 1))
    print(k)
    for num_clusters in k:
        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(norm_df)
        sse.append(kmeans.inertia_)
    kl = KneeLocator(k, sse, curve="convex", direction="increasing")
    cluster_nos = kl.elbow
    print(sse)

    # Clustering df solutions with K-Means to identify similar/outlier solutions

    kmeans = KMeans(random_state=0, n_init="auto", n_clusters=cluster_nos).fit(norm_df)
    cluster_id = list(kmeans.labels_)

    # Dimensionality reduction with PCA
    pca = PCA(n_components=2)
    low_d = pca.fit_transform(norm_df)
    comp1 = [i[0] for i in low_d]
    comp2 = [i[1] for i in low_d]
    sheet1 = pd.DataFrame(data=zip(cs_order, cluster_id, comp1, comp2), index=range(1, m + 1), columns=["rank",
                                                                                                        "cluster id",
                                                                                                        "Component 1",
                                                                                                        "Component 2"])
    print(sheet1)
    sheet1["solution"] = df_all[' Solution'].values
    sheet1 = (pd.concat([sheet1.reset_index(drop=True), df_all[col].apply(abs).reset_index(drop=True)], axis=1))

    # Individual feature frequency
    features = []
    print(df_all.head())
    for i in df_all[' Solution'].str.split(';'):
        features += i
        features = [i.strip() for i in features]
        df_freq = pd.DataFrame.from_dict(data=Counter(sorted(features)), orient='index', columns=["frequency"])
        df_freq.index.name = 'feature'
    print(df_freq)

    # Individual feature contribution with SHAP
    main_df = pd.read_csv(sys.argv[1])
    X = main_df.drop(columns=sys.argv[2])
    y = main_df[sys.argv[2]]
    feat_list = df_freq.index.tolist()
    fetch_shap(X, y, feat_list, clf)

    with pd.ExcelWriter("results.xlsx") as writer:

        # use to_excel function and specify the sheet_name and index
        # to store the dataframe in specified sheet
        sheet1.to_excel(writer, sheet_name="main", index=False)
        df_freq.to_excel(writer, sheet_name="frequency", index=True)
        df_weight.to_excel(writer, sheet_name="obj_rank", index=False)


if __name__ == '__main__':
    main()

s = pd.Series(range(5), index=list("abcde"))

s["d"] = s["b"]

s.rank()
