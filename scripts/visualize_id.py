import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

data_path = "/Users/aramis/Heap/computed_embeddings/intrinsic_dimensionality.npz"
data = np.load(data_path, allow_pickle=True)

gains = data["perturbation_strength"]  # shape (num_perturbations,)
id_chart = data["id_chart"]  # shape (num_perturbations, num_samples)
num_perturbations, num_instances = id_chart.shape


mean_id = id_chart.mean(axis=1)

perturbation_strengths = np.repeat(gains, num_instances)
df = pd.DataFrame(
    {
        "perturbation_strength": perturbation_strengths,
        "estimated_id": id_chart.flatten(),
    }
)
# df = pd.DataFrame({"perturbation_strength": gains, "mean_id": mean_id})
sns.lineplot(
    data=df,
    x="perturbation_strength",
    y="estimated_id",
)
plt.show()

# fig, ax = plt.subplots()
# ax.plot(gains, mean_id)
# ax.set_xlabel("Perturbation range ($\\pm$ dB)")
# ax.set_ylabel("Mean Intrinsic Dimensionality")

# plt.show()

# num_bins =
# bin_min, bin_max = np.quantile(id_chart.flatten(), [0.01, 0.99])

# # bins = np.linspace(bin_min, bin_max, num_bins + 1, endpoint=True)o
# bins = np.arange(id_chart.max() + 1)

# histogram_image = np.zeros((id_chart.shape[0], len(bins) - 1))
# for i in range(id_chart.shape[0]):
#     bin_memberships, _ = np.histogram(id_chart[i], bins=bins)
#     histogram_image[i] = bin_memberships

# fig, ax = plt.subplots()
# im = ax.imshow(
#     histogram_image,
#     aspect="auto",
#     origin="lower",
#     extent=[bin_min, bin_max, gains[0], gains[-1]],
#     cmap="viridis",
#     norm=mpl.colors.LogNorm(vmin=1, vmax=histogram_image.max()),
# )

# cbar = fig.colorbar(im, ax=ax)

# ax.set_xlabel("lPCA estimated ID")
# ax.set_ylabel("Perturbation range ($\\pm$ dB)")

# plt.show()
