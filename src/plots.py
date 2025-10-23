import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from scipy.stats import gaussian_kde
from prepare_data import prepare_df


def histogram(df, num_bins: int | None = 8):
    exp_scores = df['cExp']
    no_scores = df['cNo']

    bin_min = np.floor(min(exp_scores.min(), no_scores.min()))
    bin_max = np.ceil(max(exp_scores.max(), no_scores.max()))
    num_bins = num_bins if num_bins else int(bin_max - bin_min)
    bins = np.linspace(bin_min, bin_max, num_bins)

    width = ((bins[1] - bins[0]) * 0.4).astype(float)

    # --- Get counts (not densities) ---
    exp_counts, _ = np.histogram(exp_scores, bins=bins, density=False)
    no_counts, _ = np.histogram(no_scores, bins=bins, density=False)

    bin_centers = (bins[:-1] + bins[1:]) / 2

    plt.figure(figsize=(8, 5))
    plt.bar(bin_centers - width / 2, exp_counts, width=width, label='Explicit Confirmation', alpha=0.7, edgecolor='black')
    plt.bar(bin_centers + width / 2, no_counts, width=width, label='No Confirmation', alpha=0.7, edgecolor='black')

    # --- Adjust KDEs to counts (scale density by total count and bin width) ---
    x_vals = np.linspace(min(bins), max(bins), 200)
    kde_exp = gaussian_kde(exp_scores)
    kde_no = gaussian_kde(no_scores)

    bin_width = bins[1] - bins[0]
    plt.plot(x_vals, kde_exp(x_vals) * len(exp_scores) * bin_width, color='blue', lw=2)
    plt.plot(x_vals, kde_no(x_vals) * len(no_scores) * bin_width, color='darkorange', lw=2)

    # Bin labels
    range_labels = [
        f"[{bins[i]:.1f}–{bins[i + 1]:.1f})" if i < len(bins) - 2 else f"[{bins[i]:.1f}–{bins[i + 1]:.1f}]"
        for i in range(len(bins) - 1)
    ]
    plt.xticks(bin_centers, range_labels, rotation=45)

    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))

    plt.xlabel('Satisfaction Score')
    plt.ylabel('Participant Count')
    plt.title('Distribution of Satisfaction Scores by Condition')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    processed_df, cond_df = prepare_df()
    print(cond_df.to_string())
    histogram(cond_df, 8)
