---

![AIC/BIC Plot](../../images/bic_aic_gmm.png)

Although both AIC and BIC identified **K = 2** as the statistically optimal number of Gaussian components, we elected to use **K = 4** in our final model.
This decision is based on practical considerations rather than purely on information-theoretic criteria:

1. **Domain interpretability** – The dataset describes musical tracks, and four clusters give a more meaningful segmentation (e.g., energetic tracks, mellow tracks, acoustic pieces, and experimental pieces). Two clusters were too coarse to reflect these distinctions.

2. **User-facing clarity** – Presenting results as four groups provides a better balance between granularity and ease of understanding for end-users of the music system.

3. **Marginal likelihood gain** – While AIC/BIC penalize additional parameters, the log-likelihood continues to improve slightly beyond K = 2. Given our application, we value the richer representation even at the cost of a modest information-criterion penalty.

4. **Exploratory purpose** – Our clustering serves as an exploratory tool rather than a definitive generative model; a slightly larger K allows us to capture subtler stylistic nuances.

> In summary, we acknowledge that AIC and BIC favor a simpler model, but **K = 4** better aligns with the goals of musical characterization and user experience.

---
