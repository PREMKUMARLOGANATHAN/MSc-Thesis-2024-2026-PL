# MSc-Thesis-2024-2026-PL

Github repository for the codes corresponding to the master thesis on 'Investigating Climate-Driven Causal Relationships in Antimicrobial Use and Resistance in Animals'
All the code is writen by the author **Prem Kumar Loganathan** (@PREMKUMARLOGANATHAN) (email: premkumar.loganathan@student.kuleuven.be)

## What is it?
- [CCM.py](https://github.com/PREMKUMARLOGANATHAN/MSc-Thesis-2024-2026-PL/blob/main/CCM.py) – Testing Causal Relationship using Convergent Cross Mapping (CCM) (Sugihara et al., 2012)
- [afn.py](https://github.com/PREMKUMARLOGANATHAN/MSc-Thesis-2024-2026-PL/blob/main/afn.py) – Choosing Optimal Embedding Dimension (E) using Cao's FNN method (Cao, 1997)
- [climate_variables_preprocessing.py](https://github.com/PREMKUMARLOGANATHAN/MSc-Thesis-2024-2026-PL/blob/main/environmental_variables_preprocessing.py) - Converting environmental variables to the same temporal resolution as AntiMicrobial Usage (AMU) and AntiMicrobial Resistance (AMR) Dataset
- [ccm_result_1.py](https://github.com/PREMKUMARLOGANATHAN/MSc-Thesis-2024-2026-PL/blob/main/ccm_result_1.py) – Used to plot the CCM results (without bootstrapping)
- [ccm_result_with_bootstrapping.py](https://github.com/PREMKUMARLOGANATHAN/MSc-Thesis-2024-2026-PL/blob/main/ccm_result_with_bootstrapping.py) – Used to plot the CCM results (with bootstrapping)
- [manifold_visualisation.py](https://github.com/PREMKUMARLOGANATHAN/MSc-Thesis-2024-2026-PL/blob/main/manifold_visualisation.py) – Visualise the reconstructed shadow manifold
- [preprocessing.py](https://github.com/PREMKUMARLOGANATHAN/MSc-Thesis-2024-2026-PL/blob/main/preprocessing.py) – Normalise the data to zero mean and unit variance
- [tdmi.py](https://github.com/PREMKUMARLOGANATHAN/MSc-Thesis-2024-2026-PL/blob/main/tdmi.py) – Choosing Optimal Time Delay (τ) using Fraser and Swinney Average Mutual Information Criterion (Fraser & Swinney, 1986)
- [thesis_amr_code.py](https://github.com/PREMKUMARLOGANATHAN/MSc-Thesis-2024-2026-PL/blob/main/thesis_amr_code.py) – Inferring Causal Relationship between AMR and Climate Variables
- [thesis_amu_code.py](https://github.com/PREMKUMARLOGANATHAN/MSc-Thesis-2024-2026-PL/blob/main/thesis_amu_code.py) – Inferring Causal Relationship between AMU and Climate Variables
- [thesis_amu_x_amr_code.py](https://github.com/PREMKUMARLOGANATHAN/MSc-Thesis-2024-2026-PL/blob/main/thesis_amu_x_amr_code.py) - Inferring Causal Relationship between AMU and AMR
- This research is inspired by the work [Inferring a Causal Relationship between Environmental Factors and Respiratory Infections Using Convergent Cross-Mapping](https://doi.org/10.3390/e25050807) by (Chen et al., 2023)

## Bibliography

- Cao, L. (1997). Practical method for determining the minimum embedding dimension of a scalar time series. Physica D: Nonlinear Phenomena, 110(1–2), 43–50. https://doi.org/10.1016/s0167-2789(97)00118-8 
- Chen, D., Sun, X., & Cheke, R. A. (2023). Inferring a causal relationship between environmental factors and respiratory infections using convergent cross-mapping. Entropy, 25(5), 807. https://doi.org/10.3390/e25050807 
- Fraser, A. M., & Swinney, H. L. (1986). Independent coordinates for strange attractors from Mutual Information. Physical Review A, 33(2), 1134–1140. https://doi.org/10.1103/physreva.33.1134 
- Sugihara, G., May, R., Ye, H., Hsieh, C., Deyle, E., Fogarty, M., & Munch, S. (2012). Detecting causality in complex ecosystems. Science, 338(6106), 496–500. https://doi.org/10.1126/science.1227079 

© 2026 Prem Kumar Loganathan. All rights reserved.
