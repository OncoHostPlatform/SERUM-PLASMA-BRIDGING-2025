# SERUM-PLASMA-BRIDGING-2025
## Overview
This repository contains the code used to generate figures for the manuscript [C. Lahav et al., "Bridging the Gap: A Systematic Approach to Integrating Serum and Plasma Proteomic Datasets for Biomarker Studies", 2026](https://doi.org/10.1016/j.jpba.2026.117421). This study presents a computational approach for bridging between serum and plasma proteomic datasets derived from the aptamer-based SomaScan assay, enabling cross-specimen data use in predictive biomarkers development.
## Why it matters
This methodology enables integrating biobank data in liquid proteomics, thus expanding patient cohorts for faster biomarker discovery and validation.
## Citation
If you use this code or data in your research, please cite:  
C. Lahav et al., “Bridging the Gap: A Systematic Approach to Integrating Serum and Plasma Proteomic Datasets for Biomarker Studies,” Journal of Pharmaceutical and Biomedical Analysis, p. 117421, Feb. 2026, doi: 10.1016/j.jpba.2026.117421.
## Data
The `generate_manuscript_figures.py` script requires supplementary tables that will be available upon publication of the manuscript. Place the files in a subfolder named "data".
## Code
This repository consists of the following Python files:
- `generate_manuscript_figures.py` - generates figures presented in the manuscript as listed below.
- `iterative_scaler.py` - IterativeScaler class for robust z-score standardization that iteratively excludes outliers until convergence.
- `iqr_outlier_detection.py` - Calculate outlier bounds using the Interquartile Range (IQR) method.
## Figures
- Figure 3C: Association between median plasma-to-serum ratio and plasma-serum protein correlation.
- Figure 5: Inter-cohort agreement of scaling parameters.
- Figure S3B: Correlations between plasma and serum proteomes.
- Figure S4: Relationship between Pearson-Spearman correlation differences and protein measurability.
- Figure S5B: Serum-plasma Spearman correlation distributions of protein measurements in different protein sets.
- Figure S7B: Concordance between internal and external plasma datasets.
- Figure S8B: Concordance between internal and external serum datasets.
- Figure S9 matched samples: Comparison of linear scaling parameters derived from different cohorts.
- Figure S10: Inter-cohort agreement of serum-plasma protein correlations.
## Reproducibility
To reproduce the figures from our manuscript:
1. Clone this repository
2. Install the required dependencies (see Requirements section)
3. Download the supplementary tables (will be available upon publication) and place them in the script directory
4. Run the Python script with the appropriate parameters
## Requirements
This code was developed and tested using `Python 3.12.5`.
The following package dependencies are required:
  - `numpy==2.0.2`
  - `pandas==2.2.3`
  - `scipy==1.14.1`
  - `matplotlib==3.9.4`
  - `seaborn==0.13.0`
  - `scikit-learn==1.5.2`
  - `lifelines==0.30.0`
## License
This project is licensed under the BSD 3-Clause License - see the LICENSE file for details.
## Contact
For questions regarding the code or manuscript, please open an issue in this repository or contact the corresponding author.