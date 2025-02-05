## Zahra's analysis for pyramidal cell data during VIP optogenetics
This script performs analysis on pyramidal cell data to identify differentially activated and inactivated cells, compute various metrics, and generate plots for visual representation. The analysis is particularly focused on optogenetic manipulation effects on neuronal activity.

### Prerequisites
To run this script, you need the following libraries installed:
```
numpy
h5py
scipy
matplotlib
sys
pandas
pickle
seaborn
random
math
sklearn
```
You also need to have the placecell package, which includes custom functions used in the analysis.

### Data
The script reads from a condition DataFrame stored as a CSV file: conddf_neural_com_inference.csv.
Outputs are saved in the specified directory: `C:\Users\Han\Box\neuro_phd_stuff\han_2023-\thesis_proposal`.
Main Steps
- Import Libraries and Data: Load necessary libraries and import the condition DataFrame.
- Compute Metrics: Use `get_pyr_metrics_opto` to compute metrics for each day and store them in a list.
- DFF Calculation: Calculate the delta fluorescence (dff) for optogenetic (opto) and control conditions.
- Plotting: Generate various plots to visualize the data:
  1. DFF differences between opto and control conditions.
  2. Fraction of cells near reward zones.
  3. Enrichment of tuning curves.
  4. COM (center of mass) shift analysis.
  5. Proportion of inactivated and activated cells.
  6. Spatial information of inactive vs. active cells.
  7. Tuning of inactive cells over time.
- Save Results: Save figures and potentially intermediate data for further analysis.

### Set up Environment
Ensure all required libraries are installed. Add the custom path to your sys.path if needed.
```
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab')
```

### Notes
Adjust file paths and directory names as per your local setup.
The script saves figures in the specified output directory.

### Conclusion
This script facilitates the analysis of pyramidal cell data, focusing on identifying differential activity due to optogenetic manipulation. The generated plots provide visual insights into the effects of the manipulation, aiding in further understanding and interpretation of the experimental results.