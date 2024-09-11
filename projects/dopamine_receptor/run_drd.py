"""
run drd cell analysis
"""

import os, sys
import numpy as np
import statsmodels.api as sm
import scipy
import matplotlib.pyplot as plt
from scipy.io import loadmat
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone

from projects.pyr_reward.rewardcell import perireward_binned_activity
from projects.dopamine_receptor.drd import load_and_filter_fall_data, run_glm, plot_glm_results, plot_peri_reward
# Main pipeline function
def main_pipeline(src, mouse_name, days, planelut):
    for dy in days:
        day_dir = os.path.join(src, mouse_name, str(dy))
        for root, dirs, files in os.walk(day_dir):
            for file in files:
                if 'plane' in root and file.endswith('Fall.mat'):
                    fall_file = os.path.join(root, file)

                    # Step 1: Load and filter data
                    dFF_iscell_filtered, f = load_and_filter_fall_data(fall_file)

                    # Step 2: Run GLM
                    dff_res, perirew = run_glm(dFF_iscell_filtered, f)

                    # Step 3: Plot results
                    # plot_glm_results(dff_res, dy, planelut, root)
                    plot_peri_reward(perirew, dy, planelut, root, 12, 0.2)

# Example usage
if __name__ == '__main__':
    src = r'Y:\drd'
    mouse_name = 'e255'
    days = [1]
    planelut = {0: 'SR', 1: 'SP', 2: 'SO'}
    main_pipeline(src, mouse_name, days, planelut)
