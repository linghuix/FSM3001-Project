import numpy as np
from scipy.stats import f_oneway

###  knee extension angle of the most-affected limb in midstance (Stance & Swing, Stance, Swing assistance mode)
stanceswing = [0.5, 8.0, 8.2, 9.3, 11.0, 19.5, 36.7]    # Stance & Swing
stance = [-5.3, 1.4, 8.1, 8.9, 6.2, 17.1, 18.4]         # Stance
swing = [-8.8, 2.5, 5.7, 1.4, -2.5, 5.1, 7.6]           # Swing
# Perform the one-way ANOVA
statistic, p_value = f_oneway(stanceswing, stance, swing)
print(statistic, p_value)

