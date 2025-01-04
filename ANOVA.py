# The `environment.yml` file is used to manage and reproduce conda environments.
# 
# To create this file from your current conda environment:
# 1. Run `conda env export > environment.yml` in your terminal. This command exports
#    the specifications of the active conda environment to an `environment.yml` file.
# 
# To recreate the environment from an `environment.yml` file:
# 1. Run `conda env create -f environment.yml`. This will create a new conda environment
#    with the exact dependencies listed in the file.
# 
# Note: If the environment already exists and you want to update it, use:
#    `conda env update -f environment.yml`.


import numpy as np
import pandas as pd
import pingouin as pg
from scipy import stats
from scipy.stats import shapiro
from statsmodels.stats.anova import AnovaRM
from pingouin import sphericity
from pingouin import sphericity, pairwise_ttests, power_rm_anova
from statsmodels.stats.power import TTestPower
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

def one_sample_t_test(data, test_value=0, alpha=0.05):
    """
    Perform a one-sample t-test to check if the sample mean is equal to a given value (test_value),
    and calculate the statistical power of the test. Also checks for normality of the data.

    Parameters:
    - data: Sample data (array-like).
    - test_value: The value to compare the sample mean to (default is 0).
    - alpha: Significance level (default is 0.05).

    Returns:
    - dict: A dictionary containing the t-statistic, p-value, reject_null, power, and normality test result.
    """
    # Calculate the sample mean and standard deviation
    sample_mean = np.mean(data)
    sample_std = np.std(data, ddof=1)
    n = len(data)
    
    # Perform one-sample t-test (default is two-tailed)
    t_stat, p_value = stats.ttest_1samp(data, test_value)
    
    # Determine if we reject the null hypothesis
    reject_null = p_value < alpha
    
    # Calculate the effect size (Cohen's d)
    effect_size = (sample_mean - test_value) / sample_std
    
    # Perform power analysis using statsmodels
    power_analysis = TTestPower()
    power = power_analysis.solve_power(effect_size=effect_size, nobs=n, alpha=alpha, alternative='two-sided')
    
    # Perform normality check using Shapiro-Wilk test
    normality_stat, normality_p_value = stats.shapiro(data)
    normality_passed = normality_p_value > alpha  # Normality assumption is passed if p > alpha
    
    # Return results as a dictionary
    results = {
        "t_stat": t_stat,
        "p_value": p_value,
        "reject_null": reject_null,
        "power": power,
        "normality_stat": normality_stat,
        "normality_p_value": normality_p_value,
        "normality_passed": normality_passed
    }
    
    return results


def perform_ANOVA_tests(*groups):
    """
    Perform various statistical tests on the given datasets.
    
    Parameters:
    - *groups: Variable number of groups of data (e.g., group1, group2, group3, ...)
    
    Prints the results of normality tests, Levene's test, Bartlett's test, ANOVA with post hoc, and Kruskal-Wallis with post hoc.
    """

    import pingouin as pg
    import pandas as pd
    import scipy.stats as stats
    
    print('-----------------------------------------')
    # Normality tests for each group
    for i, group in enumerate(groups, 1):
        print(f'## Normality Test for Group {i}')
        print(f'Normality test for group {i}')
        results = normality_test(group)
        print(results)
    
    # Levene's test for equality of variances
    print('## Levene Test for Equality of Variance')
    print('Levene test ')
    statistic, p_value = levene_test(*groups)
    print(f'Levene Test Statistic: {statistic}, p-value: {p_value}')
    
    # Bartlett's test for equality of variances
    print('## Bartlett Test for Equality of Variance')
    print('Bartlett test ')
    statistic, p_value = bartlett_test(*groups)
    print(f'Bartlett Test Statistic: {statistic}, p-value: {p_value}')
    
    # ANOVA test with post hoc analysis (only if more than two groups)
    if len(groups) > 2:
        print('## ANOVA with Post Hoc Test')
        print('ANOVA test ')
        result = anova_with_posthoc(*groups)
        print(result)
    
    # Kruskal-Wallis test with Dunn's post hoc test (non-parametric, used when data is not normal)
    print('## Kruskal-Wallis Test with Dunn\'s Post Hoc')
    print('Kruskal-Wallis test ')
    result = kruskal_wallis_with_posthoc(*groups)
    print(result)


    # Repeated Measures ANOVA and Post Hoc for the given groups
    print('## Repeated Measures ANOVA with Post Hoc Comparisons')
    
    # Ensure all groups are of the same length (number of subjects)
    num_subjects = len(groups[0])  # Assuming all groups have the same number of subjects
    for i, group in enumerate(groups, 1):
        if len(group) != num_subjects:
            print(f"Warning: Group {i} does not have the same number of subjects. Skipping repeated measures ANOVA.")
            return

    # Create a long-format DataFrame for the repeated measures ANOVA
    subjects = list(range(1, num_subjects + 1))  # Subject identifiers (1 to n)
    conditions = [f'Condition_{i}' for i in range(1, len(groups) + 1)]  # Create condition names (Condition_1, Condition_2, etc.)

    # Creating a long-format DataFrame where each row is a subject's score under a condition
    data = pd.DataFrame({
        'Subject': subjects * len(groups),  # Repeating subjects for each condition
        'Condition': [condition for condition in conditions for _ in range(num_subjects)],  # Repeating conditions for each subject
        'Score': [score for group in groups for score in group]  # Flatten the groups into a single list
    })

    # Perform Repeated Measures ANOVA using pingouin
    anova = pg.rm_anova(dv='Score', within='Condition', subject='Subject', data=data, detailed=True)

    # Test for sphericity
    mauchly = pg.sphericity(data=data, dv='Score', within='Condition', subject='Subject')
    print("Test for sphericity:")
    print(mauchly)

    # Print ANOVA results
    print("ANOVA Results:")
    print(anova)

    # Perform post hoc comparisons (pairwise t-tests) with Bonferroni correction
    post_hoc = pg.pairwise_tests(dv='Score', within='Condition', subject='Subject', data=data, padjust='bonferroni')

    # Print post hoc comparison results
    print("\nPost Hoc Comparisons (with Bonferroni correction):")
    print(post_hoc)

    print("-----------------------------------------")
    print("-----------------------------------------")

def normality_test(data):
    """
    Perform normality tests on the given data.

    Parameters:
    data: A 1D array or list of numerical data.

    Returns:
    results: A dictionary containing test statistics and p-values for multiple normality tests.
    """
    from scipy.stats import shapiro, anderson, normaltest
    
    results = {}

    # Shapiro-Wilk Test
    stat_shapiro, p_value_shapiro = shapiro(data)
    results['Shapiro-Wilk Test'] = {'Statistic': stat_shapiro, 'P-value': p_value_shapiro}

    return results

def one_factor_repeated_anova(*groups):
    """
    Perform a one-factor repeated measures ANOVA with assumption checks.

    Parameters:
        *groups: Variable number of arrays or lists representing repeated measures data.

    Returns:
        Dictionary with results of assumption checks and ANOVA summary.
    """
    # Check input consistency
    if len(groups) < 2:
        raise ValueError("At least two groups are required for repeated measures ANOVA.")

    n_subjects = len(groups[0])
    if not all(len(group) == n_subjects for group in groups):
        raise ValueError("All groups must have the same number of observations (balanced design).")

    # Convert data into a long-form DataFrame
    data = pd.DataFrame({f"Group_{i+1}": group for i, group in enumerate(groups)})
    data["Subject"] = np.arange(1, n_subjects + 1)
    data_long = data.melt(id_vars="Subject", var_name="Condition", value_name="Score")

    # Check for normality (Shapiro-Wilk test for each group)
    normality_results = {col: shapiro(data[col]).pvalue for col in data.columns if col != "Subject"}
    normality_passed = all(p > 0.05 for p in normality_results.values())

    # Check for sphericity (Mauchly's test using Pingouin)
    sphericity_res = sphericity(data_long, dv="Score", subject="Subject", within="Condition")
    sphericity_passed = sphericity_res.pval > 0.05;

    # Perform repeated measures ANOVA
    anova_results = pg.rm_anova(data=data_long, dv="Score", within="Condition", subject="Subject")

    # Perform post hoc analysis (pairwise comparisons)
    post_hoc_results = pg.pairwise_ttests(
        data=data_long,
        dv="Score",
        within="Condition",
        subject="Subject",
        padjust="bonferroni"  # Use Bonferroni correction for multiple comparisons
    )

    # Power analysis for each post hoc comparison
    power_results = []
    power_analysis = TTestPower()

    for index, row in post_hoc_results.iterrows():
        # Calculate Cohen's d for each comparison using t-statistic and sample size
        t_stat = row['T']
        effect_size_d = t_stat / np.sqrt(n_subjects)
        
        # Perform power analysis using the Cohen's d for each pairwise comparison
        power = power_analysis.solve_power(effect_size=effect_size_d, nobs=n_subjects, alpha=0.05)
        
        power_results.append({
            "Contrast": f"{row['A']} vs {row['B']}",
            "T-statistic": t_stat,
            "Effect Size (Cohen's d)": effect_size_d,
            "Power": power
        })

    # Return results
    return {
        "Normality Results (Shapiro-Wilk)": normality_results,
        "Normality Passed": normality_passed,
        "Sphericity Test (Mauchly)": {
            "p-value": sphericity_res.pval,
            "Passed": sphericity_passed
        },
        "ANOVA Summary": anova_results,
        "Post Hoc Results": post_hoc_results,
        "power_results": power_results,
    }

### Defining the "More Affected Midstance" values for each mode (Stance & Swing, Stance, Swing)
print('## repeated ANOVA assistance mode effect on knee extension angle of More Affected side in Midstance')
stanceswing = [0.5, 8.0, 8.2, 9.3, 11.0, 19.5, 36.7]    # Stance & Swing
stance = [-5.3, 1.4, 8.1, 8.9, 6.2, 17.1, 18.4]         # Stance
swing = [-8.8, 2.5, 5.7, 1.4, -2.5, 5.1, 7.6]           # Swing

res = one_factor_repeated_anova(stanceswing, stance, swing)
print(res)


# # Reduction in Crouch: θ Initial Contact (°) More Affected
print('## assistance mode effect on knee extension angle of More Affected side in Initial Contact')
stanceswing = [0.4,7.3,3.3,5.4,10.7,8.5,11.7]   # Stance & Swing
stance = [-1.9,2.8,1.8,0.9,-0.5,6.0,0.3]        # Stance
swing = [-12.5,11.8,-2.6,3.9,5.7,15.6,10.6]     # Swing

res = one_factor_repeated_anova(stanceswing, stance, swing)
print(res)


# # Reduction in Crouch: θ Initial Contact (°) less affected
print('## assistance mode effect on knee extension angle of Less Affected side in Initial Contact')
stanceswing = [0.5,10.2,2.8,6.7,-2.1,8.8,19.3]   # Stance & Swing
stance = [-7.3,-4.3,-3.3,1.9,-11.0,1.9,-0.7]        # Stance
swing = [-6.7,6.1,3.8,10.8,-0.5,5.6,10.9,]     # Swing

res = one_factor_repeated_anova(stanceswing, stance, swing)
print(res)


# # Reduction in Crouch: θ Midstance (°) less affected
print('## assistance mode effect on knee extension angle of Less Affected side in Midstance')
stanceswing = [-6.2,5.2,5.6,8.9,8.6,6.9,11.5]   # Stance & Swing
stance = [-7.6,2.0,6.0,15.0,7.1,5.0,15.3]        # Stance
swing = [-12.4,5.2,-7.6,3.5,-2.3,-0.4,2.7]     # Swing

res = one_factor_repeated_anova(stanceswing, stance, swing)
print(res)

### check the effect of exo assistance
stanceswing_midstance = [0.5, 8.0, 8.2, 9.3, 11.0, 19.5, 36.7]      # Stance & Swing most affected
stanceswing_Initial = [0.4,7.3,3.3,5.4,10.7,8.5,11.7]               # Stance & Swing most affected

# # Perform one-sample t-test
res = one_sample_t_test(stanceswing_midstance, test_value=0, alpha=0.05)
results = normality_test(stanceswing_midstance)
print(res)

res = one_sample_t_test(stanceswing_Initial, test_value=0, alpha=0.05)
results = normality_test(stanceswing_Initial)
print(res)

stanceswing_midstance_less = [-6.2,5.2,5.6,8.9,8.6,6.9,11.5]    # Stance & Swing less affected
stanceswing_Initial_less = [0.5,10.2,2.8,6.7,-2.1,8.8,19.3]   # Stance & Swing less affected
res = one_sample_t_test(stanceswing_midstance_less, test_value=0, alpha=0.05)
results = normality_test(stanceswing_midstance_less)
print(res)
# # If p-value is less than 0.05, data is not normally distributed
print("Data is not normally distributed. Performing the Wilcoxon Signed-Rank Test.")
# Perform Wilcoxon Signed-Rank Test (comparing against a population mean of 0)
stat, p_value = stats.wilcoxon(np.array(stanceswing_midstance_less) - 0)  # 0 is the hypothesized median
print(f"Wilcoxon Signed-Rank Test p-value: {p_value}")


res = one_sample_t_test(stanceswing_Initial_less, test_value=0, alpha=0.05)
results = normality_test(stanceswing_Initial_less)
print(res)

