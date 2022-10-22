from scipy.stats import shapiro, anderson, normaltest, ttest_ind, f_oneway, ttest_rel, kruskal, mannwhitneyu, wilcoxon

"""
NORMALITY TEST
Tests whether a data sample has a Gaussian distribution.
    Assumptions
        Observations in each sample are independent and identically distributed (iid).
    Interpretation
        H0: the sample has a Gaussian distribution.
        H1: the sample does not have a Gaussian distribution.
"""
SHAPIRO = 'shapiro'
DAGOSTINO = 'dagostino'
ANDERSON = 'anderson'
def normality_test(distribution, p_value_th=0.05, mode=SHAPIRO):
    if mode == SHAPIRO:
        stat, p = shapiro(distribution)
    elif mode == DAGOSTINO:
        stat, p = normaltest(distribution)
    elif mode == ANDERSON:
        stat, p = anderson(distribution)
    else:
        raise Exception("Unrecognized mode")
    print('stat=%.3f, p=%.3f' % (stat, p))
    if p > p_value_th:
        print('Probably Gaussian')
        return False
    else:
        print('Probably not Gaussian')
        return True

"""
PARAMETRIC STATISTICAL HYPOTHESIS TESTS
This section lists statistical tests that you can use to compare data samples.

**Student’s t-test
    Tests whether the means of two independent samples are significantly different.
    
    Assumptions
        Observations in each sample are independent and identically distributed (iid).
        Observations in each sample are normally distributed.
        Observations in each sample have the same variance.
        
    Interpretation
        H0: the means of the samples are equal.
        H1: the means of the samples are unequal.

**Paired Student’s t-test
    Tests whether the means of two paired samples are significantly different.
    
    Assumptions
        Observations in each sample are independent and identically distributed (iid).
        Observations in each sample are normally distributed.
        Observations in each sample have the same variance.
        Observations across each sample are paired.
        
    Interpretation
        H0: the means of the samples are equal.
        H1: the means of the samples are unequal.

** Analysis of Variance Test (ANOVA)
    Tests whether the means of two or more independent samples are significantly different.
    
    Assumptions
        Observations in each sample are independent and identically distributed (iid).
        Observations in each sample are normally distributed.
        Observations in each sample have the same variance.
    
    Interpretation
        H0: the means of the samples are equal.
        H1: one or more of the means of the samples are unequal.
"""

TTEST = "t-test"
PAIRED_TTEST = 'paired-t-test'
ANOVA = 'anova'
def parametric_statical_test(distrib1, distrib2, p_value_th=0.05, mode=TTEST):
    if mode == TTEST:
        stat, p = ttest_ind(distrib1, distrib2)
    elif mode == PAIRED_TTEST:
        stat, p = ttest_rel(distrib1, distrib2)
    elif mode == ANOVA:
        stat, p = f_oneway(distrib1, distrib2)
    else:
        raise Exception("Unrecognized mode")
    print('stat=%.3f, p=%.3f' % (stat, p))
    if p > p_value_th:
        print('Probably the same distribution')
        return False
    else:
        print('Probably different distributions')
        return True


"""
Nonparametric Statistical Hypothesis Tests

**Mann-Whitney U Test
    Tests whether the distributions of two independent samples are equal or not.
    
    Assumptions
        Observations in each sample are independent and identically distributed (iid).
        Observations in each sample can be ranked.
        
    Interpretation
        H0: the distributions of both samples are equal.
        H1: the distributions of both samples are not equal.
        
**Wilcoxon Signed-Rank Test
    Tests whether the distributions of two paired samples are equal or not.
    
    Assumptions
        Observations in each sample are independent and identically distributed (iid).
        Observations in each sample can be ranked.
        Observations across each sample are paired.
        
    Interpretation
        H0: the distributions of both samples are equal.
        H1: the distributions of both samples are not equal.
        
**Kruskal-Wallis H Test
    Tests whether the distributions of two or more independent samples are equal or not.
    
    Assumptions
        Observations in each sample are independent and identically distributed (iid).
        Observations in each sample can be ranked.
        
    Interpretation
        H0: the distributions of all samples are equal.
        H1: the distributions of one or more samples are not equal.
"""
WILCOXON = "wilcoxon"
UTEST = "u-test"
HTEST = "h-test"
def non_parametric_statical_test(distrib1, distrib2, p_value_th=0.05, mode=HTEST):
    if mode == HTEST:
        stat, p = kruskal(distrib1, distrib2)
    elif mode == UTEST:
        stat, p = mannwhitneyu(distrib1, distrib2)
    elif mode == WILCOXON:
        stat, p = wilcoxon(distrib1, distrib2)
    else:
        raise Exception("Unrecognized mode")
    print('stat=%.3f, p=%.3f' % (stat, p))
    if p > p_value_th:
        print('Probably the same distribution')
        return False
    else:
        print('Probably different distributions')
        return True