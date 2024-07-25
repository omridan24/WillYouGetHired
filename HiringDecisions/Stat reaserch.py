#in this file We will do out statistical research. 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_1samp
from scipy.stats import t
from scipy.stats import chi2





HiringData = pd.read_csv("Data/recruitment_data.csv")
#our statistical significance level is alpha = 0.05
alpha = 0.05

#We Want to find out the connetction between The skills level of the candidate, and his/her acceptence rate.
#################################
# Question 1
#################################
#starting with the skiil score that the candidate got in their interview
#Our H0- Base assumption is that that Avg skill score of candidates that were accepted to work is 70
mu_0 = 70


# Here we will take only those who were excepted
accepted_candidates = HiringData[HiringData['HiringDecision'] == 1]

# Finding the amount of accepted candidates
num_of_accepted_cands = len(accepted_candidates)

# Calculate the unbiased variance of the SkillScore column
unbiased_variance = np.var(accepted_candidates['SkillScore'], ddof=1)


# Now we will find the Standard diviation of the accepted candidates
std_dev_skill_score = np.sqrt(unbiased_variance)


# estimate  the mean SkillScore of accepted candidates
mean_skill_score_accepted = accepted_candidates['SkillScore'].mean()

#Here we will show that we know how to calc the t statistic on our own, and later we will do that using an imported function
t_statistic_skill = (mean_skill_score_accepted - mu_0) / (std_dev_skill_score/np.sqrt(num_of_accepted_cands))


# Calculate the critical value for a two-tailed test
critical_value = t.ppf(1 - alpha/2, df=num_of_accepted_cands - 1) 

#Calculating the p-value for two tailed
# We can see that they use CDF to find the values that are equal or greater(or smaller) then the t- stat we found
p_value = 2 * (1 - t.cdf(abs(t_statistic_skill), num_of_accepted_cands - 1)) 

#The T-statistic is lower the the critical value t_statistic<critical_value so we reject the HO
#*************** finished first test

# we think that you are going to need to have 60 in personality to get excepted. our new mu_0 is <=60,
mu_0 = 60
#our H1 is that mu is > 60 


#calculating the T statistic and p- value using the imported library
t_statistic_personality, p_value = ttest_1samp(accepted_candidates['PersonalityScore'], mu_0)

critical_value = t.ppf(1 - alpha, num_of_accepted_cands - 1) 

#we found that out t_statistic < critical value, so we except H0

#****************


#Now our H0 for mu of distance of accepted candidates is 15 KM
mu_0 = 15
#Our H1 is that if distance is less then 15 KM we reject it
#calculating the T statistic and p- value using the imported library
t_statistic, p_value = ttest_1samp(accepted_candidates['DistanceFromCompany'], mu_0)
#Calc the critical value
critical_value = -t.ppf(1 - alpha, num_of_accepted_cands - 1) 
#We found that our t_Statistic is bigger than the critical value , so we except H0


#################################
# Question 2
#################################

# hypothesis testing for variance 
# Checking if skill score has to do with experiance in years
# We'll begin by taking the group of those who have over 13 years of experiance years
very_experianced = HiringData[HiringData['ExperienceYears']>13]
amount_of_very_experianced = len(very_experianced)
# we have 209 candidates in this group
# We think that the skill score for this group is very high,and that the variance is not going to be big.
#First we will find the variance in SkillScore for the whole sample
variance_whole_sample = np.var(HiringData['SkillScore'], ddof=1)
#we know that the variance in SkillScore for the whole sample is 861.63
#Our assumption is that because we have a group of people that are very expirienced,they are going to get very high skill score with a smaller variance than the whole test
# our H0 for the variance of this group is <= 750, and we will check that with right sided test
var_h0 = 750
# we will reject H0 if the chi value of the estimation will be greater then the chi value of our critical value
critical_value = chi2.ppf(1 - alpha, amount_of_very_experianced-1)
# critical value is 242.64
#We will find an estimator for the mean of their skill set
mean_skill_experianced = very_experianced['SkillScore'].mean()
#we found that the mean skill score of this group is 51.976(surprisingly low)
# Now let's find the variance of the sample
unbiased_variance_for_very_experianced = np.var(very_experianced['SkillScore'], ddof=1)
#our unbiased_variance_for_very_experianced is 756.56
#Now we will find the value of our chi stat for variance, assuming the our H0 is true
chi_stat_var = (amount_of_very_experianced-1) * unbiased_variance_for_very_experianced / (var_h0 ** 2)
# the chi value of our sample is 0.32, much lower then the critical value of 242.64 ,so we except our H0 with significance level "Ramat Muvhakut" of 0.05
#let's find the P-value
p_value = 1- chi2.cdf(chi_stat_var,amount_of_very_experianced-1)
#the p_value is 1.0




#################################
# Question 3
#################################
















