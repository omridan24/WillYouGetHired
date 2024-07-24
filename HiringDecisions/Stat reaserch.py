#in this file We will do out statistical research. 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_1samp
from scipy.stats import t

HiringData = pd.read_csv("Data/recruitment_data.csv")


#We Want to find out the connetction between The skills level of the candidate, and his/her acceptence rate.

#Our H0- Base assumption is that that Avg skill score of candidates that were accepted to work is 70
mu_0 = 70

#our statistical significance level is alpha = 0.05
alpha = 0.05
accepted_candidates = HiringData[HiringData['HiringDecision'] == 1]
#Now we will find the rejection region of the H0
num_of_accepted_cands = len(accepted_candidates)


# Calculate the unbiased variance of the SkillScore column
unbiased_variance = np.var(accepted_candidates['SkillScore'], ddof=1)


# Now we will find the Standard diviation of the accepted candidates
std_dev_skill_score = np.sqrt(unbiased_variance)


# estimate and print the mean SkillScore of accepted candidates
mean_skill_score_accepted = accepted_candidates['SkillScore'].mean()

#Now we will find the t value T value of our mean of the sample : 
t_statistic, p_value = ttest_1samp(accepted_candidates['SkillScore'], mu_0)

# Calculate the critical value for a two-tailed test
critical_value = t.ppf(1 - alpha/2, df=num_of_accepted_cands - 1) 

#The T-statistic is lower the the critical value t_statistic<critical_value so we reject the HO
#*************** finished first test

# our new mu_0 is <=60, we think that you are going to need to have 60 in personality to get excepted.
mu_0 = 60
#our H1 is that mu is > 60 
#calculating the mean of accepted candidated personality score

mean_personality_score_accepted = accepted_candidates['PersonalityScore'].mean()

#calculating the T value
t_statistic, p_value = ttest_1samp(accepted_candidates['PersonalityScore'], mu_0)

critical_value = t.ppf(1 - alpha, df=num_of_accepted_cands - 1) 

#we found that out t_statistic < critical value, so we except H0

#****************


#Now our H0 for mu of distance of accepted candidates is 15 KM
mu_0 = 15
#Our H1 is that if distance is less then 15 KM we reject it
#we will calc out t_statistic of distances 
t_statistic, p_value = ttest_1samp(accepted_candidates['DistanceFromCompany'], mu_0)
#Calc the critical value
critical_value = -t.ppf(1 - alpha, df=num_of_accepted_cands - 1) 
#We found that our t_Statistic is bigger than the critical value , so we except H0














