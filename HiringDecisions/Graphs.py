import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

HiringData = pd.read_csv("Data/recruitment_data.csv")
print(HiringData.head())


plt.figure(figsize=(12, 8))
sns.boxplot(x="HiringDecision", y="DistanceFromCompany", data=HiringData, palette=['darkblue', 'lightcoral'])
plt.title("Box Plot of Distance from Company by Hiring Decision")
plt.xlabel("Hiring Decision")
plt.ylabel("Distance from Company")
plt.grid(True)
#plt.show()

plt.figure(figsize=(12, 8))
sns.boxplot(x="HiringDecision", y="PersonalityScore", data=HiringData, palette='Set1')
plt.title("Box Plot of Personality Score by Hiring Decision")
plt.xlabel("Hiring Decision")
plt.ylabel("Personality Score")
plt.grid(True)
#plt.show()

# 1. Define Distance Bins/Groups
HiringData['DistanceGroup'] = pd.cut(HiringData['DistanceFromCompany'], bins=[0, 5, 10, 20, np.inf], labels=['0-5', '5-10', '10-20', '20+'])

# 2. Calculate Acceptance Rate per Group
AcceptanceRate = HiringData.groupby('DistanceGroup')['HiringDecision'].mean()

# 3. Convert to Percentage and Display
AcceptanceRate = AcceptanceRate * 100
print("Acceptance Rate by Distance Group:")
print(AcceptanceRate)

# 4. Visualization (Bar Plot)
plt.figure(figsize=(10, 6))
AcceptanceRate.plot(kind='bar', color='skyblue')
plt.title('Acceptance Rate by Distance Group')
plt.ylabel('Acceptance Rate (%)')
plt.grid(axis='y')
plt.show()

print(HiringData.describe())


# 1. Define Skill Score Bins/Groups (Adjust bins as needed)
HiringData['SkillGroup'] = pd.cut(HiringData['SkillScore'], bins=[0, 25, 50, 75, 100], labels=['0-25', '25-50', '50-75', '75-100'])

# 2. Calculate Acceptance Rate per Group
AcceptanceRateBySkill = HiringData.groupby('SkillGroup')['HiringDecision'].mean()

# 3. Convert to Percentage and Display
AcceptanceRateBySkill = AcceptanceRateBySkill * 100
print("Acceptance Rate by Skill Group:")
print(AcceptanceRateBySkill)

# 4. Visualization (Bar Plot)
plt.figure(figsize=(10, 6))
AcceptanceRateBySkill.plot(kind='bar', color='lightgreen')
plt.title('Acceptance Rate by Skill Group')
plt.ylabel('Acceptance Rate (%)')
plt.grid(axis='y')
plt.show()


# Calculate Acceptance Rate per Group
AcceptanceRateBySkill = HiringData.groupby('SkillGroup')['HiringDecision'].mean() * 100

# Prepare data for the pie chart
labels = AcceptanceRateBySkill.index
sizes = AcceptanceRateBySkill.values
explode = (0.05, 0.05, 0.05, 0.05)  # Slightly explode each slice

# Create the pie chart
plt.figure(figsize=(8, 8))
plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', startangle=140, shadow=True)
plt.title('Acceptance Rate by Skill Group', fontsize=14)
plt.axis('equal')  # Equal aspect ratio ensures a circular pie chart

plt.show()

# ... (Your existing code to read the data)

# 1. Define Personality Score Bins/Groups (Adjust bins as needed)
HiringData['PersonalityGroup'] = pd.cut(HiringData['PersonalityScore'], bins=[0, 25, 50, 75, 100], labels=['0-25', '25-50', '50-75', '75-100'])

# 2. Calculate Acceptance Rate per Group
AcceptanceRateByPersonality = HiringData.groupby('PersonalityGroup')['HiringDecision'].mean() * 100

# Prepare data for the pie chart
labels = AcceptanceRateByPersonality.index
sizes = AcceptanceRateByPersonality.values
explode = (0.05, 0.05, 0.05, 0.05)  # Slightly explode each slice

# Create the pie chart
plt.figure(figsize=(8, 8))
plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', startangle=140, shadow=True)
plt.title('Acceptance Rate by Personality Group', fontsize=14)
plt.axis('equal')  # Equal aspect ratio ensures a circular pie chart

plt.show()


# Calculate Acceptance Rate per Group
AcceptanceRateByDistance = HiringData.groupby('DistanceGroup')['HiringDecision'].mean() * 100

# Prepare data for the pie chart
labels = AcceptanceRateByDistance.index
sizes = AcceptanceRateByDistance.values
explode = (0.05, 0.05, 0.05, 0.05)  # Slightly explode each slice

# Create the pie chart
plt.figure(figsize=(8, 8))
plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', startangle=140, shadow=True)
plt.title('Acceptance Rate by Distance Group', fontsize=14)
plt.axis('equal') 

plt.show()



# ... (Your existing code to read the data)

# 1. Define Age Group Bins (using your custom bins and labels)
bins = [20, 25, 30, 35, 40, 45, 50]
labels = ['20-25', '25-30', '30-35', '35-40', '40-45', '45-50']

HiringData['AgeGroup'] = pd.cut(HiringData['Age'], bins=bins, labels=labels, right=False) 

# 2. Calculate Acceptance Rate per Group
AcceptanceRateByAge = HiringData.groupby('AgeGroup')['HiringDecision'].mean() * 100

# Prepare data for the pie chart
labels = AcceptanceRateByAge.index
sizes = AcceptanceRateByAge.values
explode = (0.05, 0.05, 0.05, 0.05, 0.05, 0.05)

# Create the pie chart
plt.figure(figsize=(8, 8))
plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', startangle=140, shadow=True)
plt.title('Acceptance Rate by Age Group', fontsize=14)
plt.axis('equal')

plt.show()


# 

# 1. Handle Missing Values
HiringData['PreviousCompanies'] = HiringData['PreviousCompanies'].fillna(0)  # Or replace with a suitable value

# 2. Convert to Numeric (if necessary)
HiringData['PreviousCompanies'] = pd.to_numeric(HiringData['PreviousCompanies'], errors='coerce')
HiringData['PreviousCompanies'] = HiringData['PreviousCompanies'].fillna(0)

# 3. Define Groups based on Number of Previous Companies (adjust bins if needed)
bins = [0, 1, 2, 3, np.inf]
labels = ['No previous companies', '1 previous company', '2 previous companies', '3+ previous companies']
HiringData['PreviousCompaniesGroup'] = pd.cut(HiringData['PreviousCompanies'], bins=bins, labels=labels)

# 4. Calculate Acceptance Rate per Group (filtering out NaN values)
AcceptanceRateByPreviousCompanies = HiringData.groupby('PreviousCompaniesGroup')['HiringDecision'].mean().dropna() * 100

# Prepare data for the pie chart
labels = AcceptanceRateByPreviousCompanies.index
sizes = AcceptanceRateByPreviousCompanies.values
explode = (0.05, 0.05, 0.05, 0.05) 

# Create the pie chart
plt.figure(figsize=(8, 8))
plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', startangle=140, shadow=True)
plt.title('Acceptance Rate by Number of Previous Companies', fontsize=14)
plt.axis('equal')

plt.show()