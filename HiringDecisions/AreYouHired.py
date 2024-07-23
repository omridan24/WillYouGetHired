import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

HiringData = pd.read_csv("Data/recruitment_data.csv")
print(HiringData.head())

plt.figure()
plt.hist(HiringData["ExperienceYears"],bins=20, alpha=0.7, color='skyblue', edgecolor='black')
plt.title("ExperienceYears Ages")

plt.show()