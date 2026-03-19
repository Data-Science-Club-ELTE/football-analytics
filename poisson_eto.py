import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn
from scipy.stats import poisson,skellam


fizzdf = pd.read_excel("FIZZ2526.xlsx", index_col=0)
print(fizzdf.mean(numeric_only=True))

poisson_pred = np.column_stack([[poisson.pmf(i, fizzdf.mean(numeric_only=True)[j]) for i in range(8)] for j in range(2)])
print(poisson_pred)


# plot histogram of actual goals
plt.hist(fizzdf[['HomeGoal', 'AwayGoal']].values, range(9), 
         alpha=0.7, label=['Home', 'Away'], color=["#FFA07A", "#20B2AA"])

# add lines for the Poisson distributions
pois1, = plt.plot([i-0.5 for i in range(1,9)], poisson_pred[:,0]*100,
                  linestyle='-', marker='o',label="Home", color = '#CD5C5C')
pois2, = plt.plot([i-0.5 for i in range(1,9)], poisson_pred[:,1]*100,
                  linestyle='-', marker='o',label="Away", color = '#006400')

leg=plt.legend(loc='upper right', fontsize=13, ncol=2)
leg.set_title("Poisson           Actual        ", prop = {'size':'14', 'weight':'bold'})

plt.xticks([i-0.5 for i in range(1,9)],[i for i in range(8)])
plt.xlabel("Goals per Match",size=13)
plt.ylabel("Proportion of Matches",size=13)
plt.title("Number of Goals per Match (EPL 2016/17 Season)",size=14,fontweight='bold')
#plt.ylim([-0.004, 0.4])
plt.tight_layout()
plt.show()