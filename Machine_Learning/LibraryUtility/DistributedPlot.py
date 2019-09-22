# Designed by Barun  
# Date : 21/9/2109
# Mission: Exporing AI  library like pandas ,Matplotlib, numpy!!
  
# importing pandas as pd 
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
from matplotlib import pyplot as plt

data = {"Name": ["Barun", "prakash", "Jay", "RAJ" ,"Dinesh"], 
       "capital": ["Brasilia", "Moscow", "New Dehli", "Beijing", "Pretoria"], 
       "Experince": [5, 7, 3, 9, 1], 
       "NoOflineCode": [200, 143, 1252, 1357, 5298] } 
  
data_table = pd.DataFrame(data) 
print(data_table) 

print(data['NoOflineCode'])

print("______________________________")

# Function to plot 
plt.plot(data["Name"],data["Experince"]) 
plt.show()
#print("BAR-------------------")
#plt.bar(["country"],data["area"])

plt.show()

