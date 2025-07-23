import matplotlib.pyplot as plt
import numpy as np

# plt.figure(0)

# y = np.array([300, 50, 500, 50, 800])

# labels = ['Server', 'Legal', 'Strom', 'Miete', 'Gehälter']

# plt.pie(y, labels=labels, autopct=lambda p:f'{p*sum(y)/100 :.0f}€, ({p:.2f}%)', startangle=90, shadow=True)
# plt.legend()

# plt.figure(1)

# y = np.array([300, 50, 500, 50, 800])

# labels = ['Server', 'Legal', 'Strom', 'Miete', 'Gehälter']

# plt.pie(y, labels=labels, autopct=lambda p:f'{p*sum(y)/100 :.0f}€, ({p:.2f}%)', startangle=90, shadow=True)
# plt.legend()

fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,10)) #ax1,ax2 refer to your two pies

# 1,2 denotes 1 row, 2 columns - if you want to stack vertically, it would be 2,1

y_1 = np.array([300, 50, 500, 50, 800])

labels_1 = ['Server', 'Legal', 'Strom', 'Miete', 'Gehälter']

ax1.pie(y_1,labels = labels_1, autopct = lambda p:f'{p*sum(y_1)/100 :.0f}€, ({p:.2f}%)', startangle=260, shadow=True) #plot first pie
ax1.set_title('Ohne EXIST - 1700€')

y_2 = np.array([300, 50, 500, 50])

labels_2 = ['Server', 'Legal', 'Strom', 'Miete']

ax2.pie(y_2 ,labels = labels_2, autopct = lambda p:f'{p*sum(y_2)/100 :.0f}€, ({p:.2f}%)', startangle=90, shadow=True) #plot second pie
ax2.set_title('Mit EXIST - 900€')

plt.savefig('pie_chart_better.png')
plt.show()