import random
from itertools import count
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

plt.style.use('fivethirtyeight')




data = pd.read_csv('data.csv')
x = data['episode']
y1 = data['critical_accuracy']
y2 = data['warning_accuracy']
y3 = data["normal_accuracy"]
y4 = data['Proactive']
y5 = data['Reactive']

plt.cla()

plot1 = plt.figure(1)
plt.plot(x, y1, label='Critical Accuracy')
plt.plot(x, y2, label='Warning Accuracy')
plt.plot(x, y3, label='Normal Accuracy')

plt.legend(loc='lower right')
# plt.tight_layout()


plot2 = plt.figure(2)
plt.plot(x, y4, label='Proactive Failure Recovery')
plt.plot(x, y5, label='Reactive Failure Recovery')


plt.legend(loc='lower right')
# plt.tight_layout()
# plt.tight_layout()

plot1.show()
plot2.show()
plt.show()

    





