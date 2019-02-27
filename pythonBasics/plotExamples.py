import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(start=0,
                stop=10,
                num=1e4)
#Create an function which is the combination of a polynomial and sinus functions with some random noise.
y = -1*x**2 + 3*x + 2* np.sin(10* x) + np.random.rand(x.size)

approxPolynom = np.polyfit(x,y,deg=2)
approxY = np.poly1d(approxPolynom)

#Plot both the function and the approximation to the line chart.
plt.plot(x,y)
plt.plot(x,approxY(x))
plt.savefig('approx.png')

del x,y,approxY,approxPolynom
plt.clf()
'''
Lets play with dices. Some games the sum of the dices should be 7. 
Simulate dice throws with Python and calculate the probability of that the sum is 7. 

http://www.netexl.com/howtoplay/sevens-out/
https://en.wikipedia.org/wiki/Sevens,_Elevens,_and_Doubles
'''
throws = np.array([])
NUMBER_OF_TRIES = 10000 #This value can be increased for more sophisticated simulations
throws = np.array([[np.random.randint(low=1,high=7),np.random.randint(low=1,high=7)] for i in range(NUMBER_OF_TRIES)])

#We can count the throws.
import collections
counter = collections.Counter(throws[:,0])

#It is better to present the throws on a histogram.
plt.hist(x=throws[:,0])
plt.ylabel("Count")
plt.xticks([i+1 for i in range(5)])
plt.title("Dice #1")
plt.show()
plt.clf()

plt.hist(x=throws[:,1])
plt.ylabel("Count")
plt.xticks([i+1 for i in range(5)])
plt.title("Dice #2")
plt.show()
'''
Histograms shows that each number occurs the same times. 
We can assume that the dices follow uniform distribution.
So our virtual dice behaves as a normal dice.

What about their sum. 
'''
sumOfThrow = throws[:,1] +throws[:,0]
plt.hist(x=sumOfThrow)
plt.ylabel("Count")
plt.title("Sum of Dices")
plt.show()
'''
OK It seems to be their sum is not follow uniform distribution.
Interesting. We may should have paid attention at Statistics class. :)

Test it wether the sum follows normal distribution or not.
'''
avgSumOfThrow = np.average(sumOfThrow)
stdSumOfThrow = np.std(sumOfThrow)

from scipy import stats
#ksResult =stats.kstest((sumOfThrow - avgSumOfThrow)/stdSumOfThrow,'norm')
ksResult = stats.kstest(sumOfThrow,'norm',args=(avgSumOfThrow,stdSumOfThrow), N=NUMBER_OF_TRIES)
KS_REJECTION_LEVEL = 0.05
if ksResult.pvalue < KS_REJECTION_LEVEL:
    print("Sum of Dices does not follow Normal Distribution")
else:
    print("Sum of Dices follows Normal Distribution")

'''
Central Limit Theorem
'''

