import  pandas as pd
import matplotlib.pyplot as plt
import scipy.stats

df = pd.read_csv('data/dataset.csv')
'''
1) Records where the product is not waste (there is no error)
2) How many waste are in the dataset?
3) What is the expected Length of a product?
4) What is the variance of the Width?
5) Draw the histogram of Weight, Length, Width, Height. 
6) Does these attributes follow normal distribution?
7) Test whether they follow normal distribution or not with Kolmogorov-Smirov Test
8) Calculate their correlation matrix.
 
'''

print(df[df.Error == False])
print("There are %d waste in the dataset" %df[df.Error == True].shape[0])
print("Expected Length of a product is %lf" % df.Length.mean())
print("Variance of Width is %lf" % df.Width.var())

plt.hist(df.Weight)
plt.title("PDF of Weight")
plt.show()
plt.clf()
plt.hist(df.Weight, cumulative=True)
plt.title("CDF of Weight")
plt.show()

ksResult = scipy.stats.kstest(
    df.Weight,'norm',args=(
        df.Weight.mean(),
        df.Weight.std()),
    N=df.Weight.size)