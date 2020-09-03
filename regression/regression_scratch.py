from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
import random

def create_dataset(n,variance,step=2,correlation=False):
    val = 1
    ys = []
    for i in range(n):
        y = val + random.randrange(-variance,variance)
        ys.append(y)
        if correlation and correlation== "pos":
            val += step
        elif correlation and correlation=="neg":
            val -= step
    xs = [i for i in range(len(ys))]
    return np.array(xs,dtype=np.float64) , np.array(ys , dtype=np.float64)

def best_fit_slope(xs,ys):
    m = ( ((mean(xs)*mean(ys)) - mean(xs*ys)) /
        (mean(xs)*mean(xs) - mean(xs*xs) )  )
    return m

def best_intercept(slope,xs,ys):
    b = mean(ys) - m*mean(xs)
    return b

def squared_error(ys_orig,ys_line):
    return sum((ys_line-ys_orig)**2)

def coefficient_of_determination(ys_orig,ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    squared_error_regr = squared_error(ys_orig,ys_line)
    squared_error_mean = squared_error(ys_orig,y_mean_line)
    return 1 - (squared_error_regr/squared_error_mean)


xs , ys = create_dataset(40 , 20 , 2 , correlation="neg")

m = best_fit_slope(xs,ys)
b = best_intercept(m , xs, ys)

regression_line = [(m*x)+b for x in xs]

predic_x = 8
predic_y = m*predic_x + b

r_squared = coefficient_of_determination(ys , regression_line)
print(r_squared)


plt.scatter(xs , ys)
plt.scatter(predic_x,predic_y , s = 100)
plt.plot(xs,regression_line)
plt.show()