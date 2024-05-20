from gurobipy import *

# Data
# FLInstanceGenerator(n, T): A class to generate random instances, and construct 
# FLFullInstance: Passive struct to hold an entire offline instance.
# FLInstanceDistributor(FLFullInstance): A class to hold an entire offline instance, distribute it to offline algorithms (return it entirely), and distribute it to online algorithms (return it partially / incrementally).

# Full Instance description (necessary data held in FLFullInstance):
# n: dimension of points (int)
# T: total number of time periods (int)
# x_0, x_1, ..., x_T: points in R^n 
# Gamma: model parameter (float)

# MIP
# Problem:
# input: x_0, x_1, ..., x_T, n, Gamma, (T) (T is total time periods oracle known)


# Competitive Algorithm
# Problem: 
# input: x_0, x_1, ..., t, n, Gamma, (t) (t is current time)

def main():
    print("Hello World!")

if __name__ == '__main__':
    main()
