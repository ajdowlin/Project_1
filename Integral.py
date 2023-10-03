# This script shows two approaches to numerically solve a Gaussian Integral

import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
import mplhep


def Gaussian(x, sigma, mean): #Returns value at x of a Gaussian function of some sigma and mean
    return [np.exp(-((xi-mean)**2)/((sigma**2)*2)) for xi in x]

def ReimmanMethod(x, sigma, mean, method):
    vol_tot = 0
    step = float(x[1] - x[0])
    if method == 'Left':
        xL = x[:-1]
        yL = Gaussian(xL, sigma, mean)
        for yi in yL:
            vol_i = yi*step
            vol_tot += vol_i
    if method == 'Right':
        xR = x[1:]
        yR = Gaussian(xR, sigma, mean)
        for yi in yR:
            vol_i = yi*step
            vol_tot += vol_i
    if method == 'Middle':
        xM = x[1:]
        xM = [xi - step/2 for xi in xM]
        yM = Gaussian(xM, sigma, mean)
        for yi in yM:
            vol_i = yi*step
            vol_tot += vol_i

    return vol_tot
            
def TrapezoidalMethod(x, sigma, mean):
    vol_tot = 0
    step = x[1] - x[0]
    y = Gaussian(x, sigma, mean)
    vol_ends = step*(y[0] + y[-1])/2 # manually add in endpoints
    vol_tot += vol_ends
    for yi in y[1:-1]: # add rest of trapezoidal volume
        vol_i = yi*step
        vol_tot += vol_i

    return vol_tot

def SimpsonMethod(x, sigma, mean):
    vol_tot = 0
    step = x[1] - x[0]
    y = Gaussian(x, sigma, mean)
    vol_tot += (y[0]*step/3) + (y[-1]*step/3)
    for yi in y[:-2:2]:
        vol_i = (2*step/3)*yi
        vol_tot += vol_i
    for yi in y[1:-1:2]:
        vol_i = (4*step/3)*yi
        vol_tot += vol_i

    return vol_tot

def main():
    sigma_p = 10
    mean_p = 0
    sigma_1 = 1
    mean_1 = 0
    sigma_2 = 1
    mean_2 = 0
    
    start = -30
    stop = 30
    step = 0.01
   #  mplhep.style.use("LHCb2") # Used to generate format of plots in report
    fig, axes = plt.subplots()
    axes.set_xlabel("x", loc = 'center', fontsize = 15)
    axes.set_ylabel("y", loc = 'center', fontsize = 15)
    x = np.arange(start, stop, step)
    y = Gaussian(x, sigma_p, mean_p)
    axes.plot(x, y, color = 'blue')
    xfill = np.arange(5, stop, step)
    yfill = Gaussian(xfill, sigma_p, mean_p)
    plt.fill_between(xfill, yfill, step = 'pre', alpha = 0.4)
    fig.set_size_inches(10,7)
    plt.savefig("Plot1") # Plot that shows area needed to calculate for physics problem
    plt.show() 
    plt.clf()


    fig, axes = plt.subplots()
    axes.set_xlabel("x", loc = 'center', fontsize = 15)
    axes.set_ylabel("y", loc = 'center', fontsize = 15)
    x1 = np. arange(-3, 3, step)
    y1 = Gaussian(x1, sigma_1, mean_1)
    axes.plot(x1, y1, color = 'blue')
    xfill = np.arange(-sigma_1, sigma_1, step)
    yfill = Gaussian(xfill, sigma_1, mean_1)
    fig.set_size_inches(10,7)
    plt.fill_between(xfill, yfill, step = 'pre', alpha = 0.4)
    plt.savefig("Plot2")
    plt.show() # Plot that shows area needed to calculate first phenomena
    plt.clf()


    fig, axes = plt.subplots()
    axes.set_xlabel("x", loc = 'center', fontsize = 15)
    axes.set_ylabel("y", loc = 'center', fontsize = 15)
    axes.plot(x1, y1, color = 'blue')
    fig.set_size_inches(10,7)
    plt.fill_between(x1, y1, step = 'pre', alpha = 0.4)
    plt.savefig("Plot3")
    plt.show() # Plot that shows area needed to calculate second phenomena
    plt.clf()

    # now want to look at dependence of solutions on how many rectangles chosen in Riemman sum
    
    nRec = np.arange(start = 5, stop = 100, step = 2)
    Rvol_p = []
    Lvol_p = []
    Mvol_p = []
    Tvol_p = []
    Svol_p = []

    Rvol_1 = []
    Lvol_1 = []
    Mvol_1 = []
    Tvol_1 = []
    Svol_1 = []

    Rvol_2 = []
    Lvol_2 = []
    Mvol_2 = []
    Tvol_2 = []
    Svol_2 = []
    
    for n in nRec:
        xp = np.arange(5, 10, 5/n)
        Rvol_p.append(ReimmanMethod(xp, sigma_p, mean_p, method = 'Right') *1000/((2*np.pi)**0.5))
        Lvol_p.append(ReimmanMethod(xp, sigma_p, mean_p, method = 'Left')*1000/((2*np.pi)**0.5))
        Mvol_p.append(ReimmanMethod(xp, sigma_p, mean_p, method = 'Middle')*1000/((2*np.pi)**0.5))
        Tvol_p.append(TrapezoidalMethod(xp, sigma_p, mean_p)*1000/((2*np.pi)**0.5))
        Svol_p.append(SimpsonMethod(xp, sigma_p, mean_p)*1000/((2*np.pi)**0.5))

        x1 = np.arange(-sigma_1, sigma_1, 2*sigma_1/n)
        Rvol_1.append(ReimmanMethod(x1, sigma_1, mean_1, method = 'Right')/((2*np.pi)**0.5))
        Lvol_1.append(ReimmanMethod(x1, sigma_1, mean_1, method = 'Left')/((2*np.pi)**0.5))
        Mvol_1.append(ReimmanMethod(x1, sigma_1, mean_1, method = 'Middle')/((2*np.pi)**0.5))
        Tvol_1.append(TrapezoidalMethod(x1, sigma_1, mean_1)/((2*np.pi)**0.5))
        Svol_1.append(SimpsonMethod(x1, sigma_1, mean_1)/((2*np.pi)**0.5))

        x2 = np.arange(-10, 10, 20/n)
        Rvol_2.append(ReimmanMethod(x2, sigma_2, mean_2, method = 'Right'))
        Lvol_2.append(ReimmanMethod(x2, sigma_2, mean_2, method = 'Left'))
        Mvol_2.append(ReimmanMethod(x2, sigma_2, mean_2, method = 'Middle'))
        Tvol_2.append(TrapezoidalMethod(x2, sigma_2, mean_2))
        Svol_2.append(SimpsonMethod(x2, sigma_2, mean_2))


    print('phy')
    print(Rvol_p[-1])
    print(Lvol_p[-1])
    print(Mvol_p[-1])
    print(Tvol_p[-1])
    print(Svol_p[-1])
    print('phen1')
    print(Rvol_1[-1])
    print(Lvol_1[-1])
    print(Mvol_1[-1])
    print(Tvol_1[-1])
    print(Svol_1[-1])
    print('phen2')
    print(Rvol_2[-1])
    print(Lvol_2[-1])
    print(Mvol_2[-1])
    print(Tvol_2[-1])
    print(Svol_2[-1])

    
    fig, axes = plt.subplots()
    axes.set_xlabel("n", loc = 'center', fontsize = 15)
    axes.set_ylabel("Integral Value", loc = 'center', fontsize = 15)
    axes.plot(nRec, Rvol_p, color = 'blue', label = 'Reimman Right')
    axes.plot(nRec, Lvol_p, color = 'red', label = 'Reimman Left')
    axes.plot(nRec, Mvol_p, color = 'green', label = 'Reimman Middle')
    axes.plot(nRec, Tvol_p, color = 'orange', label = 'Trapezoidal')
    axes.plot(nRec, Svol_p, color = 'purple', label = 'Simpson')
    axes.legend()
    fig.set_size_inches(10,7)
    plt.savefig("Plot4")
    plt.show() # Plot that shows area needed to calculate second phenomena
    plt.clf()

    fig, axes = plt.subplots()
    axes.set_xlabel("n", loc = 'center', fontsize = 15)
    axes.set_ylabel("Integral Value", loc = 'center', fontsize = 15)
    axes.plot(nRec, Rvol_1, color = 'blue', label = 'Reimman Right')
    axes.plot(nRec, Lvol_1, color = 'red', label = 'Reimman Left')
    axes.plot(nRec, Mvol_1, color = 'green', label = 'Reimman Middle')
    axes.plot(nRec, Tvol_1, color = 'orange', label = 'Trapezoidal')
    axes.plot(nRec, Svol_1, color = 'purple', label = 'Simpson')
    axes.legend()
    fig.set_size_inches(10,7)
    plt.savefig("Plot5")
    plt.show() # Plot that shows area needed to calculate second phenomena
    plt.clf()

    fig, axes = plt.subplots()
    axes.set_xlabel("n", loc = 'center', fontsize = 15)
    axes.set_ylabel("Integral Value", loc = 'center', fontsize = 15)
    axes.plot(nRec, Rvol_2, color = 'blue', label = 'Reimman Right')
    axes.plot(nRec, Lvol_2, color = 'red', label = 'Reimman Left')
    axes.plot(nRec, Mvol_2, color = 'green', label = 'Reimman Middle')
    axes.plot(nRec, Tvol_2, color = 'orange', label = 'Trapezoidal')
    axes.plot(nRec, Svol_2, color = 'purple', label = 'Simpson')
    axes.legend()
    fig.set_size_inches(10,7)
    plt.savefig("Plot6")
    plt.show() # Plot that shows area needed to calculate second phenomena
    plt.clf()

    
    
    
if __name__== "__main__":
    main()
    

