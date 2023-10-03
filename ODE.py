# This script shows two approaches to numerically solving the Bessel Differential Equation

import numpy as np
from scipy.special import jv, yn, gamma, digamma
import math
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
import mplhep
import os

# Euler's Method for solving the Bessel Differential Equation of order n




# The Bessel Eqn is broken into a series of 1st Order ODE's to apply Euler's Method

def F(xi, z1, z2, n):
    return (-z2/xi) - ((((xi**2)-(n**2))/(xi**2))*z1)

def EulerMethod(n, dx, x0, y0, yprime0, xf):

    x = np.arange(start = x0, stop = xf, step = dx) # list of all x values to be calculated for
    z1 = np.zeros(len(x)) # create 'blank' lists of same length of x to represent 1st order ODE's
    z2 = np.zeros(len(x))
    
    z1[0] = y0 # initial condition for y
    z2[0] = yprime0 # intial condition for y'

    for i in range(len(x)-1):
        xi = x[i]
        z1[i+1] = z1[i] + dx*z2[i]
        z2[i+1] = z2[i] + dx*F(xi, z1[i], z2[i], n)


    return x, z1, z2

# Similarly, the Bessel Eqn is broken into a series of 1st Order ODE's to apply the Runge-Kutta Method
def RungeKutta(n, dx, x0, y0, yprime0, xf):
    
    x = np.arange(start = x0, stop = xf, step = dx)
    z1 = np.zeros(len(x)) # create 'blank' lists of same length of x to represent 1st order ODE's
    z2 = np.zeros(len(x))

    z1[0] = y0 # initial condition for y
    z2[0] = yprime0 # intial condition for y'

    for i in range(len(x)-1):
        xi = x[i]
        z1i = z1[i]
        z2i = z2[i]

        m1 = dx*z2i
        k1 = dx*F(xi, z1i, z2i, n)

        m2 = dx*(z2i + 0.5*k1)
        k2 = dx*F(xi + 0.5*dx, z1i + 0.5*m1, z2i + 0.5*k1, n)

        m3 = dx*(z2i + 0.5*k1)
        k3 = dx*F(xi+ 0.5*dx, z1i + 0.5*m2, z2i + 0.5*k2, n)

        m4 = dx*(z2i + k3)
        k4 = dx*F(xi + dx, z1i + m3, z2i + k3, n)

        z1[i+1] = z1[i] + (m1 + 2*m2 + 2*m3 + m4)/6
        z2[i+1] = z2[i] + (k1 + 2*k2 + 2*k3 +k4)/6

    return x, z1, z2
        


# Frobenius method solution approximation for Bessel function of 1st kind,  of some order, keeping some number of terms
def JBessel(order, terms, x):
    y = []
    for xi in x:
        yi = 0
        for m in range(terms):
            const = (((-1)**m)/(gamma(m+1)*gamma(m + order+1)*(2**(2*m + order))))
            yi = yi + const*(xi**(2*m + order))
        y.append(yi)
    return y

# Frobenius method solution approximation for Bessel function of 2nd kind, of some order, keeping some number of terms expansion

def YBessel_nonInt(order, terms, x): #for Bessel functions of 2nd kind of non-integer order
    y = []
    J_pos = JBessel(order, terms, x)
    J_neg = JBessel(-order, terms, x)
    for J_pos_val, J_neg_val, xi in zip(J_pos, J_neg, x):
        yi = ((J_pos_val*np.cos(order*np.pi)) - J_neg_val)/np.sin(order*np.pi)
        y.append(yi)
    return y

def YBessel(order, terms, x): #for Bessel functions of 2nd kind, of some integer order, keeping some number of terms in expansion
    y = []
    J_pos = JBessel(order, terms, x)
    for xi, J_pos_val in zip(x, J_pos):
        yi = 0
        term1 = (2/np.pi)*np.log(xi/2)*J_pos_val
        term2 = 0
        term3 = 0

        for m in range(terms):
            term3_i = ((-1/np.pi)*((-1)**m)*(digamma(m + order + 1)+digamma(m+1)))/(gamma(m+1)*gamma(m + order + 1)*(2**(2*m + order)))*(xi**(2*m + order))
            term3 = term3 + term3_i

        for m in range(order):
            term2_i = (-1/np.pi)*gamma(m + order)*(xi**(2*m - order))/(gamma(m+1)*(2**(2*m - order)))
            term2 = term2 + term2_i
        yi = term1 + term2 + term3
        y.append(yi)
    return y

def main():
    order = 1
    start = 1.8412
    stop = 25
    step = 0.1
    y0 = 0.5819
    yprime0 = 0
    xEuler, yEuler, yEulerprime = EulerMethod(order, step, start, y0, yprime0, stop)
    xRunge, yRunge, yRungeprime = RungeKutta(order, step, start, y0, yprime0, stop)
    xAnalytic = np.arange(0, stop, step)
    yAnalytic1 = JBessel(order, 50, xAnalytic)
    #yAnalytic1 = jv(order, xAnalytic)
    #yAnalytic2 = YBessel(order, 50, xAnalytic)
    #yAnalytic2 = yn(order, xAnalytic)
    
    
    # plot Analytic, Euler method and Runge Kutta methods using a step of 0.1
    
    #mplhep.style.use("LHCb2")  #Used to generate plot format in report

    
    fig, axes = plt.subplots()
    axes.set_xlabel("x", loc = 'center', fontsize = 15)
    axes.set_ylabel("y", loc = 'center', fontsize = 15)
    axes.plot(xEuler, yEuler, color = 'blue', label = 'Euler')
    axes.plot(xRunge, yRunge, color = 'red', label = 'Runge Kutta')
    axes.plot(xAnalytic, yAnalytic1, color = 'green', label = 'Analytic')
    #axes.plot(xAnalytic, yAnalytic2, color = 'green')
    axes.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0) 
    fig.set_size_inches(10,7)
    plt.savefig("Plot7")
    plt.show()
    plt.clf()

    # plot Analytic and Runge Kutta methods using a step of 0.5

    order = 1
    start = 1.8412
    stop = 25
    step = 0.5
    y0 = 0.5819
    yprime0 = 0
    xEuler, yEuler, yEulerprime = EulerMethod(order, step, start, y0, yprime0, stop)
    xRunge, yRunge, yRungeprime = RungeKutta(order, step, start, y0, yprime0, stop)
    xAnalytic = np.arange(0, stop, step)
    yAnalytic1 = JBessel(order, 50, xAnalytic)
    #yAnalytic1 = jv(order, xAnalytic)
    #yAnalytic2 = YBessel(order, 50, xAnalytic)
    #yAnalytic2 = yn(order, xAnalytic)

    fig, axes = plt.subplots()
    axes.set_xlabel("x", loc = 'center', fontsize = 15)
    axes.set_ylabel("y", loc = 'center', fontsize = 15)
    #axes.plot(xEuler, yEuler, color = 'blue', label = 'Euler')
    axes.plot(xRunge, yRunge, color = 'red', label = 'Runge Kutta')
    axes.plot(xAnalytic, yAnalytic1, color = 'green', label = 'Analytic')
    #axes.plot(xAnalytic, yAnalytic2, color = 'green')
    axes.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    fig.set_size_inches(10,7)
    plt.savefig("Plot8")
    plt.show()
    plt.clf()



    # Phenomena 1 verify

    order = 1
    start = 1.8412
    stop = 25
    step = 0.5
    y0 = 0.5819
    yprime0 = 0
    xEuler, yEuler, yEulerprime = EulerMethod(order, step, start, y0, yprime0, stop)
    xRunge, yRunge, yRungeprime = RungeKutta(order, step, start, y0, yprime0, stop)
    xAnalytic = np.arange(0, stop, step)
    yAnalytic1 = JBessel(-order, 50, xAnalytic)
    #yAnalytic1 = jv(order, xAnalytic)
    #yAnalytic2 = YBessel(order, 50, xAnalytic)
    #yAnalytic2 = yn(order, xAnalytic)

    
    fig, axes = plt.subplots()
    axes.set_xlabel("x", loc = 'center', fontsize = 15)
    axes.set_ylabel("y", loc = 'center', fontsize = 15)
    #axes.plot(xEuler, yEuler, color = 'blue', label = 'Euler')
    axes.plot(xRunge, yRunge, color = 'red', label = 'Runge Kutta order 1')
    axes.plot(xAnalytic, yAnalytic1, color = 'green', label = 'Analytic order -1')
    #axes.plot(xAnalytic, yAnalytic2, color = 'green')
    axes.legend(fontsize = 20)
    fig.set_size_inches(10,7)
    plt.savefig("Plot9")
    plt.show()
    plt.clf()


 

    #Phenomena 2 verify

    order = 2
    start = 3.0542
    stop = 25
    step = 0.1
    y0 = 0.4864
    yprime0 = 0
    #xEuler, yEuler, yEulerprime = EulerMethod(order, step, start, y0, yprime0, stop)
    xRunge, yRunge, yRungeprime = RungeKutta(order, step, start, y0, yprime0, stop)
    xAnalytic = np.arange(0, stop, step)
    #J_1 = JBessel(1, 50, xAnalytic)
    J_1 = jv(1, xAnalytic)
    #J_3 = JBessel(3, 50, xAnalytic)
    J_3 = jv(3, xAnalytic)
    yAnalytic = [0.5*(j1-j3) for j1, j3 in zip(J_1, J_3)]

    
    fig, axes = plt.subplots()
    axes.set_xlabel("x", loc = 'center', fontsize = 15)
    axes.set_ylabel("y", loc = 'center', fontsize = 15)
    #axes.plot(xEuler, yEuler, color = 'blue', label = 'Euler')
    axes.plot(xRunge, yRungeprime, color = 'red', label = 'Runge Kutta order 2 (derivative)')
    axes.plot(xAnalytic, yAnalytic, color = 'green', label = 'Analytic')
    #axes.plot(xAnalytic, yAnalytic2, color = 'green')
    axes.legend(fontsize = 20)
    fig.set_size_inches(10,7)
    plt.savefig("Plot10")
    plt.show()
    plt.clf()


    
    

if __name__== "__main__":
    main()
    

    
    
    

        
