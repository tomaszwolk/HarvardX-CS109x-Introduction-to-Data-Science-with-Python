import numpy
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression

# The below helper function, 'plot_boundary' plots the boundary for a given logistic regression function

# The details are not important, but you are encouraged to review it after the exercise

def plot_boundary(x, y, model, title, ax, plot_data=True, fill=True, color='Greens',degree=0):
    
    
    if plot_data:
        # PLOT DATA
        ax.scatter(x[y==1,0], x[y==1,1], c='green')
        ax.scatter(x[y==0,0], x[y==0,1], c='brown')
    
    # CREATE MESH
    interval = numpy.arange(min(x.min(), y.min()),max(x.max(), y.max()),0.05)
    n = numpy.size(interval)
    x1, x2 = numpy.meshgrid(interval, interval)
    x1 = x1.reshape(-1,1)
    x2 = x2.reshape(-1,1)
    xx = numpy.concatenate((x1, x2), axis=1)

    # PREDICT ON MESH POINTS
    xxpoly = PolynomialFeatures(degree).fit_transform(xx)
    yy = model.predict(xxpoly)    
    yy = yy.reshape((n, n))

    # PLOT DECISION SURFACE
    x1 = x1.reshape(n, n)
    x2 = x2.reshape(n, n)
    if fill:
        ax.contourf(x1, x2, yy, alpha=0.5, cmap=color)
    else:
        ax.contour(x1, x2, yy, alpha=0.5, cmap=color)
    
    # LABEL AXIS, TITLE
    ax.set_title(title)
    ax.set_xlabel('Latitude')
    ax.set_ylabel('Longitude')
    
    
    return ax


# The helper function below, fits a Logistic Regression model, and plots the boundary around it using the function above

def fit_and_plot_dt(x, y, c, title, ax, plot_data=True, fill=True, color='Blues',degree=0):

    lreg = LogisticRegression(C=c, max_iter=6000)

    x1 = PolynomialFeatures(degree).fit_transform(x)
    lreg.fit(x1, y)

    # PLOT DECISION TREE BOUNDARY
    ax = plot_boundary(x, y, lreg, title, ax, plot_data, fill, color,degree=degree)
    
    return ax
