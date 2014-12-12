
# Example showing a fit with poisson likelihood
# see Blobels Terascale Analysis Center lecture 20 Jan 2010, pg. 33

from ConstrainedFit import clhood
from ConstrainedFit import clsq

from scipy.special import gammaln
from scipy.stats import poisson, norm
from math import log

# Fit with poisson likelihood:
def run( lBlobel=False ):
    
    # Data, two counts, errors not needed:
    data= [ 9.0, 16.0 ]
    # errors= [ 3.0, 4.0 ]

    # Fit variable is parameter of poisson distribution:
    upar= [ 12.0 ]

    # Likelihood is sum of log(poisson) for each data point:
    def lfun( mpar ):
        result= 0.0
        for datum, parval in zip( data, mpar ):
            parval= parval.item()
            result-= log( poisson.pmf( datum, parval ) )
            # Calculated log(poisson):
            # result-= datum*log( parval ) - gammaln( datum+1.0 ) - parval
        return result

    # Constraints force poisson distribution with same parameter
    # for every data point:
    def constrFun( mpar, upar ):
        return [ mpar[0] - upar[0], 
                 mpar[1] - upar[0] ]

    solver= clhood.clhoodSolver( data, upar, lfun, constrFun )
    print "Constraints before solution"
    print solver.getConstraints()
    solver.solve( lBlobel=lBlobel, lpr=True )
    solver.printResults( corr=True )

    return

# Linear fit with Gauss (normal) likelihood, and with clsq
# for comparison, expect identical results:

def rung( lBlobel=False ):

    # Data and errors:
    xabs= [ 1.0, 2.0, 3.0, 4.0, 5.0 ]
    data= [ 1.1, 1.9, 2.9, 4.1, 5.1 ]
    errors= [ 0.1, 0.1, 0.1, 0.1, 0.1 ]

    # Linear function (straight line) parameters:
    upar= [ 0.0, 1.0 ]

    # Likelihood is sum of log(Gauss) for each data point:
    def lfun( mpar ):
        result= 0.0
        for datum, parval, error in zip( data, mpar, errors ):
            parval= parval.item()
            result-= log( norm.pdf( datum, parval, error ) )
            # result+= 0.5*((datum-parval)/error)**2
        return result

    # Constraints force linear function for each data point:
    def constrFun( mpar, upar ):
        constraints= []
        for xval, parval in zip( xabs, mpar ):
            constraints.append( upar[0] + upar[1]*xval - parval )
        return constraints

    # Solution using constrained log(likelihood) minimisation:
    print "\nMax likelihood constrained fit"
    solver= clhood.clhoodSolver( data, upar, lfun, constrFun )
    print "Constraints before solution"
    print solver.getConstraints()
    solver.solve( lBlobel=lBlobel, lpr=True )
    solver.printResults( corr=True )

    # Solution using constrained least squares:
    print "\nLeast squares constrained fit"
    covm= clsq.covmFromErrors( errors )
    solver= clsq.clsqSolver( data, covm, upar, constrFun )
    print "Constraints before solution"
    print solver.getConstraints()
    solver.solve( lBlobel=lBlobel )
    solver.printResults( corr=True )

    return

