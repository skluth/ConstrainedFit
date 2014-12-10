
from ConstrainedFit import clhood
from ConstrainedFit import clsq

from scipy.special import gammaln
from math import log
from scipy.stats import poisson

def run( lBlobel=False ):
    
    data= [ 9.0, 16.0 ]
    errors= [ 3.0, 4.0 ]

    upar= [ 12.0 ]

    def lfun( mpar ):
        result= 0.0
        for datum, parval in zip( data, mpar ):
            parval= parval.item()
            # Likelihood is Poisson for every data point
            result-= log( poisson.pmf( datum, parval ) )
            # Calculated log(poisson):
            # result-= datum*log( parval ) - gammaln( datum+1.0 ) - parval
        return result

    def constrFun( mpar, upar ):
        return [ mpar[0] - upar[0], 
                 mpar[1] - upar[0] ]

    solver= clhood.clhoodSolver( data, upar, lfun, constrFun )
    print "Constraints before solution"
    print solver.getConstraints()
    solver.solve( lBlobel=lBlobel, lpr=True )
    solver.printResults( corr=True )

    return

def rung( lBlobel=False ):

    xabs= [ 1.0, 2.0, 3.0, 4.0, 5.0 ]
    data= [ 1.1, 1.9, 2.9, 4.1, 5.1 ]
    errors= [ 0.1, 0.1, 0.1, 0.1, 0.1 ]
    
    upar= [ 0.0, 1.0 ]

    def lfun( mpar ):
        result= 0.0
        for datum, parval, error in zip( data, mpar, errors ):
            # result-= TMath.Log( TMath.Gaus( datum, parval, 
            #                                 error, True ) )
            result+= 0.5*((datum-parval)/error)**2
        return result

    def constrFun( mpar, upar ):
        constraints= []
        for xval, parval in zip( xabs, mpar ):
            constraints.append( upar[0] + upar[1]*xval - parval )
        return constraints

    print "\nMax likelihood constrained fit"
    solver= clhood.clhoodSolver( data, upar, lfun, constrFun )
    print "Constraints before solution"
    print solver.getConstraints()
    solver.solve( lBlobel=lBlobel, lpr=True )
    solver.printResults( corr=True )

    print "\nLeast squares constrained fit"
    covm= clsq.covmFromErrors( errors )
    solver= clsq.clsqSolver( data, covm, upar, constrFun )
    print "Constraints before solution"
    print solver.getConstraints()
    solver.solve( lBlobel=lBlobel )
    solver.printResults( corr=True )

    return

