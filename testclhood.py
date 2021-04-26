#!/usr/bin/env python3

# unit tests for iterative constrained least squares
# S. Kluth 11/2011

import unittest

from numpy import matrix
from scipy.stats import norm
from math import log

from clsq import columnMatrixFromList

import clhood

class dodo:

    def __init__( self ):
        self.xabs= [ 1.0, 2.0, 3.0, 4.0, 5.0 ]
        self.data= [ 1.1, 1.9, 2.9, 4.1, 5.1 ]
        self.errors= [ 0.1, 0.1, 0.1, 0.1, 0.1 ]
        self.upar= [ 0.1, 1.1 ]
        return


def linearConstrFun( mpar, upar, xv ):
    constraints= []
    for mparval, xval in zip( mpar, xv ):
        constraints.append( upar[0]+upar[1]*xval - mparval )
    return constraints

def lhoodfun( mpar, data, errors ):
    result= 0.0
    for datum, parval, error in zip( data, mpar, errors ):
        parval= parval.item()
#        result-= log( norm.pdf( datum, parval, error ) )
        result+= 0.5*((datum-parval)/error)**2
    return result


class clhoodSolverTest( unittest.TestCase ):

    def setUp( self ):
        self.__dodo= dodo()
        self.__solver= clhood.clhoodSolver( self.__dodo.data,
                                            self.__dodo.upar,
                                            lhoodfun,
                                            linearConstrFun,
                                            largs=(self.__dodo.data,
                                                   self.__dodo.errors),
                                            cargs=(self.__dodo.xabs,) )
        return

    def test_clhoodSolverInversion( self ):
        self.__solver.solve()
        self.__checkSolution()
        return

    def test_clhoodSolverPartition( self ):
        self.__solver.solve( lBlobel=True )
        self.__checkSolution()
        return

    def __checkSolution( self ):
        mpars= self.__solver.getMpars()
        upars= self.__solver.getUpars()
        expectedmpars= [ 0.98, 2.0, 3.02, 4.04, 5.06 ]
        for par, expectedpar in zip( mpars, expectedmpars ):
            self.assertAlmostEqual( par, expectedpar )
        expectedupars= [ -0.04, 1.02 ]
        for par, expectedpar in zip( upars, expectedupars ):
            self.assertAlmostEqual( par, expectedpar )
        return


class LikelihoodTest( unittest.TestCase ):

    def setUp( self ):
        def fun( mpar ):
            return mpar[0] + 2.0*mpar[1]**2 + 3.0*mpar[2]**3
        self.__ll= clhood.Likelihood( fun )
        mpar= [ 1.0, 2.0, 3.0 ]
        self.__mparv= columnMatrixFromList( mpar )
        return

    def test_value( self ):
        val= self.__ll.value( self.__mparv )
        expectedval= 90.0
        self.assertAlmostEqual( val, expectedval )
        return

    def test_firstDerivatives( self ):
        v= self.__ll.firstDerivatives( self.__mparv )
        derivatives= v.ravel().tolist()[0]
        expectedderivatives= [ 1.0, 8.0, 81.0 ]
        for derivative, expectedderivative in zip( derivatives, 
                                                   expectedderivatives ):
            self.assertAlmostEqual( derivative, expectedderivative )
        return

    def test_secondDerivative( self ):
        m= self.__ll.secondDerivatives( self.__mparv )
        derivatives= m.ravel().tolist()[0]
        expectedderivatives= [ 0.0, 0.0, 0.0, 
                               0.0, 4.0, 0.0, 
                               0.0, 0.0, 54.0 ]
        for derivative, expectedderivative in zip( derivatives, 
                                                   expectedderivatives ):
            self.assertAlmostEqual( derivative, expectedderivative, places=6 )
        return


if __name__ == '__main__':
    suite1= unittest.TestLoader().loadTestsFromTestCase( clhoodSolverTest )
    suite2= unittest.TestLoader().loadTestsFromTestCase( LikelihoodTest )
    for suite in [ suite1, suite2 ]:
        unittest.TextTestRunner( verbosity=2 ).run( suite )

