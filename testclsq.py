#!/usr/bin/python

# unit tests for iterative constrained least squares
# S. Kluth 11/2011

import unittest

from numpy import matrix, zeros

import clsq

class dodo:

    def __init__( self ):
        self.xabs= [ 1.0, 2.0, 3.0, 4.0, 5.0 ]
        self.data= [ 1.1, 1.9, 2.9, 4.1, 5.1 ]
        self.errors= [ 0.1, 0.1, 0.1, 0.1, 0.1 ]
        self.upar= [ 0.1, 1.1 ]
        self.covm= clsq.covmFromErrors( self.errors )
        self.mparv= clsq.columnMatrixFromList( self.data )
        self.datav= clsq.columnMatrixFromList( self.data )
        self.uparv= clsq.columnMatrixFromList( self.upar )
        return


def linearConstrFun( mpar, upar, xv ):
    constraints= []
    for mparval, xval in zip( mpar, xv ):
        constraints.append( upar[0]+upar[1]*xval - mparval )
    return constraints


class constraintsTest( unittest.TestCase ):

    def setUp( self ):
        self.__dodo= dodo()
        self.__constraints= clsq.Constraints( linearConstrFun, 
                                              args=(self.__dodo.xabs,) )
        return

    def test_calculate( self ):
        mpar= self.__dodo.data
        upar= self.__dodo.upar
        constraints= self.__constraints.calculate( mpar, upar )
        constraints= constraints.ravel().tolist()[0]
        expectedConstraints= [ 0.1, 0.4, 0.5, 0.4, 0.5 ]
        for constraint, expectedConstraint in zip( constraints, 
                                                   expectedConstraints ):
            self.assertAlmostEqual( constraint, expectedConstraint )
        return

    def test_derivativeM( self ):
        dfdmpm= self.__constraints.derivativeM( self.__dodo.mparv,
                                                self.__dodo.uparv )
        dfdmplist= dfdmpm.ravel().tolist()[0]
        expecteddfdmplist= [ -1.,  0.,  0.,  0.,  0.,
                              0., -1.,  0.,  0.,  0.,
                              0.,  0., -1.,  0.,  0.,
                              0.,  0.,  0., -1.,  0.,
                              0.,  0.,  0.,  0., -1. ]
        for dfdmp, expecteddfdmp in zip( dfdmplist, expecteddfdmplist ):
             self.assertAlmostEqual( dfdmp, expecteddfdmp )
        return

    def test_derivativeU( self ):
        dfdupm= self.__constraints.derivativeU( self.__dodo.mparv, 
                                                self.__dodo.uparv )
        dfduplist= dfdupm.ravel().tolist()[0]
        expecteddfduplist= [ 1., 1.,
                             1., 2.,
                             1., 3.,
                             1., 4.,
                             1., 5. ]
        for dfdup, expecteddfdup in zip( dfduplist, expecteddfduplist ):
            self.assertAlmostEqual( dfdup, expecteddfdup )
        return

class clsqSolverTest( unittest.TestCase ):

    def setUp( self ):
        self.__dodo= dodo()
        self.__solver= clsq.clsqSolver( self.__dodo.data,
                                        self.__dodo.covm,
                                        self.__dodo.upar,
                                        linearConstrFun,
                                        args=(self.__dodo.xabs,) )
        return

    def test_clsqSolverInversion( self ):
        self.__solver.solve()
        self.__checkSolution()
        return

    def test_clsqSolverPartition( self ):
        self.__solver.solve( lBlobel=True )
        self.__checkSolution()
        return

    def __checkSolution( self ):
        mpar= self.__solver.getMpar()
        upar= self.__solver.getUpar()
        expectedmpar= [ 0.98, 2.0, 3.02, 4.04, 5.06 ]
        for par, expectedpar in zip( mpar, expectedmpar ):
            self.assertAlmostEqual( par, expectedpar )
        expectedupar= [ -0.04, 1.02 ]
        for par, expectedpar in zip( upar, expectedupar ):
            self.assertAlmostEqual( par, expectedpar )
        return


if __name__ == '__main__':
    suite1= unittest.TestLoader().loadTestsFromTestCase( constraintsTest )
    suite2= unittest.TestLoader().loadTestsFromTestCase( clsqSolverTest )
    for suite in [ suite1, suite2 ]:
        unittest.TextTestRunner( verbosity=2 ).run( suite )

