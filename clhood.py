
# Constrained likelihood fit using constrained least squares as basis
# but replacing inverse covariance matrix by matrix of 2nd derivatives
# of likelihood function
# S. Kluth 12/2011

from clsq import clsqSolver, fivePointStencil, setH
from numpy import matrix, zeros


class clhoodSolver( clsqSolver ):

    # Ctor:
    def __init__( self, data, upar, lhoodfun, constraintfunction, 
                  largs=(), cargs=(), epsilon=0.0001,
                  maxiter=100, deltachisq=0.0001,
                  mparnames=None, uparnames=None,
                  title= "Constrained maximum likelihood" ):
        clsqSolver.__init__( self, data, None, upar,
                             constraintfunction, cargs, epsilon,
                             maxiter, deltachisq,
                             mparnames, uparnames ) 
        self.__mparv= self.getMparv()
        self.__covm= None
        self.__lhood= Likelihood( lhoodfun, largs )
        return

    # Override for inversion solution:
    def _prepareRhsv( self, dim, datadim, upardim, constrv ):
        rhsv= clsqSolver._prepareRhsv( self, dim, datadim, upardim, constrv )
        dldp= self.__lhood.firstDerivatives( self.__mparv )
        rhsv[:datadim]= - dldp
        return rhsv

    # Override for partition solution:
    def _prepareDeltapar( self, datadim, upardim, constrdim,
                         c11, c21, c31, c32, c33, constrv ):
        deltapar= clsqSolver._prepareDeltapar( self, datadim, upardim, constrdim,
                                               c11, c21, c31, c32, c33, constrv )
        dldp= self.__lhood.firstDerivatives( self.__mparv )
        deltapar[:datadim]-= c11*dldp
        deltapar[datadim:datadim+upardim]-= c21*dldp
        deltapar[datadim+upardim:]-= c31*dldp
        return deltapar

    # Overrides to get covariance and inverse:
    def getCovm( self ):
        dl2dp2= self.__lhood.secondDerivatives( self.__mparv )
        self.__invm= dl2dp2
        self.__covm= self.__invm.getI()
        return self.__covm
    def getInvm( self ):
        dl2dp2= self.__lhood.secondDerivatives( self.__mparv )
        self.__invm= dl2dp2
        return self.__invm

    # Overrides for printing:
    # def printTitle( self ):
    #     print  "\nConstrained maximum likelihood"
    #     return
    # def printFitParameters( self, chisq, ndof, ffmt ):
    #     fmtstr= "\nLikelihood= {0:"+ffmt+"}"
    #     print fmtstr.format( self.__lhood.value( self.__mparv ) )
    #     clsqSolver.printFitParameters( self, chisq, ndof, ffmt )
    #     return
    def getFitStats( self ):
        result= clsqSolver.getFitStats( self )
        result.update( { "lhood": self.__lhood.value( self.__mparv ) } ) 
        return result

# Class contains likelihood specific calculations:
class Likelihood:

    # Ctor, takes external likelihood function and its extra arguments:
    def __init__( self, fun, args=(), eps=1.0e-4 ):
        self.__lfun= fun
        self.__args= args
        self.__eps= eps
        return 

    # Return likelihood value given measured patameters:
    def value( self, mpar ):
        return self.__calculate( mpar )
    def __calculate( self, mpar ):
        return self.__lfun( mpar, *self.__args )

    # Calculate 1st derivatives of likelihood function:
    def firstDerivatives( self, mpar ):
        nmpar= len(mpar)
        h= matrix( zeros( shape=(nmpar,1) ) )
        dldp= matrix( zeros( shape=(nmpar,1) ) )
        for ipar in range( nmpar ):
            h[ipar]= setH( self.__eps, mpar[ipar] )
            dldp[ipar]= fivePointStencilWrapper( self.__calculate, mpar, h )
            h[ipar]= 0.0
        return dldp

    # Calculate 2nd derivatives of likelihood function:
    def secondDerivatives( self, mpar ):
        def calcd2ldp2( mpar, hi, hj ):
            def dldp( mpar ):
                return fivePointStencilWrapper( self.__calculate, mpar, hi )
            return fivePointStencilWrapper( dldp, mpar, hj )
        nmpar= len(mpar)
        hi= matrix( zeros( shape=(nmpar,1) ) )
        hj= matrix( zeros( shape=(nmpar,1) ) )
        d2ldp2= matrix( zeros( shape=(nmpar,nmpar) ) )
        for ipar in range( nmpar ):
            hi[ipar]= setH( self.__eps, mpar[ipar] )
            for jpar in range( ipar, nmpar ):
                hj[jpar]= setH( self.__eps, mpar[jpar] )
                d2ldp2[ipar,jpar]= calcd2ldp2( mpar, hi, hj )
                if jpar > ipar:
                    d2ldp2[jpar,ipar]= d2ldp2[ipar,jpar]
                hj[jpar]= 0.0
            hi[ipar]= 0.0
        return d2ldp2

# Helper for derivative calculation which drops one argument
# so we can reuse the fivePointStencil from clsq:
def fivePointStencilWrapper( function, varpar, h ):
    def funwrapper( varpar, fixpar ):
        return function( varpar )
    return fivePointStencil( funwrapper, varpar, h, None )

