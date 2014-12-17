
# Constrained least squares fit implementing CERN 60-30, and
# additions from Blobel
# S. Kluth 12/2011

from numpy import matrix, zeros, set_printoptions, linalg, delete
from scipy.optimize import brentq
from scipy.stats import chisqprob
from math import sqrt

# Helper functions to prepare inputs:
def columnMatrixFromList( listin ):
    dim= len( listin )
    columnmatrix= matrix( zeros(shape=(dim,1)) )
    for i in range( dim ):
        columnmatrix[i]= listin[i]
    return columnmatrix
def covmFromErrors( errors ):
    covm= []
    n= len( errors )
    for i in range( n ):
        row= n*[0]
        row[i]= errors[i]**2
        covm.append( row )
    return covm

# Main class to run constrained least squares fit:
class clsqSolver:

    # Ctor:
    def __init__( self, data, covm, upar, 
                  constraintfunction, args=(), epsilon=0.0001,
                  maxiter= 100, deltachisq=0.0001,
                  mparnames=None, uparnames=None,
                  ndof=None ):
        self.__constraints= Constraints( constraintfunction, args, epsilon )
        self.__datav= columnMatrixFromList( data )
        self.__mparv= self.__datav.copy()
        self.__uparv= columnMatrixFromList( upar )
        self.__covm= matrix( covm )
        self.__invm= None
        self.__pm= None
        self.__pminv= None
        self.__niterations= 0
        self.__uparnames= self.__setParNames( uparnames, len(self.__uparv) )
        self.__mparnames= self.__setParNames( mparnames, len(self.__mparv) )
        self.__maxiterations= maxiter
        self.__deltachisq= deltachisq
        if ndof == None:
            self.__ndof= len(self.__datav) - len(self.__uparv)
        else:
            self.__ndof= ndof
        self.__fixedUparFunctions= {}
        self.__fixedMparFunctions= {}
        return

    # Set parameter names to default or from user input:
    def __setParNames( self, parnames, npar ):
        myparnames= []
        for ipar in range( npar ):
            if parnames and ipar in parnames:
                parname= parnames[ipar]
            else:
                parname= "Parameter " + repr(ipar).rjust(2)
            myparnames.append( parname )
        return myparnames

    # Method to run clsq solver:
    def solve( self, lpr=False, lBlobel=False ):
        self.__lBlobel= lBlobel
        # If run again reset measured parameters to data values, chi^2 and iteration counter:
        self.__mparv= self.__datav.copy()
        self.__chisq= 0.0
        self.__niterations= 0
        datadim= self.__datav.shape[0]
        upardim= self.__uparv.shape[0]
        if lpr:
            print "Chi^2 before fit:", self.__chisq
        # Iterate clsq calculation until convergence or maximum:
        for iiter in range( self.__maxiterations ):
            self.__niterations+= 1
            if lpr:
                print "iteration", iiter
                if lBlobel:
                    print "Solution by partition"
                else:
                    print "Solution by inversion"
            # Calculate constraint derivatives and constraints:
            dcdmpm= self.__constraints.derivativeM( self.__mparv, self.__uparv )
            dcdupm= self.__constraints.derivativeU( self.__mparv, self.__uparv )
            constrv= - self.__constraints.calculate( self.__mparv, self.__uparv )
            # Solve problem:
            cnstrdim= constrv.shape[0]
            dim= datadim + upardim + cnstrdim
            startpar= matrix( zeros(shape=(dim,1)) )
            startpar[:datadim]= self.__mparv
            startpar[datadim:datadim+upardim]= self.__uparv
            # Choose one solution algorithm, result is difference (delta)
            # w.r.t. startparameters and submatrix c33 w.r.t. measured parameters,
            # then update parameters:
            if not lBlobel:
                deltapar, c33= self.__solveByInversion( dcdmpm, 
                                                        dcdupm, 
                                                        constrv )
            else:
                deltapar, c33= self.__solveByPartition( dcdmpm, 
                                                        dcdupm, 
                                                        constrv )
            newpar= startpar + deltapar
            self.__mparv= newpar[:datadim]
            self.__uparv= newpar[datadim:datadim+upardim]
            # Check if chi^2 changed below threshold or maximum number of
            # iterations reached:
            chisqnew= self.calcChisq( dcdmpm, c33 )
            if lpr:
                print "Chi^2=", chisqnew
            if( abs(chisqnew-self.__chisq) < self.__deltachisq and
                iiter > 0 ):
                break
            self.__chisq= chisqnew
        return

    # Solve directly using linalg.solve:
    def __solveByInversion( self, dcdmpm, dcdupm, constrv ):
        self.__pm= self.__makeProblemMatrix( dcdmpm, dcdupm )
        self.__pminv= None
        datadim= dcdmpm.shape[1]
        upardim= dcdupm.shape[1]
        dim= self.__pm.shape[0]
        rhsv= self._prepareRhsv( dim, datadim, upardim, constrv )
        deltapar= linalg.solve( self.__pm, rhsv )
        self.__pminv= self.__pm.getI()
        c33= self.__pminv[datadim+upardim:,datadim+upardim:]
        return deltapar, c33
    # To be overriden in subclass for constrained likelihood:
    def _prepareRhsv( self, dim, datadim, upardim, constrv ):
        rhsv= matrix( zeros(shape=(dim,1)) )
        rhsv[datadim+upardim:]= constrv
        return rhsv
    def __makeProblemMatrix( self, dcdmpm, dcdupm ):
        invm= self.getInvm()
        datadim= invm.shape[0]
        cnstrdim= dcdmpm.shape[0]
        upardim= dcdupm.shape[1]
        dim= datadim+upardim+cnstrdim
        pm= matrix( zeros(shape=(dim,dim)) )
        pm[:datadim,:datadim]= invm
        pm[:datadim,datadim+upardim:]= dcdmpm.getT()
        pm[datadim+upardim:,:datadim]= dcdmpm
        pm[datadim:datadim+upardim,datadim+upardim:]= dcdupm.getT()
        pm[datadim+upardim:,datadim:datadim+upardim]= dcdupm
        return pm

    # Solve a la Blobel and CERN 60-30:
    def __solveByPartition( self, dcdmpm, dcdupm, constrv ):
        c11, c21, c31, c32, c33, errorMatrix= self.__makeInverseProblemMatrix( dcdmpm,
                                                                               dcdupm )
        self.__pminv= errorMatrix
        datadim= dcdmpm.shape[1]
        upardim= dcdupm.shape[1]
        constrdim= dcdmpm.shape[0]
        deltapar= self._prepareDeltapar( datadim, upardim, constrdim,
                                         c11, c21, c31, c32, c33, constrv )
        return deltapar, c33
    # To be overriden in subclass for constrained likelihood:
    def _prepareDeltapar( self, datadim, upardim, constrdim,
                         c11, c21, c31, c32, c33, constrv ):
        dim= datadim+upardim+constrdim
        deltapar= matrix( zeros(shape=(dim,1)) )
        deltapar[:datadim]= c31.getT()*constrv
        deltapar[datadim:datadim+upardim]= c32.getT()*constrv
        deltapar[datadim+upardim:]= c33*constrv
        return deltapar
    def __makeInverseProblemMatrix( self, dcdmpm, dcdupm ):
        covm= self.getCovm()
        dcdmpmT= dcdmpm.getT()
        dcdupmT= dcdupm.getT()
        wbinv= dcdmpm*covm*dcdmpmT
        wb= wbinv.getI()
        wa= dcdupmT*wb*dcdupm
        wainv= wa.getI()
        btwbb= dcdmpmT*wb*dcdmpm
        awainvat= dcdupm*wainv*dcdupmT
        wbawainvatwbbwinv= wb*awainvat*wb*dcdmpm*covm
        c11= ( covm - covm*btwbb*covm 
               + covm*dcdmpmT*wbawainvatwbbwinv )
        c21= - wainv*dcdupmT*wb*dcdmpm*covm
        c22= wainv
        c31= wb*dcdmpm*covm - wbawainvatwbbwinv
        c32= wb*dcdupm*wainv
        c33= - wb + wb*awainvat*wb
        datadim= c11.shape[0]
        upardim= c21.shape[0]
        constrdim= c33.shape[0]
        dim= datadim+upardim+constrdim
        errorMatrix= matrix( zeros(shape=(dim,dim)) )
        errorMatrix[:datadim,:datadim]= c11
        errorMatrix[:datadim,datadim:datadim+upardim]= c21.getT()
        errorMatrix[datadim:datadim+upardim,:datadim]= c21
        errorMatrix[datadim:datadim+upardim,datadim:datadim+upardim]= c22
        errorMatrix[datadim+upardim:,datadim+upardim:]= -c33
        return c11, c21, c31, c32, c33, errorMatrix

    # Calculate chi^2 for data vs measured parameters:
    def calcChisq( self, dcdmpm, c33 ):
        deltay= self.__datav - self.__mparv
        c= dcdmpm*deltay
        chisq= - c.getT()*c33*c
        return chisq

    # Return last chi^2 calculated in solve:
    def getChisq( self ):
        return float( self.__chisq[0,0] )

    # Return number of degrees of freedom:
    def getNdof( self ):
        return self.__ndof

    # Print information on parameters:
    def __printPars( self, par, parerrs, parnames, fixedParFunctions,
                     ffmt=".4f", pulls=None ):
        for ipar in range(len(par)):
            name= parnames[ipar]
            print "{0:>15s}:".format( name ),
            fmtstr= "{0:10"+ffmt+"} +/- {1:10"+ffmt+"}"
            print fmtstr.format( par[ipar], parerrs[ipar] ),
            if pulls:
                fmtstr= "{0:10"+ffmt+"}"
                print fmtstr.format( pulls[ipar] ),
            if ipar in fixedParFunctions:
                print "(fixed)",
            print
        return

    def printTitle( self ):
        print "\nConstrained least squares CLSQ"
        return

    # Print fit chi^2 information:
    def printFitParameters( self, chisq, ndof, ffmt ):
        fmtstr= "\nChi^2= {0:"+ffmt+"} for {1:d} d.o.f, Chi^2/d.o.f= {2:"+ffmt+"}, P-value= {3:"+ffmt+"}"
        print fmtstr.format( chisq, ndof, chisq/float(ndof), 
                             chisqprob( chisq, ndof ) )
        return

    # Print fit summary:
    def printResults( self, ffmt=".4f", cov=False, corr=False ):
        self.printTitle()
        print "\nResults after fit",
        if self.__lBlobel:
            print "using solution by partition:"
        else:
            print "using solution by inversion:"
        print "\nIterations:", self.__niterations
        print "\nConstraints:"
        constraints= self.getConstraints()
        for constraint in constraints:
            fmtstr= "{0:.6f}"
            print fmtstr.format( constraint ),
        print
        ndof= self.getNdof()
        chisq= self.getChisq()
        self.printFitParameters( chisq, ndof, ffmt )
        print "\nUnmeasured parameters and errors:"
        print "           Name       Value          Error"
        upar= self.getUpars()
        self.__printPars( upar, self.getUparErrors(), 
                          self.__uparnames, self.__fixedUparFunctions,
                          ffmt=ffmt )
        set_printoptions( linewidth=132, precision=4 )
        if len(upar) > 1:
            if cov:
                print "\nCovariance matrix:"
                self.__printMatrix( self.getUparErrorMatrix(), ".3e", 
                                    self.__uparnames )
            if corr:
                print "\nCorrelation matrix:"
                self.__printMatrix( self.getUparCorrMatrix(), ".3f",
                                    self.__uparnames )
        print "\nMeasured parameters:"
        print "           Name       Value          Error       Pull"
        mpar= self.getMpars()
        self.__printPars( mpar, self.getMparErrors(), 
                          self.__mparnames, self.__fixedMparFunctions,
                          ffmt=ffmt, 
                          pulls=self.getMparPulls() )
        if len(mpar) > 1:
            if cov:
                print "\nCovariance matrix:"
                self.__printMatrix( self.getMparErrorMatrix(), ".3e", 
                                    self.__mparnames )
            if corr:
                print "\nCorrelation matrix:"
                self.__printMatrix( self.getMparCorrMatrix(), ".3f", 
                                    self.__mparnames )
        if corr:
            print "\nTotal correlations unmeasured and measured parameters:"
            self.__printMatrix( self.getCorrMatrix(), ".3f", 
                                self.__mparnames+self.__uparnames )
        set_printoptions()
        return

    # Print a matrix formatted:
    def __printMatrix( self, m, ffmt, names ):
        mshape= m.shape
        print "{0:10s}".format( "" ),
        for i in range(mshape[0]):
            print "{0:>10s}".format( names[i] ),
        print
        for i in range(mshape[0]):
            print "{0:>10s}".format( names[i] ),
            for j in range(mshape[1]):
                fmtstr= "{0:10"+ffmt+"}"
                print fmtstr.format( m[i,j] ),
            print
        return
    
    # Get constraint values as list:
    def getConstraints( self ):
        constrv= self.__constraints.calculate( self.__mparv, self.__uparv )
        lconstr= [ value for value in constrv.flat ]
        return lconstr

    # Accessors for unmeasured and measured parameters and errors after fit:
    def getPars( self ):
        return self.getUpars()+self.getMpars()
    def getParErrors( self ):
        return self.getUparErrors()+self.getMparErrors()
    def getUparv( self ):
        return self.__uparv
    def getMparv( self ):
        return self.__mparv
    def getUpars( self ):
        return [ upar for upar in self.__uparv.flat ]
    def getMpars( self ):
        return [ mpar for mpar in self.__mparv.flat ]
    def getUparNames( self ):
        return list( self.__uparnames )
    def getMparNames( self ):
        return list( self.__mparnames )
    def getUparErrors( self ):
        if self.__pminv == None:
            self.__pminv= self.__pm.getI()
        datadim= self.__datav.shape[0]
        upardim= self.__uparv.shape[0]
        errors= []
        for i in range( datadim, datadim+upardim ):
            errors.append( sqrt( abs(self.__pminv[i,i]) ) )
        return errors
    def getMparErrors( self ):
        if self.__pminv == None:
            self.__pminv= self.__pm.getI()
        datadim= self.__datav.shape[0]
        errors= []
        for i in range( datadim ):
            errors.append( sqrt( abs(self.__pminv[i,i]) ) )
        return errors
    def getUparErrorMatrix( self ):
        if self.__pminv == None:
            self.__pminv= self.__pm.getI()
        datadim= self.__datav.shape[0]
        upardim= self.__uparv.shape[0]
        return self.__pminv[datadim:datadim+upardim,datadim:datadim+upardim]
    def getMparErrorMatrix( self ):
        if self.__pminv == None:
            self.__pminv= self.__pm.getI()
        datadim= self.__datav.shape[0]
        return self.__pminv[:datadim,:datadim]
    def getErrorMatrix( self ):
        if self.__pminv == None:
            self.__pminv= self.__pm.getI()
        datadim= self.__datav.shape[0]
        upardim= self.__uparv.shape[0]
        return self.__pminv[:datadim+upardim,:datadim+upardim]
    def getUparCorrMatrix( self ):
        return self.__getCorrMatrix( "u" )
    def getMparCorrMatrix( self ):
        return self.__getCorrMatrix( "m" )
    def getCorrMatrix( self ):
        return self.__getCorrMatrix( "a" )
    def __getCorrMatrix( self, paropt ):
        if paropt == "u":
            errormatrix= self.getUparErrorMatrix()
        elif paropt == "m":
            errormatrix= self.getMparErrorMatrix()
        elif paropt == "a":
            errormatrix= self.getErrorMatrix()
        mshape= errormatrix.shape
        corrmatrix= matrix( zeros(shape=mshape) )
        for i in range( mshape[0] ):
            for j in range( mshape[1] ):
                corrmatrix[i,j]= ( errormatrix[i,j]/
                                   sqrt(errormatrix[i,i]*errormatrix[j,j]) )
        return corrmatrix

    def getCovm( self ):
        return self.__covm

    def getInvm( self ):
        if self.__invm == None:
            self.__invm= self.__covm.getI()
        return self.__invm

    def getData( self ):
        return [ datum for datum in self.__datav.flat ]
    def getDatav( self ):
        return self.__datav

    # Calculate pulls for measured parameters a la Blobel
    # from errors on Deltax = "measured parameter - data"
    def getMparPulls( self ):
        covm= self.getCovm()
        #mpar= [ par for par in self.__mparv.flat ]
        mpar= self.getMpars()
        #data= [ datum for datum in self.__datav.flat ]
        data= self.getData()
        errors= self.getMparErrors()
        for i in range( len(errors) ):
            errors[i]= sqrt( covm[i,i] - errors[i]**2 )
        pulls= []
        for datum, mparval, error in zip( data, mpar, errors ):
            pull= mparval - datum
            if error > 1.0e-7:
                pull/= error
#            else:
#                print "getMparPulls: Warning: error < 1e-7"
            pulls.append( pull )
        return pulls

    # Fix or release parameters:
    def fixPar( self, ipar, ptype, val=None, lpr=True ):
        if ptype == "u":
            self.fixUpar( ipar, val, lpr )
        elif ptype == "m":
            self.fixMpar( ipar, val, lpr )
        else:
            print "fixPar: ptype not recognised:", ptype
        return
    def releasePar( self, ipar, ptype, val=None, lpr=True ):
        if ptype == "u":
            self.releaseUpar( ipar, lpr )
        elif ptype == "m":
            self.releaseMpar( ipar, lpr )
        else:
            print "releasePar: ptype not recognised:", ptype
        return
    def fixUpar( self, parspec, val=None, lpr=True ):
        ipar= self.__parameterIndex( parspec, self.__uparnames )
        if val == None:
            val= self.__uparv[ipar].item()
        fixUparConstraint= funcobj( ipar, val, "u" )
        self.__fixedUparFunctions[ipar]= fixUparConstraint
        self.__constraints.addConstraint( fixUparConstraint )
        if lpr:
            print "Fixed unmeasured parameter", self.__uparnames[ipar], "to", val
        return
    def releaseUpar( self, parspec, lpr=True ):
        ipar= self.__parameterIndex( parspec, self.__uparnames )
        fixUparConstraint= self.__fixedUparFunctions[ipar]
        self.__constraints.removeConstraint( fixUparConstraint )
        del self.__fixedUparFunctions[ipar]
        if lpr:
            print "Released unmeasured parameter", self.__uparnames[ipar]
        return
    def fixMpar( self, parspec, val=None, lpr=True ):
        ipar= self.__parameterIndex( parspec, self.__mparnames )
        if val == None:
            val= self.__mparv[ipar].item()
        fixMparConstraint= funcobj( ipar, val, "m" )
        self.__fixedMparFunctions[ipar]= fixMparConstraint
        self.__constraints.addConstraint( fixMparConstraint )
        if lpr:
            print "Fixed measured parameter", self.__mparnames[ipar], "to", val
        return
    def releaseMpar( self, parspec, lpr=True ):
        ipar= self.__parameterIndex( parspec, self.__mparnames )
        fixMparConstraint= self.__fixedMparFunctions[ipar]
        self.__constraints.removeConstraint( fixMparConstraint )
        del self.__fixedMparFunctions[ipar]
        if lpr:
            print "Released measured parameter", self.__mparnames[ipar]
        return

    # Set a value for unmeasured or measured parameter:
    def setUpar( self, parspec, val ):
        ipar= self.__parameterIndex( parspec, self.__uparnames )
        self.__uparv[ipar]= val
        print "Set unmeasured parameter", self.__uparnames[ipar], "to", val
        return
    def setMpar( self, parspec, val ):
        ipar= self.__parameterIndex( parspec, self.__mparnames )
        self.__mparv[ipar]= val
        print "Set measured parameter", self.__mparnames[ipar], "to", val
        return

    # Profiling of unmeasured or measured parameters by fixing given
    # parameter and re-solving for a range of parameter values:
    def profileUpar( self, parspec, nstep=5, stepsize=1.0 ):
        ipar= self.__parameterIndex( parspec, self.__uparnames )
        steps, results= self.__profile( ipar, nstep, stepsize,
                                        self.getUpar, self.getUparErrors, 
                                        self.fixUpar, self.releaseUpar )
        return steps, results
    def profileMpar( self, parspec, nstep=5, stepsize=1.0 ):
        ipar= self.__parameterIndex( parspec, self.__mparnames )
        steps, results= self.__profile( ipar, nstep, stepsize,
                                        self.getMpar, self.getMparErrors, 
                                        self.fixMpar, self.releaseMpar )
        return steps, results
    def __profile( self, ipar, nstep, stepsize,
                   getPar, getParErrors, fixPar, releasePar ):
        value= getPar()[ipar]
        error= getParErrors()[ipar]
        steps= []
        for i in range( nstep ):
            steps.append( (i-nstep/2)*error*stepsize + value )
        results= []
        for step in steps:
            fixPar( ipar, step, lpr=False )
            self.solve()
            results.append( self.getChisq() )
            releasePar( ipar, lpr=False )
        self.solve()
        return steps, results

    # # Error calculation a la Minuits Minos algorithm, find for given 
    # # parameter the values which solve Chi^2 = Chi^2_min+delta.  Technically,
    # # the public methods pass the matching method references to the
    # # private method containing the algorithm:
    # def minosUpar( self, parspec, chisqmin=None, delta=1.0 ):
    #     ipar= self.__parameterIndex( parspec, self.__uparnames )
    #     return self.__minos( ipar, chisqmin, delta,
    #                           self.getUpars, self.getUparErrors, 
    #                           self.fixUpar, self.releaseUpar )
    # def minosMpar( self, parspec, chisqmin=None, delta=1.0 ):
    #     ipar= self.__parameterIndex( parspec, self.__mparnames )
    #     return self.__minos( ipar, chisqmin, delta,
    #                           self.getMpars, self.getMparErrors, 
    #                           self.fixMpar, self.releaseMpar )
    # def __minos( self, ipar, chisqmin, delta,
    #              getPar, getParErrors, fixPar, releasePar ):
    #     if chisqmin == None:
    #         chisqmin= self.getChisq()
    #     result= getPar()[ipar]
    #     error= getParErrors()[ipar]
    #     def fun( x ):
    #         fixPar( ipar, x, lpr=False )
    #         self.solve()
    #         deltachisq= self.getChisq() - chisqmin
    #         releasePar( ipar, lpr=False )
    #         return deltachisq - delta
    #     a= result
    #     b= result + (sqrt(delta)+1.0)*error
    #     errhi= brentq( fun, a, b, xtol=1.0e-6, rtol=1.0e-6 )
    #     a= result - (sqrt(delta)+1.0)*error
    #     b= result
    #     errlo= brentq( fun, a, b, xtol=1.0e-6, rtol=1.0e-6 )
    #     self.solve()
    #     return errhi-result, errlo-result

    # # 2D Chi^2 profile obtained by fixing both parameters on a grid
    # # and re-solving at each point:
    # def profile2d( self, ipar1, type1, ipar2, type2, sigma=3.0, nstep=21 ):
    #     values= []
    #     errors= []
    #     def setupPars( ipar, ptype ):
    #         fixpar= None
    #         releasepar= None
    #         if ptype == "u":
    #             values.append( self.getUpars()[ipar] )
    #             errors.append( self.getUparErrors()[ipar] )
    #             fixpar= self.fixUpar
    #             releasepar= self.releaseUpar
    #         elif ptype == "m":
    #             values.append( self.getMpars()[ipar] )
    #             errors.append( self.getMparErrors()[ipar] )
    #             fixpar= self.fixMpar
    #             releasepar= self.releaseMpar
    #         return fixpar, releasepar
    #     fixpar1, releasepar1= setupPars( ipar1, type1 )
    #     fixpar2, releasepar2= setupPars( ipar2, type2 )
    #     steplists= []
    #     stepsize= 1.0/(nstep/2)
    #     for value, error in zip( values, errors ):
    #         steps= []
    #         for istep in range( nstep ):
    #             steps.append( (istep-nstep/2)*error*stepsize*sigma + value )
    #         steplists.append( steps )
    #     resultlist= []
    #     chisqmin= self.getChisq()
    #     for step1 in steplists[0]:
    #         fixpar1( ipar1, step1, lpr=False )
    #         results= []
    #         for step2 in steplists[1]:
    #             fixpar2( ipar2, step2, lpr=False )
    #             self.solve()
    #             result= ( step1, step2, self.getChisq() - chisqmin )
    #             resultlist.append( result )
    #             releasepar2( ipar2, lpr=False )
    #         releasepar1( ipar1, lpr=False )
    #     self.solve()
    #     return resultlist

    # Helper method to find global index of a parameter corresponding to
    # solution vector (unmeasured and measured parameters) given index or name:
    def __parameterIndex( self, parspec, parnames ):
        if isinstance( parspec, int ):
            ipar= parspec
        elif isinstance( parspec, str ):
            ipar= parnames.index( parspec )
        else:
            raise TypeError( "parspec neither int nor str" )
        return ipar

# Helper function object (callable) class to handle extra constraints
# for parameter fixing and releasing:
class funcobj:
    def __init__( self, ipar, val, opt="u" ):
        self.__ipar= ipar
        self.__val= val
        def uparConstraint( mpar, upar ):
            return upar[self.__ipar] - self.__val
        def mparConstraint( mpar, upar ):
            return mpar[self.__ipar] - self.__val
        if opt == "u":
            self.__constraint= uparConstraint
        elif opt == "m":
            self.__constraint= mparConstraint
        else:
            raise NameError( "Option is neither u nor m" )
        return
    def __call__( self, mpar, upar ):
        return [ self.__constraint( mpar, upar ) ]

# Class contains constraints calculations:
class Constraints:

    # Ctor, takes external constraint function reference and its optional arguments:
    def __init__( self, Fun, args=(), epsilon=1.0e-4 ):
        self.__ConstrFun= Fun
        self.__args= args
        self.__precision= epsilon
        self.__Constraints= []
        return

    # Add or remove a constraint, i.e. activate or deactivate it, this handles
    # parameter fixing or releasing by introducing additional constraints:
    def addConstraint( self, constraintFun ):
        self.__Constraints.append( constraintFun )
        return
    def removeConstraint( self, constraintFun ):
        self.__Constraints.remove( constraintFun )
        return

    # Return all active constraints given values of measured and unmeasured
    # parameters:
    def calculate( self, mpar, upar ):
        constraints= self.__ConstrFun( mpar, upar, *self.__args )
        for constraint in self.__Constraints:
            constraints+= constraint( mpar, upar )
        constraintsvector= columnMatrixFromList( constraints )
        return constraintsvector

    # Calculate derivatives of constraints w.r.t. measured or unmeasured parameters.
    # The private method varies the first and keeps the second vector of parameters
    # fixed, the two public methods set up their internal functions accordingly. The
    # numerical method is a five-point-stencil:
    def derivativeM( self, mpar, upar ):
        def calcM( varpar, fixpar ):
            return self.calculate( varpar, fixpar )
        return self.__derivative( calcM, mpar, upar )
    def derivativeU( self, mpar, upar ):
        def calcU( varpar, fixpar ):
            return self.calculate( fixpar, varpar )
        return self.__derivative( calcU, upar, mpar )
    def __derivative( self, function, varpar, fixpar ):
        columns= []
        varpardim= len(varpar)
        h= matrix( zeros( shape=(varpardim,1) ) )
        for ipar in range( varpardim ):
            h[ipar]= setH( self.__precision, varpar[ipar] )
            column= fivePointStencil( function, varpar, h, fixpar )
            columns.append( column )
            h[ipar]= 0.0
        if len(columns) > 0:
            cnstrdim= len(columns[0])
        else:
            cnstrdim= 0
        dcdp= matrix( zeros( shape=(cnstrdim,varpardim) ) )
        for ipar in range( varpardim ):
            dcdp[:,ipar]= columns[ipar]
        return dcdp

# Helper functions for numerical derivative calculations:
def setH( eps, val ):
    result= eps
    if abs( val ) > 1.0e-6:
        result*= val
    return result
def fivePointStencil( function, varpar, h, fixpar ):
    dfdp= ( - function( varpar + 2.0*h, fixpar ) 
              + 8.0*function( varpar + h, fixpar ) 
              - 8.0*function( varpar - h, fixpar )
              + function( varpar - 2.0*h, fixpar ) )/( 12.0*h.sum() )
    return dfdp


def getPar( solver, ipar, ptype ):
    result= None
    if ptype == "u":
        result= solver.getUpars()[ipar]
    elif ptype == "m":
        result= solver.getMpars()[ipar]
    else:
        print "getPar: ptype not recognised:", ptype
    return result
def getErr( solver, ipar, ptype ):
    error= None
    if ptype == "u":
        error= solver.getUparErrors()[ipar]
    elif ptype == "m":
        error= solver.getMparErrors()[ipar]
    else:
        print "getError: ptype not recognised:", ptype
    return error
def getParName( solver, ipar, ptype ):
    name= None
    if ptype == "u":
        name= solver.getUparNames()[ipar]
    elif ptype == "m":
        name= solver.getMparNames()[ipar]
    else:
        print "getParName: ptype not recognised:", ptype
    return name


def profile( solver, ipar, ptype, nstep=11, sigma=3.0 ):
    value= getPar( solver, ipar, ptype )
    error= getErr( solver, ipar, ptype )
    steps= []
    stepsize= 1.0/(nstep/2)
    for istep in range( nstep ):
        steps.append( (istep-nstep/2)*error*stepsize*sigma + value )
    results= []
    for step in steps:
        solver.fixPar( ipar, ptype, step, lpr=False )
        solver.solve()
        results.append( ( step, solver.getChisq() ) )
        solver.releasePar( ipar, ptype, lpr=False )
    solver.solve()
    return results



# Error calculation a la Minuits Minos algorithm, find for given 
# parameter the values which solve Chi^2 = Chi^2_min+delta.  Technically,
# the public methods pass the matching method references to the
# private method containing the algorithm:
def minos( solver, ipar, ptype, chisqmin=None, delta=1.0 ):
    if chisqmin == None:
        chisqmin= solver.getChisq()
    result= getPar( solver, ipar, ptype )
    error= getErr( solver, ipar, ptype )
    def fun( x ):
        solver.fixPar( ipar, ptype, x, lpr=False )
        solver.solve()
        deltachisq= solver.getChisq() - chisqmin
        solver.releasePar( ipar, ptype, lpr=False )
        return deltachisq - delta
    a= result
    b= result + (sqrt(delta)+1.0)*error
    errhi= brentq( fun, a, b, xtol=1.0e-6, rtol=1.0e-6 )
    a= result - (sqrt(delta)+1.0)*error
    b= result
    errlo= brentq( fun, a, b, xtol=1.0e-6, rtol=1.0e-6 )
    solver.solve()
    return errhi-result, errlo-result

# 2D Chi^2 profile obtained by fixing both parameters on a grid
# and re-solving at each point:
def profile2d( solver, ipar1, type1, ipar2, type2, sigma=3.0, nstep=21, lDelta=True ):
    values= []
    errors= []
    for ipar, ptype in ( (ipar1,type1), (ipar2,type2) ):
        values.append( getPar( solver, ipar, ptype ) )
        errors.append( getErr( solver, ipar, ptype ) )
    steplists= []
    stepsize= 1.0/(nstep/2)
    for value, error in zip( values, errors ):
        steps= []
        for istep in range( nstep ):
            steps.append( (istep-nstep/2)*error*stepsize*sigma + value )
        steplists.append( steps )
    resultlist= []
    chisqmin= 0.0
    if lDelta:
        solver.fixPar( ipar1, type1, lpr=False )
        solver.fixPar( ipar2, type2, lpr=False )
        solver.solve()
        chisqmin= solver.getChisq()
        solver.releasePar( ipar1, type1, lpr=False )
        solver.releasePar( ipar2, type2, lpr=False )
    for step1 in steplists[0]:
        solver.fixPar( ipar1, type1, step1, lpr=False )
        results= []
        for step2 in steplists[1]:
            solver.fixPar( ipar2, type2, step2, lpr=False )
            solver.solve()
            result= ( step1, step2, solver.getChisq() - chisqmin )
            resultlist.append( result )
            solver.releasePar( ipar2, type2, lpr=False )
        solver.releasePar( ipar1, type1, lpr=False )
    solver.solve()
    return resultlist

