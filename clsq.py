
# Constrained least squares fit implementing CERN 60-30, and
# additions from Blobel
# S. Kluth 12/2011

from numpy import matrix, zeros, set_printoptions, linalg, delete
from scipy.optimize import brentq
from math import sqrt
from ROOT import TMath


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


class clsqSolver:

    def __init__( self, data, covm, upar, 
                  constraintfunction, args=(), epsilon=0.0001,
                  maxiter= 100, deltachisq=0.0001,
                  mparnames=None, uparnames=None,
                  ndof=None ):
        self.constraints= Constraints( constraintfunction, args, epsilon )
        self.datav= columnMatrixFromList( data )
        self.mparv= self.datav.copy()
        self.uparv= columnMatrixFromList( upar )
        self.covm= matrix( covm )
        self.invm= None
        self.pm= None
        self.pminv= None
        self.niterations= 0
        self.uparnames= self.__setParNames( uparnames, len(self.uparv) )
        self.mparnames= self.__setParNames( mparnames, len(self.mparv) )
        self.maxiterations= maxiter
        self.deltachisq= deltachisq
        if ndof == None:
            self.ndof= len(self.datav) - len(self.uparv)
        else:
            self.ndof= ndof
        self.fixedUparFunctions= {}
        self.fixedMparFunctions= {}
        return

    def __setParNames( self, parnames, npar ):
        myparnames= []
        for ipar in range( npar ):
            if parnames and ipar in parnames:
                parname= parnames[ipar]
            else:
                parname= "Parameter " + repr(ipar).rjust(2)
            myparnames.append( parname )
        return myparnames

    def solve( self, lpr=False, lBlobel=False ):
        self.lBlobel= lBlobel
        self.mparv= self.datav.copy()
        datadim= self.datav.shape[0]
        upardim= self.uparv.shape[0]
        constrv= self.constraints.calculate( self.mparv, self.uparv )
        cnstrdim= constrv.shape[0]
        dim= datadim + upardim + cnstrdim
        startpar= matrix( zeros(shape=(dim,1)) )
        self.chisq= 0.0
        if lpr:
            print "Chi^2 before fit:", self.chisq
        self.niterations= 0
        for iiter in range( self.maxiterations ):
            self.niterations+= 1
            if lpr:
                print "iteration", iiter
                if lBlobel:
                    print "Solution by partition"
                else:
                    print "Solution by inversion"
            # Calculate derivatives and constraints:
            dcdmpm= self.constraints.derivativeM( self.mparv, self.uparv )
            dcdupm= self.constraints.derivativeU( self.mparv, self.uparv )
            constrv= - self.constraints.calculate( self.mparv, self.uparv )
            # Solve problem:
            startpar[:datadim]= self.mparv
            startpar[datadim:datadim+upardim]= self.uparv
            if not lBlobel:
                deltapar, c33= self.__solveByInversion( dcdmpm, 
                                                        dcdupm, 
                                                        constrv )
            else:
                deltapar, c33= self.solveByPartition( dcdmpm, 
                                                      dcdupm, 
                                                      constrv )
            newpar= startpar + deltapar
            self.mparv= newpar[:datadim]
            self.uparv= newpar[datadim:datadim+upardim]
            # Check if chi^2 changed:
            chisqnew= self.calcChisq( dcdmpm, c33 )
            if lpr:
                print "Chi^2=", chisqnew
            if( abs(chisqnew-self.chisq) < self.deltachisq and
                iiter > 0 ):
                break
            self.chisq= chisqnew
        return

    def __solveByInversion( self, dcdmpm, dcdupm, constrv ):
        self.pm= self.__makeProblemMatrix( dcdmpm, dcdupm )
        self.pminv= None
        datadim= dcdmpm.shape[1]
        upardim= dcdupm.shape[1]
        dim= self.pm.shape[0]
        rhsv= self.prepareRhsv( dim, datadim, upardim, constrv )
        deltapar= linalg.solve( self.pm, rhsv )
        self.pminv= self.pm.getI()
        c33= self.pminv[datadim+upardim:,datadim+upardim:]
        return deltapar, c33

    def prepareRhsv( self, dim, datadim, upardim, constrv ):
        rhsv= matrix( zeros(shape=(dim,1)) )
        rhsv[datadim+upardim:]= constrv
        return rhsv

    def solveByPartition( self, dcdmpm, dcdupm, constrv ):
        c11, c21, c31, c32, c33, errorMatrix= self.makeInverseProblemMatrix( dcdmpm,
                                                                             dcdupm )
        self.pminv= errorMatrix
        datadim= dcdmpm.shape[1]
        upardim= dcdupm.shape[1]
        constrdim= dcdmpm.shape[0]
        deltapar= self.prepareDeltapar( datadim, upardim, constrdim,
                                        c11, c21, c31, c32, c33, constrv )
        return deltapar, c33

    def prepareDeltapar( self, datadim, upardim, constrdim,
                         c11, c21, c31, c32, c33, constrv ):
        dim= datadim+upardim+constrdim
        deltapar= matrix( zeros(shape=(dim,1)) )
        deltapar[:datadim]= c31.getT()*constrv
        deltapar[datadim:datadim+upardim]= c32.getT()*constrv
        deltapar[datadim+upardim:]= c33*constrv
        return deltapar

    def calcChisq( self, dcdmpm, c33 ):
        deltay= self.datav - self.mparv
        c= dcdmpm*deltay
        chisq= - c.getT()*c33*c
        return chisq

    def getChisq( self ):
        return float( self.chisq[0,0] )

    def getNdof( self ):
        return self.ndof

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

    def printFitParameters( self, chisq, ndof, ffmt ):
        fmtstr= "\nChi^2= {0:"+ffmt+"} for {1:d} d.o.f, Chi^2/d.o.f= {2:"+ffmt+"}, P-value= {3:"+ffmt+"}"
        print fmtstr.format( chisq, ndof, chisq/float(ndof), 
                             TMath.Prob( chisq, ndof ) )
        return

    def printResults( self, ffmt=".4f", cov=False, corr=False ):
        self.printTitle()
        print "\nResults after fit",
        if self.lBlobel:
            print "using solution by partition:"
        else:
            print "using solution by inversion:"
        print "\nIterations:", self.niterations
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
        upar= self.getUpar()
        self.__printPars( upar, self.getUparErrors(), 
                          self.uparnames, self.fixedUparFunctions,
                          ffmt=ffmt )
        set_printoptions( linewidth=132, precision=4 )
        if len(upar) > 1:
            if cov:
                print "\nCovariance matrix:"
                # print self.getUparErrorMatrix()
                self.__printMatrix( self.getUparErrorMatrix(), ".3e", 
                                    self.uparnames )
            if corr:
                print "\nCorrelation matrix:"
                # print self.getUparCorrMatrix()
                self.__printMatrix( self.getUparCorrMatrix(), ".3f",
                                    self.uparnames )
        print "\nMeasured parameters:"
        print "           Name       Value          Error       Pull"
        mpar= self.getMpar()
        self.__printPars( mpar, self.getMparErrors(), 
                          self.mparnames, self.fixedMparFunctions,
                          ffmt=ffmt, 
                          pulls=self.getMparPulls() )
        if len(mpar) > 1:
            if cov:
                print "\nCovariance matrix:"
                self.__printMatrix( self.getMparErrorMatrix(), ".3e", 
                                    self.mparnames )
            if corr:
                print "\nCorrelation matrix:"
                self.__printMatrix( self.getMparCorrMatrix(), ".3f", 
                                    self.mparnames )
        if corr:
            print "\nTotal correlations unmeasured and measured parameters:"
            self.__printMatrix( self.getCorrMatrix(), ".3f", 
                                self.mparnames+self.uparnames )
        set_printoptions()
        return

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

    def getConstraints( self ):
        constrv= self.constraints.calculate( self.mparv, self.uparv )
        lconstr= [ value for value in constrv.flat ]
        return lconstr

    def getUparv( self ):
        return self.uparv
    def getUpar( self ):
        return [ upar for upar in self.uparv.flat ]
    def getMpar( self ):
        return [ mpar for mpar in self.mparv.flat ]
    def getUparErrors( self ):
        if self.pminv == None:
            self.pminv= self.pm.getI()
        datadim= self.datav.shape[0]
        upardim= self.uparv.shape[0]
        errors= []
        for i in range( datadim, datadim+upardim ):
            errors.append( sqrt( abs(self.pminv[i,i]) ) )
        return errors
    def getMparErrors( self ):
        if self.pminv == None:
            self.pminv= self.pm.getI()
        datadim= self.datav.shape[0]
        errors= []
        for i in range( datadim ):
            errors.append( sqrt( abs(self.pminv[i,i]) ) )
        return errors
    def getUparErrorMatrix( self ):
        if self.pminv == None:
            self.pminv= self.pm.getI()
        datadim= self.datav.shape[0]
        upardim= self.uparv.shape[0]
        return self.pminv[datadim:datadim+upardim,datadim:datadim+upardim]
    def getMparErrorMatrix( self ):
        if self.pminv == None:
            self.pminv= self.pm.getI()
        datadim= self.datav.shape[0]
        return self.pminv[:datadim,:datadim]
    def getErrorMatrix( self ):
        if self.pminv == None:
            self.pminv= self.pm.getI()
        datadim= self.datav.shape[0]
        upardim= self.uparv.shape[0]
        return self.pminv[:datadim+upardim,:datadim+upardim]
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
        return self.covm

    def getInvm( self ):
        if self.invm == None:
            self.invm= self.covm.getI()
        return self.invm

    def getData( self ):
        return self.datav

    # Calculate pulls for measured parameters a la Blobel
    # from errors on Deltax = "measured parameter - data"
    def getMparPulls( self ):
        covm= self.getCovm()
        mpar= [ par for par in self.mparv.flat ]
        data= [ datum for datum in self.datav.flat ]
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

    def makeInverseProblemMatrix( self, dcdmpm, dcdupm ):
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

    def fixUpar( self, parspec, val=None, lpr=True ):
        ipar= self.__parameterIndex( parspec, self.uparnames )
        if val == None:
            val= [ upar for upar in self.uparv.flat ][ipar]
        fixUparConstraint= funcobj( ipar, val, "u" )
        self.fixedUparFunctions[ipar]= fixUparConstraint
        self.constraints.addConstraint( fixUparConstraint )
        if lpr:
            print "Fixed unmeasured parameter", self.uparnames[ipar], "to", val
        return
    def releaseUpar( self, parspec, lpr=True ):
        ipar= self.__parameterIndex( parspec, self.uparnames )
        fixUparConstraint= self.fixedUparFunctions[ipar]
        self.constraints.removeConstraint( fixUparConstraint )
        del self.fixedUparFunctions[ipar]
        if lpr:
            print "Released unmeasured parameter", self.uparnames[ipar]
        return

    def __parameterIndex( self, parspec, parnames ):
        if isinstance( parspec, int ):
            ipar= parspec
        elif isinstance( parspec, str ):
            ipar= parnames.index( parspec )
        else:
            raise TypeError( "parspec neither int nor str" )
        return ipar

    def fixMpar( self, parspec, val=None, lpr=True ):
        ipar= self.__parameterIndex( parspec, self.mparnames )
        if val == None:
            val= [ mpar for mpar in self.mparv.flat ][ipar]
        fixMparConstraint= funcobj( ipar, val, "m" )
        self.fixedMparFunctions[ipar]= fixMparConstraint
        self.constraints.addConstraint( fixMparConstraint )
        if lpr:
            print "Fixed measured parameter", self.mparnames[ipar], "to", val
        return
    def releaseMpar( self, parspec, lpr=True ):
        ipar= self.__parameterIndex( parspec, self.mparnames )
        fixMparConstraint= self.fixedMparFunctions[ipar]
        self.constraints.removeConstraint( fixMparConstraint )
        del self.fixedMparFunctions[ipar]
        if lpr:
            print "Released measured parameter", self.mparnames[ipar]
        return

    def setUpar( self, parspec, val ):
        ipar= self.__parameterIndex( parspec, self.uparnames )
        self.uparv[ipar]= val
        print "Set unmeasured parameter", self.uparnames[ipar], "to", val
        return
    def setMpar( self, parspec, val ):
        ipar= self.__parameterIndex( parspec, self.mparnames )
        self.mparv[ipar]= val
        print "Set measured parameter", self.mparnames[ipar], "to", val
        return

    def profileUpar( self, parspec, nstep=5, stepsize=1.0 ):
        ipar= self.__parameterIndex( parspec, self.uparnames )
        steps, results= self.__profile( ipar, nstep, stepsize,
                                        self.getUpar, self.getUparErrors, 
                                        self.fixUpar, self.releaseUpar )
        return steps, results
    def profileMpar( self, parspec, nstep=5, stepsize=1.0 ):
        ipar= self.__parameterIndex( parspec, self.mparnames )
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

    def minosUpar( self, parspec, chisqmin=None, delta=1.0 ):
        ipar= self.__parameterIndex( parspec, self.uparnames )
        return self.__minos( ipar, chisqmin, delta,
                              self.getUpar, self.getUparErrors, 
                              self.fixUpar, self.releaseUpar )
    def minosMpar( self, parspec, chisqmin=None, delta=1.0 ):
        ipar= self.__parameterIndex( parspec, self.mparnames )
        return self.__minos( ipar, chisqmin, delta,
                              self.getMpar, self.getMparErrors, 
                              self.fixMpar, self.releaseMpar )
    def __minos( self, ipar, chisqmin, delta,
                 getPar, getParErrors, fixPar, releasePar ):
        if chisqmin == None:
            chisqmin= self.getChisq()
        result= getPar()[ipar]
        error= getParErrors()[ipar]
        def fun( x ):
            fixPar( ipar, x, lpr=False )
            self.solve()
            deltachisq= self.getChisq() - chisqmin
            releasePar( ipar, lpr=False )
            return deltachisq - delta
        a= result
        b= result + (sqrt(delta)+1.0)*error
        errhi= brentq( fun, a, b, xtol=1.0e-6, rtol=1.0e-6 )
        a= result - (sqrt(delta)+1.0)*error
        b= result
        errlo= brentq( fun, a, b, xtol=1.0e-6, rtol=1.0e-6 )
        self.solve()
        return errhi-result, errlo-result

    def contour( self, iparspec, iopt, jparspec, jopt,
                 delta=1.0, nstep=20, stepsize=0.1 ):
        self.solve()
        chisqmin= self.getChisq()
        if iopt == "u":
            ipar= self.__parameterIndex( iparspec, self.uparnames )
            value= self.getUpar()[ipar]
            error= self.getUparErrors()[ipar]
            fixPar= self.fixUpar
            releasePar= self.releaseUpar
        elif iopt == "m":
            ipar= self.__parameterIndex( iparspec, self.mparnames )
            value= self.getMpar()[ipar]
            error= self.getMparErrors()[ipar]
            fixPar= self.fixMpar
            releasePar= self.releaseMpar
        if jopt == "u":
            jpar= self.__parameterIndex( jparspec, self.uparnames )
            minos= self.minosUpar
            getPar= self.getUpar
        elif jopt == "m":
            jpar= self.__parameterIndex( jparspec, self.mparnames )
            minos= self.minosMpar
            getPar= self.getMpar
        steps= []
        for i in range( nstep ):
            steps.append( (i-nstep/2+0.5)*error*stepsize*sqrt(delta) + value )
        contourx= []
        contoury= []
        for step in steps:
            fixPar( ipar, step, lpr=False )
            self.solve()
            result= getPar()[jpar]
            try:
                errhi, errlo= minos( jpar, chisqmin, delta )
                contourx.append( step )
                contoury.append( errhi+result )
                contourx.append( step )
                contoury.append( errlo+result )
            except ValueError:
                print "contour: minos problem:", ipar, step, jpar
            releasePar( ipar, lpr=False )
        return contourx, contoury

    def profile2d( self, ipar, iopt, jpar, jopt, nstep=11, stepsize=0.5 ):
        values= []
        errors= []
        if iopt == "u":
            values.append( self.getUpar()[ipar] )
            errors.append( self.getUparErrors()[ipar] )
            fixIpar= self.fixUpar
            releaseIpar= self.releaseUpar
        elif iopt == "m":
            values.append( self.getMpar()[ipar] )
            errors.append( self.getMparErrors()[ipar] )
            fixIpar= self.fixMpar
            releaseIpar= self.releaseMpar
        if jopt == "u":
            values.append( self.getUpar()[jpar] )
            errors.append( self.getUparErrors()[jpar] )
            fixJpar= self.fixUpar
            releaseJpar= self.releaseUpar
        elif jopt == "m":
            values.append( self.getMpar()[jpar] )
            errors.append( self.getMparErrors()[jpar] )
            fixJpar= self.fixMpar
            releaseJpar= self.releaseMpar
        steplists= []
        for value, error in zip( values, errors ):
            steps= []
            for i in range( nstep ):
                steps.append( (i-nstep/2)*error*stepsize + value )
            steplists.append( steps )
        resultlists= []
        for istep in steplists[0]:
            fixIpar( ipar, istep, lpr=False )
            results= []
            for jstep in steplists[1]:
                fixJpar( jpar, jstep, lpr=False )
                self.solve()
                results.append( self.getChisq() )
                releaseJpar( jpar, lpr=False )
            releaseIpar( ipar, lpr=False )
            resultlists.append( results )
        self.solve()
        return steplists, resultlists


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


class Constraints:

    def __init__( self, Fun, args=(), epsilon=1.0e-4 ):
        self.__ConstrFun= Fun
        self.__args= args
        self.__precision= epsilon
        self.__Constraints= []
        return

    def addConstraint( self, constraintFun ):
        self.__Constraints.append( constraintFun )
    def removeConstraint( self, constraintFun ):
        self.__Constraints.remove( constraintFun )

    def calculate( self, mpar, upar ):
        constraints= self.__ConstrFun( mpar, upar, *self.__args )
        for constraint in self.__Constraints:
            constraints+= constraint( mpar, upar )
        constraintsvector= columnMatrixFromList( constraints )
        return constraintsvector

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
