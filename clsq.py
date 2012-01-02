
from numpy import matrix, zeros, set_printoptions, linalg, delete
from math import sqrt, exp
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
                  mparnames=None, uparnames=None ):
        self.constraints= Constraints( constraintfunction, args, epsilon )
        self.datav= columnMatrixFromList( data )
        self.mparv= self.datav.copy()
        self.uparv= columnMatrixFromList( upar )
        self.covm= matrix( covm )
        self.invm= None
        self.pm= None
        self.pminv= None
        self.niterations= 0
        self.mparnames= mparnames
        self.uparnames= uparnames
        self.maxiterations= maxiter
        self.deltachisq= deltachisq
        self.constrdim= None
        return

    def solve( self, lpr=False, lBlobel=False ):
        self.lBlobel= lBlobel
        datadim= self.datav.shape[0]
        upardim= self.uparv.shape[0]
        constrv= self.constraints.calculate( self.mparv, self.uparv )
        cnstrdim= constrv.shape[0]
        self.constrdim= cnstrdim
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
            if abs(chisqnew-self.chisq) < self.deltachisq:
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
        return self.chisq.ravel().tolist()[0][0]

    def getNdof( self ):
        return len(self.datav) + self.constrdim - len(self.uparv)

    def __printPars( self, par, parerrs, parnames, ffmt=".4f", pulls=None ):
        for ipar in range(len(par)):
            name= "Parameter " + repr(ipar).rjust(2)
            if parnames and ipar in parnames:
                name= parnames[ipar]
            print "{0:>15s}:".format( name ),
            fmtstr= "{0:10"+ffmt+"} +/- {1:10"+ffmt+"}"
            print fmtstr.format( par[ipar], parerrs[ipar] ),
            if pulls:
                fmtstr= "{0:10"+ffmt+"}"
                print fmtstr.format( pulls[ipar] )
            else:
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
                          self.uparnames, ffmt=ffmt )
        set_printoptions( linewidth=132, precision=4 )
        if len(upar) > 1:
            if cov:
                print "Covariance matrix:"
                print self.getUparErrorMatrix()
            if corr:
                print "Correlation matrix:"
                print self.getUparCorrMatrix()
        print "\nMeasured parameters:"
        print "           Name       Value          Error       Pull"
        mpar= self.getMpar()
        self.__printPars( mpar, self.getMparErrors(), 
                          self.mparnames, ffmt=ffmt, 
                          pulls=self.getMparPulls() )
        if len(mpar) > 1:
            if cov:
                print "Covariance matrix:"
                print self.getMparErrorMatrix()
            if corr:
                print "Correlation matrix:"
                print self.getMparCorrMatrix()
        if corr:
            print "\nTotal correlations unmeasured and measured parameters:"
            print self.getCorrMatrix()
        set_printoptions()
        return

    def getConstraints( self ):
        constrv= self.constraints.calculate( self.mparv, self.uparv )
        lconstr= constrv.ravel().tolist()[0]
        return lconstr

    def getUpar( self ):
        return self.uparv.ravel().tolist()[0]
    def getMpar( self ):
        return self.mparv.ravel().tolist()[0]
    def getUparErrors( self ):
        if self.pminv == None:
            self.pminv= self.pm.getI()
        datadim= self.datav.shape[0]
        upardim= self.uparv.shape[0]
        errors= []
        for i in range( datadim, datadim+upardim ):
            errors.append( sqrt( self.pminv[i,i] ) )
        return errors
    def getMparErrors( self ):
        if self.pminv == None:
            self.pminv= self.pm.getI()
        datadim= self.datav.shape[0]
        errors= []
        for i in range( datadim ):
            errors.append( sqrt( self.pminv[i,i] ) )
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

    def getMparPulls( self ):
        covm= self.getCovm()
        mpar= self.mparv.ravel().tolist()[0]
        data= self.datav.ravel().tolist()[0]
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


class Constraints:

    def __init__( self, Fun, args=(), epsilon=1.0e-4 ):
        self.ConstrFun= Fun
        self.args= args
        self.precision= epsilon 
        return

    def calculate( self, mpar, upar ):
        constraints= self.ConstrFun( mpar, upar, *self.args )
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
            h[ipar]= setH( self.precision, varpar[ipar] )
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
