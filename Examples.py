
# Blobels branching ratio example
# See Blobels lecture Terascale School on data combination and limit setting
# (DESY, Oct 2011)
# Input data from attachment to school agenda brExample.txt
# S. Kluth 2012

def Branchingratios( opt="m" ):

    from ConstrainedFit import clsq

    data= [ 0.265, 0.28, 0.37, 0.166, 0.42, 0.5, 0.20, 0.16, 
            0.72, 0.6, 0.37, 0.64, 0.45, 0.028, 10.0, 7.5 ]
    errors= [ 0.014, 0.05, 0.06, 0.013, 0.15, 0.2, 0.08, 0.08,
              0.15, 0.4, 0.16, 0.40, 0.45, 0.009, 5.0, 2.5 ]
    # Error scale factor a la Blobel lecture:
    if "e" in opt:
        print "Apply scaling *2.8 of error[13]"
        errors[13]= errors[13]*2.8
    covm= clsq.covmFromErrors( errors )

    upar= [ 0.33, 0.36, 0.16, 0.09, 0.055 ]
    upnames= { 0: "B1", 1: "B2", 2: "B3", 3: "B4", 4: "B5" }

    def brConstrFun( mpar, upar ):
        constraints= []
        x= []
        for i in range( 5 ):
            x.append( upar[i] )
        for i in range( 5, 21 ):
            x.append( mpar[i-5] )

        constraints.append( x[0]+x[1]+x[2]+x[3]+x[4]-1.0 )
        constraints.append( x[3]-x[5]*x[0] )
        constraints.append( x[3]-x[6]*x[0] )
        constraints.append( x[3]-x[7]*x[0] )
        
        constraints.append( x[3]-(x[1]+x[2])*x[8] )
        constraints.append( x[3]-(x[1]+x[2])*x[9] )
        constraints.append( x[3]-(x[1]+x[2])*x[10] )
        constraints.append( x[3]-(x[1]+x[2])*x[11] )
        constraints.append( x[3]-(x[1]+x[2])*x[12] )
        
        constraints.append( x[1]-(x[1]+x[2])*x[13] )
        constraints.append( x[1]-(x[1]+x[2])*x[14] )
        
        constraints.append( x[0]-(x[1]+x[2])*x[15] )
        constraints.append( x[0]-(x[1]+x[2])*x[16] )
        
        constraints.append( 3.0*x[4]-x[0]*x[17] )
        
        constraints.append( x[3]-x[18] )
        
        constraints.append( (x[1]+x[2])-x[4]*x[19] )
        constraints.append( (x[1]+x[2])-x[4]*x[20] )

        return constraints

    solver= clsq.clsqSolver( data, covm, upar, brConstrFun, epsilon=0.00001,
                             uparnames=upnames )
    print "Constraints before solution"
    print solver.getConstraints()
    lBlobel= False
    if "b" in opt:
        lBlobel= True
    solver.solve( lBlobel=lBlobel )
    lcov= False
    lcorr= False
    if "corr" in opt:
        lcov= True
        lcorr= True
    solver.printResults( cov=lcov, corr=lcorr )

    if "m" in opt:
        _doMinos( solver, "u" )
    if "cont" in opt:
        _doContour( solver, ipar1=2, type1="u", ipar2=3, type2="u" )

    return


# Linear fit with nine data points and additional data to pass
# coordinate points using optional arguments:

def LinearFit():

    from ConstrainedFit import clsq

    xabs= [ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 ]
    data= [ 1.1, 1.9, 2.9, 4.1, 5.1, 6.1, 6.9, 7.9, 9.1 ]
    errors= [ 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1 ]
    covm= clsq.covmFromErrors( errors )

    upar= [ 0.1, 1.1 ]
    upnames= { 0: "a", 1: "b" }

    def linearConstrFun( mpar, upar, xv ):
        constraints= []
        for mparval, xval in zip( mpar, xv ):
            constraints.append( upar[0]+upar[1]*xval - mparval )
        return constraints

    solver= clsq.clsqSolver( data, covm, upar, linearConstrFun, 
                             uparnames=upnames, args=(xabs,) )
    print "Constraints before solution"
    print solver.getConstraints()
    solver.solve()
    solver.printResults()

    return


# Example showing a fit with poisson likelihood, see Blobels 
# Terascale Analysis Center lecture 20 Jan 2010, pg. 33

def PoissonLikelihood( opt="" ):
    
    from ConstrainedFit import clhood
    from ConstrainedFit import clsq
    from scipy.special import gammaln
    from scipy.stats import poisson, norm
    from math import log

    # Data, two counts, errors not needed(!):
    data= [ 9.0, 16.0 ]
    # errors= [ 3.0, 4.0 ]
    mpnames= { 0: "count 1", 1: "count 2" }

    # Fit variable is parameter of poisson distribution:
    upar= [ 12.0 ]
    upnames= { 0: "mu" }

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

    solver= clhood.clhoodSolver( data, upar, lfun, constrFun, 
                                 uparnames=upnames, mparnames=mpnames )
    print "Constraints before solution"
    print solver.getConstraints()
    lBlobel=False
    lPrint= False
    lCorr= False
    if "b" in opt:
        lBlobel= True
    if "p" in opt:
        lPrint= True
    if "c" in opt:
        lCorr= True
    solver.solve( lBlobel=lBlobel, lpr=lPrint )
    solver.printResults( corr=lCorr )

    return

# Linear fit with Gauss (normal) likelihood, and with clsq
# for comparison, expect identical results:

def GaussLikelihood( opt="" ):

    from ConstrainedFit import clhood, clsq
    from scipy.stats import norm
    from math import log

    # Data and errors:
    xabs= [ 1.0, 2.0, 3.0, 4.0, 5.0 ]
    data= [ 1.1, 1.9, 2.9, 4.1, 5.1 ]
    errors= [ 0.1, 0.1, 0.1, 0.1, 0.1 ]

    # Linear function (straight line) parameters:
    upar= [ 0.0, 1.0 ]
    upnames= { 0: "a", 1: "b" }

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

    # Configure options:
    lBlobel=False
    lPrint= False
    lCorr= False
    if "b" in opt:
        lBlobel= True
    if "p" in opt:
        lPrint= True
    if "c" in opt:
        lCorr= True

    # Solution using constrained log(likelihood) minimisation:
    print "\nMax likelihood constrained fit"
    solver= clhood.clhoodSolver( data, upar, lfun, constrFun, uparnames=upnames )
    print "Constraints before solution"
    print solver.getConstraints()
    solver.solve( lBlobel=lBlobel, lpr=lPrint )
    solver.printResults( corr=lCorr )

    # Solution using constrained least squares:
    print "\nLeast squares constrained fit"
    covm= clsq.covmFromErrors( errors )
    solver= clsq.clsqSolver( data, covm, upar, constrFun, uparnames=upnames )
    print "Constraints before solution"
    print solver.getConstraints()
    solver.solve( lBlobel=lBlobel, lpr=lPrint )
    solver.printResults( corr=lCorr )

    return

# Straight line fit to points with errors in x and y, example from 
# Blobels lecture pg 29 with values estimated from the plot
# S Kluth 12.12.2014

def StraightLine( opt="" ):

    from ConstrainedFit import clsq
    from numpy import matrix, zeros

    # Data, errors and correlations:
    xdata= [ 1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0 ]
    ydata= [ 3.0, 2.5, 3.0, 5.0, 7.0, 5.5, 7.5 ]
    xerrs= [ 0.5, 0.3, 0.3, 0.5, 0.5, 0.3, 0.3 ]
    yerrs= [ 0.7, 1.0, 0.5, 0.7, 0.7, 1.0, 0.7 ]
    xyrho= [ -0.25, 0.5, 0.5, -0.25, 0.25, 0.95, -0.25 ]
    #xyrho= [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
    covm= matrix( zeros( (14,14) ) )
    data= []
    npoints= len(xdata)
    for i in range( npoints ):
        subcovm= matrix( [ [ xerrs[i]**2, xyrho[i]*xerrs[i]*yerrs[i] ], 
                           [ xyrho[i]*xerrs[i]*yerrs[i], yerrs[i]**2 ] ] )
        covm[2*i:2*i+2,2*i:2*i+2]= subcovm
        data.append( xdata[i] )
        data.append( ydata[i] )
    print covm
    print data

    # Fit parameters for straight line:
    upar= [ 1.0, 0.5 ]
    upnames= { 0: "a", 1: "b" }
    #upar= [ 0.0, 1.0, 1.0 ]
    #upnames= { 0: "a", 1: "b", 2: "c" }

    # Constraint function forces y_i = a + b*x_i for every
    # pair of measurements x_i, y_i:
    def straightlineConstraints( mpar, upar ):
        constraints= []
        for i in range( npoints ):
            constraints.append( upar[0] + upar[1]*mpar[2*i] 
                                # + upar[2]*mpar[2*i]**2
                                - mpar[2*i+1] )
        return constraints

    # Setup the solver and solve:
    solver= clsq.clsqSolver( data, covm, upar, straightlineConstraints,
                             uparnames=upnames )
    print "Constraints before solution"
    print solver.getConstraints()
    lBlobel= False
    lCorr= False
    if "b" in opt:
        lBlobel= True
    if "corr" in opt:
        lCorr= True
    solver.solve( lBlobel=lBlobel )
    solver.printResults( corr=lCorr )
    if "m" in opt:
        _doMinos( solver, "u" )
    if "cont" in opt:
        _doContour( solver, ipar1=0, type1="u", ipar2=1, type2="u" )

    return

    # Plot:
    global tg, lell, tf, tt
    from ROOT import TGraph, TF1, TText
    from array import array
    xarr= array( "f", xdata )
    yarr= array( "f", ydata )
    tg= TGraph( npoints, xarr, yarr )
    tg.SetMarkerStyle( 20 )
    tg.SetMinimum( 0.0 )
    tg.SetMaximum( 9.0 )
    tg.SetTitle( "straight line 2D fit" )
    xa= tg.GetXaxis()
    ya= tg.GetYaxis()
    xa.SetTitle( "X" )
    ya.SetTitle( "Y" )
    tg.Draw( "ap" )
    lell= []
    for i in range( npoints ):
        te= _makeEllipse( xdata[i], ydata[i], xerrs[i], yerrs[i], xyrho[i] )
        lell.append( te )
        te.Draw( "s" )
    solution= solver.getUpar()
    tf= TF1( "tf", "[0]+[1]*x", 0.0, 15.0 )
    for i in range( len(upar) ):
        tf.SetParameter( i, solution[i] )
        tf.SetParName( 0, upnames[i] )
    tf.Draw( "same" )
    tt= TText( 1, 8, "y= a + b*x" )
    tt.Draw( "same" )

    return


# Blobels triangle example, see Blobels Terascale Analysis Center 
# lecture 20 Jan 2010, pg. 11. Essentially, fit triangle area A 
# from measured sides a, b, c and angle gamma
# a = 10 +/- 0.05
# b =  7 +/- 0.2
# c =  9 +/- 0.2
# gamma = 1 +/- 0.02
# Area s = sqrt(p(p-a)(p-b)(p-c)) = p(p-c)tan(gamma/2) with p = (a+b+c)/2
# Two constraints: 1) tan(gamma/2)= s/(p(p-c))
#                  2) A = s
# Four measured (a,b,c,gamma) and one unmeasured parameter (A).
# The constraints vec(f) are brought into the normal form vec(f)=0, i.e
# in triangleConstrFun the two constraints are calculated to return 0
# S. Kluth 2012

def Triangle( opt="" ):

    from ConstrainedFit import clsq
    from math import sqrt, tan

    data= [ 10.0, 7.0, 9.0, 1.0 ]
    errors= [ 0.05, 0.2, 0.2, 0.02 ]
    covm= clsq.covmFromErrors( errors )
    mpnames= { 0: "a", 1: "b", 2: "c", 3: "gamma" }

    upar= [ 30.0 ]
    upnames= { 0: "A" }

    def triangleConstrFun( mpar, upar ):
        a= mpar[0]
        b= mpar[1]
        c= mpar[2]
        gamma= mpar[3]
        aa= upar[0]
        p= (a+b+c)/2.0
        s= sqrt( p*(p-a)*(p-b)*(p-c) )
        return [ tan(gamma/2.0)-s/(p*(p-c)), aa-s ]

    solver= clsq.clsqSolver( data, covm, upar, triangleConstrFun,
                             uparnames=upnames, mparnames=mpnames )
    print "Constraints before solution"
    print solver.getConstraints()
    lBlobel= False
    lCorr= False
    if "b" in opt:
        lBlobel= True
    solver.solve( lBlobel=lBlobel )
    solver.printResults( corr=True )

    if "m" in opt:
        _doMinos( solver )
    if "c" in opt:
        _doContour( solver, ipar1=0, type1="u", ipar2=3, type2="m" )

    return

def _doMinos( solver, opt="um" ):
    print "\nMinos error profiling:"
    fmtstr= "{0:>10s}: {1:10.4f} + {2:6.4f} - {3:6.4f}"
    if "u" in opt:
        results= solver.getUpar()
        upnames= solver.getUparNames()
        for ipar in range( len(results) ):
            errhi, errlo= solver.minosUpar( ipar )
            print fmtstr.format( upnames[ipar], results[ipar], errhi, abs(errlo) )
    if "m" in opt:
        results= solver.getMpar()
        mpnames= solver.getMparNames()
        for ipar in range( len(results) ):
            errhi, errlo= solver.minosMpar( ipar )
            print fmtstr.format( mpnames[ipar], results[ipar], errhi, abs(errlo) )
    return

def _doContour( solver, ipar1=0, type1="u", ipar2=1, type2="m" ):
    from array import array
    from ROOT import TGraph, TMultiGraph
    def getUMParErrName( pindx, ptype ):
        par= None
        err= None
        name= None
        if ptype == "u":
            par= solver.getUpar()[pindx]
            err= solver.getUparErrors()[pindx]
            name= solver.getUparNames()[pindx]
        elif ptype == "m":
            par= solver.getMpar()[pindx]
            err= solver.getMparErrors()[pindx]
            name= solver.getMparNames()[pindx]
        else:
            print "getUMPar: error, ptype not recognised:", ptype
        return par, err, name
    par1, err1, name1= getUMParErrName( ipar1, type1 )
    par2, err2, name2= getUMParErrName( ipar2, type2 )
    print "\nContour plot " + name1 + " - " + name2 + ":"
    global te1, te2, te3, tg1, tg2, tg3, tmg
    corr= solver.getCorrMatrix()
    icorr1= ipar1
    icorr2= ipar2
    if type1 == "u" or type2 == "u":
        nmpar= len(solver.getMpar())
        if type1 == "u":
            icorr1= nmpar + ipar1
        if type2 == "u":
            icorr2= nmpar + ipar2
    rho= corr[icorr1,icorr2]
    te1= _makeEllipse( par1, par2, err1, err2, rho )
    te2= _makeEllipse( par1, par2, 2.0*err1, 2.0*err2, rho )
    te3= _makeEllipse( par1, par2, 3.0*err1, 3.0*err2, rho )
    vx1, vy1= solver.contour( ipar1, type1, ipar2, type2, delta=1.0 )
    vx2, vy2= solver.contour( ipar1, type1, ipar2, type2, delta=4.0 )
    vx3, vy3= solver.contour( ipar1, type1, ipar2, type2, delta=9.0 )
    ax1= array( "d", vx1 )
    ay1= array( "d", vy1 )
    ax2= array( "d", vx2 )
    ay2= array( "d", vy2 )
    ax3= array( "d", vx3 )
    ay3= array( "d", vy3 )
    n= len( vx1 )
    tg1= TGraph( n, ax1, ay1 )
    tg2= TGraph( n, ax2, ay2 )
    tg3= TGraph( n, ax3, ay3 )
    tg1.SetMarkerStyle( 20 )
    tg2.SetMarkerStyle( 20 )
    tg3.SetMarkerStyle( 20 )
    tmg= TMultiGraph()
    tmg.Add( tg1 )
    tmg.Add( tg2 )
    tmg.Add( tg3 )
    tmg.Draw( "ap" )
    xa= tmg.GetXaxis()
    ya= tmg.GetYaxis()
    xa.SetTitle( name1 )
    ya.SetTitle( name2 )
    te1.Draw( "s" )
    te2.Draw( "s" )
    te3.Draw( "s" )
    return


def _makeEllipse( x, y, dx, dy, rho, phimin=0.0, phimax=360.0 ):
    from math import atan, sqrt, cos, pi
    from ROOT import TEllipse
    sd= dx**2 + dy**2
    dd= dx**2 - dy**2
    if dd != 0.0:
        phi= atan( 2.0*rho*dx*dy/dd )/2.0
    else:
        phi= pi/2.0
    a= sqrt( (sd+dd/cos(2.0*phi))/2.0 )
    b= sqrt( (sd-dd/cos(2.0*phi))/2.0 )
    phideg= phi*180.0/pi
    te= TEllipse( x, y, a, b, phimin, phimax, phideg )
    te.SetFillStyle( 0 )
    return te


