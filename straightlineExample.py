
# Straight line fit to points with errors in x and y
# Example from Blobels lecture pg 29 with values
# estimated from the plot
# S Kluth 12.12.2014


def run( lBlobel=False ):

    from ConstrainedFit import clsq
    from numpy import matrix, zeros

    # data errors and correlations:
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
    solver.solve( lBlobel=lBlobel )
    solver.printResults( corr=True )

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
        te= makeEllipse( xdata[i], ydata[i], xerrs[i], yerrs[i], xyrho[i] )
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


def makeEllipse( x, y, dx, dy, rho, phimin=0.0, phimax=360.0 ):
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
