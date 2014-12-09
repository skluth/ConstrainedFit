
# Blobels branching ratio example
# See Blobles lecture Terascale School on data combination and limit setting
# (DESY, Oct 2011)
# Input data from from attachment to school agenda
# S. Kluth 2012

def run( lBlobel=False ):

    from ConstrainedFit import clsq

    data= [ 0.265, 0.28, 0.37, 0.166, 0.42, 0.5, 0.20, 0.16, 
            0.72, 0.6, 0.37, 0.64, 0.45, 0.028, 10.0, 7.5 ]
    errors= [ 0.014, 0.05, 0.06, 0.013, 0.15, 0.2, 0.08, 0.08,
              0.15, 0.4, 0.16, 0.40, 0.45, 0.009, 5.0, 2.5 ]
    # Error scale factor a la Blobel
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
    solver.solve( lBlobel=lBlobel )
    solver.printResults( cov=True, corr=True )
    return

