
import clsq

def run():

    xabs= [ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 ]
    data= [ 1.1, 1.9, 2.9, 4.1, 5.1, 6.1, 6.9, 7.9, 9.1 ]
    errors= [ 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1 ]
    covm= []
    n= len( errors )
    for i in range( n ):
        row= n*[0]
        row[i]= errors[i]**2
        covm.append( row )
    upar= [ 0.1, 1.1 ]

    def linearConstrFun( mpar, upar, xv ):
        constraints= []
        for mparval,xval in zip(mpar,xv):
            constraints.append( upar[0]+upar[1]*xval - mparval )
        return constraints

    solver= clsq.clsqSolver( data, covm, upar, 
                             linearConstrFun, args=(xabs,) )
    print "Constraints before solution"
    print solver.getConstraints()
    solver.solve()
    solver.printResults()
    return


