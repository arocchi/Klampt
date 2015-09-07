"""Operations for rigid rotations in Klampt.  All rotations are
represented by a 9-list specifying the entries of the rotation matrix
in column major form.

In other words, given a 3x3 matrix
   [a11,a12,a13]
   [a21,a22,a23]
   [a31,a32,a33],
Klamp't represents the matrix as a list [a11,a21,a31,a12,a22,a32,a13,a23,a33].

The reasons for this representation are 1) simplicity, and 2) a more
convenient interface with C code.
"""

import math
import vectorops

def __str__(R):
    """Converts a rotation to a string."""
    return '\n'.join([' '.join([str(ri) for ri in r]) for r in matrix(R)])

def identity():
    """Returns the identity rotation"""
    return [1.,0.,0.,0.,1.,0.,0.,0.,1.]

def inv(R):
    """Inverts the rotation"""
    Rinv = [R[0],R[3],R[6],R[1],R[4],R[7],R[2],R[5],R[8]]
    return Rinv

def apply(R,point):
    """Applies the rotation to a point"""
    return (R[0]*point[0]+R[3]*point[1]+R[6]*point[2],
            R[1]*point[0]+R[4]*point[1]+R[7]*point[2],
            R[2]*point[0]+R[5]*point[1]+R[8]*point[2])

def matrix(R):
    """Returns the 3x3 rotation matrix corresponding to R"""
    return [[R[0],R[3],R[6]],
            [R[1],R[4],R[7]],
            [R[2],R[5],R[8]]]

def from_matrix(mat):
    """Returns an R corresponding to the 3x3 rotation matrix mat"""
    R = [mat[0][0],mat[1][0],mat[2][0],mat[0][1],mat[1][1],mat[2][1],mat[0][2],mat[1][2],mat[2][2]]
    return R

def mul(R1,R2):
    """Multiplies two rotations."""
    m1=matrix(R1)
    m2T=matrix(inv(R2))
    mres = matrix(identity())
    for i in xrange(3):
        for j in xrange(3):
            mres[i][j] = vectorops.dot(m1[i],m2T[j])
    #print "Argument 1"
    #print __str__(R1)
    #print "Argument 2"
    #print __str__(R2)
    #print "Result"
    R = from_matrix(mres)
    #print __str__(R)
    return R

def trace(R):
    """Computes the trace of the rotation matrix."""
    return R[0]+R[4]+R[8]

def angle(R):
    """Returns absolute deviation of R from identity"""
    ctheta = (trace(R) - 1.0)*0.5
    return math.acos(max(min(ctheta,1.0),-1.0))

def moment(R):
    """Returns the moment w (exponential map) representation of R such
    that e^[w] = R.  Equivalent to axis-angle representation with
    w/||w||=axis, ||w||=angle."""
    theta = angle(R)
    if abs(theta-math.pi)<1e-5:
        #can't do normal version because the scale factor reaches a singularity
        x2=(R[0]+1.)*0.5
        y2=(R[4]+1.)*0.5
        z2=(R[8]+1.)*0.5
        if x2 < 0:
            assert(x2>-1e-5)
            x2=0
        if y2 < 0:
            assert(y2>-1e-5)
            y2=0
        if z2 < 0:
            assert(z2>-1e-5)
            z2=0
        x = math.pi*math.sqrt(x2)
        y = math.pi*math.sqrt(y2)
        z = math.pi*math.sqrt(z2)
        #determined up to sign changes, we know r12=2xy,r13=2xz,r23=2yz
        xy=R[3]
        xz=R[6]
        yz=R[7]
        if(x > y):
            if(x > z):
                #x is largest
                if(xy < 0): y=-y
                if(xz < 0): z=-z
            else:
                #z is largest
                if(yz < 0): y=-y
                if(xz < 0): x=-x
        else:
            if(y > z):
                #y is largest
                if(xy < 0): x=-x
                if(yz < 0): z=-z
            else:
                #z is largest
                if(yz < 0): y=-y
                if(xz < 0): x=-x
        return [x,y,z]
    #normal
    scale = 0.5
    if abs(theta) > 1e-5:
        scale = 0.5*theta/math.sin(theta)
    x = (R[3+2]-R[6+1]) * scale;
    y = (R[6+0]-R[0+2]) * scale;
    z = (R[0+1]-R[3+0]) * scale;
    return [x,y,z]

def axis_angle(R):
    """Returns the (axis,angle) pair representing R"""
    m = moment(R)
    return (vectorops.unit(m),vectorops.norm(m))

def from_axis_angle(aa):
    """Converts an axis-angle representation (axis,angle) to a 3D rotation
    matrix."""
    return rotation(aa[0],aa[1])

def from_moment(w):
    """Converts a moment representation w to a 3D rotation matrix."""
    length = vectorops.norm(w)
    if length < 1e-7: return identity()
    return rotation(vectorops.mul(w,1.0/length),length)

def from_quaternion(q):
    """Given a unit quaternion (w,x,y,z), produce the corresponding rotation
    matrix."""
    w,x,y,z = q
    x2 = x + x; y2 = y + y; z2 = z + z;
    xx = x * x2;   xy = x * y2;   xz = x * z2;
    yy = y * y2;   yz = y * z2;   zz = z * z2;
    wx = w * x2;   wy = w * y2;   wz = w * z2;

    a11 = 1.0 - (yy + zz)
    a12 = xy - wz
    a13 = xz + wy
    a21 = xy + wz
    a22 = 1.0 - (xx + zz)
    a23 = yz - wx
    a31 = xz - wy
    a32 = yz + wx
    a33 = 1.0 - (xx + yy)
    return [a11,a21,a31,a12,a22,a32,a13,a23,a33]

def quaternion(R):
    """Given a Klamp't rotation representation, produces the corresponding
    unit quaternion (x,y,z,w)."""
    tr = trace(R) + 1.0;
    a11,a21,a31,a12,a22,a32,a13,a23,a33 = R

    #If the trace is nonzero, it's a nondegenerate rotation
    if tr > 1e-5:
        s = math.sqrt(tr)
        w = s * 0.5
        s = 0.5 / s
        x = (a32 - a23) * s
        y = (a13 - a31) * s
        z = (a21 - a12) * s
        return vectorops.unit((w,x,y,z))
    else:
        #degenerate it's a rotation of 180 degrees
        nxt = [1, 2, 0]
        #check for largest diagonal entry
        i = 0
        if a22 > a11: i = 1
        if a33 > max(a11,a22): i = 2
        j = nxt[i]
        k = nxt[j]
        M = matrix(R)

        q = [0.0]*4
        s = math.sqrt((M[i][i] - (M[j][j] + M[k][k])) + 1.0);
        q[i] = s * 0.5
    
        if abs(s)<1e-7:
            raise ValueError("Could not solve for quaternion... Invalid rotation matrix?")
        else:
            s = 0.5 / s;
            q[3] = (M[k][j] - M[j][k]) * s;
            q[j] = (M[i][j] + M[j][i]) * s;
            q[k] = (M[i][k] + M[i][k]) * s;
        x,y,z,w = q[3],q[0],q[1],q[2]
        return vectorops.unit([w,x,y,z])
    
def distance(R1,R2):
    """Returns the absolute angle one would need to rotate in order to get
    from R1 to R2"""
    R = mul(R1,inv(R2))
    return angle(R)

def error(R1,R2):
    """Returns a 3D "difference vector" that describes how far R1 is from R2.
    More precisely, this is the Lie derivative."""
    R = mul(R1,inv(R2))
    return moment(R)

def cross_product(w):
    """Returns the cross product matrix associated with w.

    The matrix [w]R is the derivative of the matrix R as it rotates about
    the axis w/||w|| with angular velocity ||w||.
    """
    return [0.,w[2],-w[1],  -w[2],0.,w[0],  w[1],-w[0],0.]

def rotation(axis,angle):
    """Given a unit axis and an angle in radians, returns the rotation
    matrix."""
    cm = math.cos(angle)
    sm = math.sin(angle)

    #m = s[r]-c[r][r]+rrt = s[r]-c(rrt-I)+rrt = cI + rrt(1-c) + s[r]
    R = vectorops.mul(cross_product(axis),sm)
    for i in xrange(3):
        for j in xrange(3):
            R[i*3+j] += axis[i]*axis[j]*(1.-cm)
    R[0] += cm
    R[4] += cm
    R[8] += cm
    return R

def canonical(v):
    """Given a unit vector v, finds R that defines a basis [x,y,z] such that
    x = v and y and z are orthogonal"""
    if abs(vectorops.normSquared(v) - 1.0) > 1e-4:
        raise RuntimeError("Nonunit vector supplied to canonical()")
    assert(len(v)==3)
    if abs(v[0]-1.0) < 1e-5:
        return identity()
    elif abs(v[0]+1.0) < 1e-5:
        #flip of basis
        R = identity()
        R[0] = -1.0
        R[4] = -1.0
        return R
    R = v + [0.]*6
    (x,y,z) = tuple(v)
    scale = (1.0-x)/(1.0-x*x);
    R[3]= -y;
    R[4]= x + scale*z*z;
    R[5]= -scale*y*z;
    R[6]= -z;
    R[7]= -scale*y*z;
    R[8]= x + scale*y*y;
    return R

def vector_rotation(v1,v2):
    """Finds the minimal-angle matrix that rotates v1 to v2.  v1 and v2
    are assumed to be nonzero"""
    a1 = vectorops.unit(v1)
    a2 = vectorops.unit(v2)
    cp = vectorops.cross(a1,a2)
    dp = vectorops.dot(a1,a2)
    if abs(vectorops.norm(cp)) < 1e-4:
        if dp < 0:
            R0 = canonical(a1)
            #return a rotation 180 degrees about the canonical y axis
            return rotation(R0[3:6],math.pi)
        else:
            return identity()
    else:
        angle = math.acos(max(min(dp,1.0),-1.0))
        axis = vectorops.mul(cp,1.0/vectorops.norm(cp))
        return rotation(axis,angle)

def interpolate(R1,R2,u):
    """Interpolate linearly between the two rotations R1 and R2. """
    R = mul(inv(R1),R2)
    m = moment(R)
    angle = vectorops.norm(m)
    if angle==0: return R1
    axis = vectorops.div(m,angle)
    return mul(R1,rotation(axis,angle*u))
