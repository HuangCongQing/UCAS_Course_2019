import numpy as np; 

m=5;
n=7; 

c = np.array(  [2, 1.5, 0, 0, 0, 0, 0]  );
A = np.array( [ [12, 24, -1, 0, 0, 0, 0], 
		[16, 16, 0, -1, 0, 0, 0],
		[30, 12, 0, 0, -1, 0, 0],
		[1, 0, 0, 0, 0, 1, 0],
		[0, 1, 0, 0, 0, 0, 1] ] );
b = np.array(  [ 120, 120, 120, 15, 15]  ); 

x = np.array(  [14, 1, 72, 120, 312, 1, 14]  ); 
y = np.array(  [0.01, 0.01, 0.01, -1, -1 ] ); 
s = np.zeros( n ); 

s[0] = 2 - 12 * y[0] - 16*y[1] - 30*y[2] -y[3]; 
s[1] = 1.5 - 24*y[0] - 16*y[1] - 12*y[2] -y[4]; 
s[2] = y[0]; 
s[3] = y[1]; 
s[4] = y[2]; 
s[5] = -y[3]; 
s[6] = -y[4];

alpha = 0.995; 
epsilon = 0.001;
beta = 0.7;

num = 0; 
theta = 0;

print("  iter	x1	x2	theta	max(xi*si)	min(xi*si)	mu	#");
while ( True ): 
	found = True;
	print("============== iter: %d ============" % (num) );
	maxxs = max( x * s ); 
	minxs = min( x * s );

	mu = beta * np.dot( x, s) / n; 

	print("  %d	%lf	%lf	%lf	%lf	%lf	%lf	#" % (num, x[0], x[1], theta, maxxs, minxs, mu) );
	num = num + 1; 
	for i in range(n) :
		print("x[%d]:	%lf	s[%d]:	%lf	x*s[%d]:	%lf	" % (i, x[i], i, s[i], i, x[i]*s[i] ) );
		if ( x[i] * s[i] > epsilon ):
			found = False;


	if (found == True ):
		break;
		
	XSinv = np.zeros( shape=(n,n) ); 
	for i in range(n) : 
		XSinv[i, i] = x[i] / s[i]; 

	XinvS = np.zeros( shape=(n,n) ); 
	for i in range(n) : 
		XinvS[i, i] = s[i] / x[i]; 

	muSinvE = np.zeros( n ); 
	muXinvE = np.zeros( n ); 

	for i in range(n):
		muXinvE[i] = mu / x[i]; 
	for i in range(n): 
		muSinvE[i] = mu / s[i]; 

	AXSinvAt = np.dot( np.dot( A, XSinv ), np.transpose(A) );

	dy = np.linalg.solve( AXSinvAt, b - np.dot(A, muSinvE) ); 
	dx = -x + muSinvE +  np.dot( np.dot( XSinv, np.transpose(A) ), dy );
	ds = -s - np.dot( XinvS, dx ) + muXinvE;

	for i in range(n):
		print("dx[%d]:	%lf" % (i, dx[i] ) );
	for i in range(m):
		print("dy[%d]:	%lf" % (i, dy[i] ) );
	for i in range(n):
		print("ds[%d]:	%lf" % (i, ds[i] ) );

	thetax = -1; 
	for i in range(n) :
		if (dx[i] < 0 ):
			t = x[i] / (-dx[i]); 
			print("t[%d]:	%lf" % (i, t) );
			if ( t < thetax or thetax < 0 ):
				thetax = t; 
	
	thetas = -1;
	for i in range(n) :
		if (ds[i] < 0 ):
			t = s[i] / (-ds[i]); 
			if ( t < thetas or thetas < 0 ):
				thetas = t; 

	theta = min( 1, thetax*alpha, thetas*alpha);

	print("thetax:	%lf	thetas:	%lf	theta:	%lf" % (thetax, thetas, theta) );
	x = x + theta * dx; 
	s = s + theta * ds; 
	y = y + theta * dy; 


