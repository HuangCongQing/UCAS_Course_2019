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

x = np.array(  [10, 10, 240, 200, 300, 5, 5]  ); 
y = np.array(  [1, 1, 1, -60, -60 ] ); 
s = np.zeros( n ); 
s = c - np.dot( np.transpose(A), y );

alpha = 0.995; 
epsilon = 0.001;

num = 0; 
theta = 0;
print("  iter	x1	x2	theta	max(xi*si)	min(xi*si)	#");
while ( True ): 
	found = True;
	print("============== iter: %d ============" % (num) );
	maxxs = max( x * s ); 
	minxs = min( x * s );

	print("  %d	%lf	%lf	%lf	%lf	%lf		#" % (num, x[0], x[1], theta, maxxs, minxs) );
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

	AXSinvAt = np.dot( np.dot( A, XSinv ), np.transpose(A) );

	dy = np.linalg.solve( AXSinvAt, b ); 
	dx = -x + np.dot( np.dot( XSinv, np.transpose(A) ), dy );
	ds = -s - np.dot( XinvS, dx );

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


