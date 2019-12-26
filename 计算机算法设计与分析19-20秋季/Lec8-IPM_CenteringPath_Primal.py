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
y = np.zeros( m );
s = np.zeros( n ); 
e = np.array( [1] * n);

alpha = 0.995; 
epsilon = 0.000000001;
beta = 0.3;

num = 0; 
theta = 0;
mu = 10;

print("  iter	x1	x2	theta	max(xi*si)	min(xi*si)	mu	#");
while ( True ): 
	found = True;
	print("============== iter: %d ============" % (num) );

	X = np.zeros( shape=(n,n) ); 
	for i in range(n) : 
		X[i, i] = x[i]; 

	X2 = np.zeros( shape=(n,n) ); 
	for i in range(n) : 
		X2[i, i] = x[i]*x[i]; 

	AX2c = np.dot( A, np.dot( X2, c) );
	AX2Atr = np.dot( A, np.dot( X2, np.transpose(A)) );
	muAXe = mu * np.dot( A, np.dot(X, e) );

	y = np.linalg.solve( AX2Atr, AX2c - muAXe );

	s = c - np.dot( np.transpose(A), y );
	maxxs = max( x * s ); 
	minxs = min( x * s );


	print("  %d	%lf	%lf	%lf	%lf	%lf	%lf	#" % (num, x[0], x[1], theta, maxxs, minxs, mu) );

	mu = mu * 0.9; 
	num = num + 1; 
	for i in range(n) :
		print("x[%d]:	%lf	s[%d]:	%lf	x*s[%d]:	%lf	" % (i, x[i], i, s[i], i, x[i]*s[i] ) );
		if ( abs( x[i] * s[i]) > epsilon ):
			found = False;

	if (found == True ):
		break;



	X2c = np.dot( X2, c); 
	X2Atry = np.dot( X2, np.dot( np.transpose(A), y) );

	dx = x + 1/mu * ( X2Atry - X2c );

	for i in range(n):
		print("dx[%d]:	%lf" % (i, dx[i] ) );

	thetax = -1; 
	for i in range(n) :
		if (dx[i] < 0 ):
			t = x[i] / (-dx[i]); 
			print("t[%d]:	%lf" % (i, t) );
			if ( t < thetax or thetax < 0 ):
				thetax = t; 

	theta = min( 1, thetax*alpha);


	print("thetax:	%lf	theta:	%lf" % (thetax, theta) );
	x = x + theta * dx; 


