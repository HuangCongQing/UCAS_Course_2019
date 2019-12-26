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

x = np.array(  [12, 1, 72, 120, 312, 1, 14]  ); 
y = np.array(  [0.01, 0.01, 0.01, -1, -1 ] ); 
s = np.zeros( n ); 
e = np.array( [ 1 ] * n );

alpha = 0.95; 
epsilon = 0.001;
beta = 0.8;

num = 0; 
theta = 0;

print("  iter	x1	x2	theta	max(xi*si)	min(xi*si)	mu	#");
while ( True ): 
	found = True;
	print("============== iter: %d ============" % (num) );
	s = c - np.dot( np.transpose(A), y );

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
		
	X = np.zeros( shape=(n,n) ); 
	for i in range(n) : 
		X[i, i] = x[i]; 

	Xinv = np.zeros( shape=(n,n) ); 
	for i in range(n) : 
		Xinv[i, i] = 1.0/x[i]; 

	X2 = np.zeros( shape=(n,n) ); 
	for i in range(n) : 
		X2[i, i] = x[i]*x[i]; 

	AX2c = np.dot( A, np.dot( X2, c) );
	AX2Atr = np.dot( A, np.dot( X2, np.transpose(A)) );
	AX2Atry = np.dot( AX2Atr, y );
	muAXe = mu * np.dot( A, np.dot(X, e) );

	dy = np.linalg.solve( AX2Atr,  -AX2c + AX2Atry + muAXe );

	X2c = np.dot( X2, c); 
	X2Atry = np.dot( X2, np.dot( np.transpose(A), y) );
	muXe = mu * np.dot( X, e); 
	X2Atr = np.dot( X2, np.transpose(A) );

	dx = 1/mu * ( X2c - X2Atry - muXe + np.dot(X2Atr, dy) );

	for i in range(n):
		print("dx[%d]:	%lf" % (i, dx[i] ) );
	for i in range(m):
		print("dy[%d]:	%lf" % (i, dy[i] ) );

	thetax = -1; 
	for i in range(n) :
		if (dx[i] < 0 ):
			t = x[i] / (-dx[i]); 
			print("t[%d]:	%lf	for dx" % (i, t) );
			if ( t < thetax or thetax < 0 ):
				thetax = t; 

	Atrdy = np.dot( np.transpose(A), dy); 

	s = s - mu * np.dot( Xinv, e );
	thetay = 9999999; 
	for i in range(n):
		if(Atrdy[i] > 0 ):
			t = s[i] / (Atrdy[i]); 
			print("t[%d]:	%lf	for dy" % (i, t) );
			if(t < thetay ):
				thetay = t;

	thetax = min( 1, thetax*alpha );
	thetay = min( 1, thetay*alpha );

	print("thetax:	%lf	thetay:	%lf	theta:	%lf" % (thetax, thetay, theta) );
	x = x + thetax * dx; 
	y = y + thetay * dy; 


