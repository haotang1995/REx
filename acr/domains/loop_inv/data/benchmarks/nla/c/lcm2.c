#include <stdio.h>
#include <assert.h>

int mainQ(int a, int b){
     assert(a>=1);
     assert(b>=1);
     int x,y,u,v;

     x=a;
     y=b;
     u=b;
     v=a;

	// Adding procedure to track correctness of GCD
     p=1;
     q=0;
     r=0;
     s=1;

     // loop 1
     while(1) {
	  //assert(x*u + y*v == 2*a*b);
       //assert((x >= 1) && (y >= 1));
	  //assert(1 == p*s - r*q);
	  //assert(x == b*r + a*p);
	  //assert(y == a*q + b*s);
       
       // assert(GCD(x,y) == GCD(a,b));
	  //%%%traces: int a, int b, int x, int y, int u, int v

	  if (!(x!=y)) break;

	  if (x>y){
	       x=x-y;
	       v=v+u;

		  p = p-q; 
	       r = r-s;
	  }
	  else {
	       y=y-x;
	       u=u+v;

	       q = q-p; 
	       s = s-r;
	  }
     }

     //x==gcd(a,b)
     int r = (u+v)/2;
     return r; //lcm

}


int main(int argc, char **argv){
     mainQ(atoi(argv[1]), atoi(argv[2]));
     return 0;
}

