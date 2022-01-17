#include<bits/stdc++.h>

using namespace std;
typedef long long ll;

int main(){
	srand((unsigned)time(NULL));

	ll in_circle = 0;
	ll toss = 10000000;
	for(ll i=0;i<toss;++i){
		double x = ((double)rand() / RAND_MAX) * 2 - 1;
		double y = ((double)rand() / RAND_MAX) * 2 - 1;
		double dis = x * x + y * y;
		if(dis <= 1){
			in_circle++;
		}
	}
	cout << 4 * in_circle / (double)toss << endl;

return 0;
}

