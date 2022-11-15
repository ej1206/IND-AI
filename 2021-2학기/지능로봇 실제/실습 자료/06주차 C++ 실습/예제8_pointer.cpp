#include <iostream>
using namespace std;

int main() {
	int n = 10, m;
	char c = 'A';
	double d;

	int* p = &n;
	char* q = &c;
	double* r = &d;

	*p = 25;     // n狼 林家俊 25 历厘 -> n = 25 
	*q = 'A';    // c狼 林家俊 'A' 历厘
	*r = 3.14;   // d狼 林家俊 3.14 历厘

	m = *p + 10;

	cout << n << ' ' << *p << '\n';
	cout << c << ' ' << *q << '\n';
	cout << d << ' ' << *r << '\n';
	cout << m;
}