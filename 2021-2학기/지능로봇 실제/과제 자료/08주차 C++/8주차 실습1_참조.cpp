#include <iostream>
using namespace std;

int main() {

	cout << "i" << '\t' << "n" << '\t' << "refn" << endl;

	int i = 1;
	int n = 2;
	int& refn = n;  // 참조 변수 선언 / refn은 n의 별명이기 때문에 둘 값이 같다.
	n = 4;
	refn++;
	cout << i << '\t' << n << '\t' << refn << endl;

	refn = i;
	refn++;
	cout << i << '\t' << n << '\t' << refn << endl;

	int* p = &refn; // 참조 포인터 변수 선언
	*p = 20;
	cout << i << '\t' << n << '\t' << refn << endl;

}
