#include <iostream>
using namespace std;

int main() {
	int a, b, sum;
	
	cout << "두 정수를 입력하세요 -> ";
	cin >> a >> b;

	sum = a + b;

	cout << "합은 " << sum << "\n";
	cout << "합은 " << add(a, b) <<"\n";
	cout << "전역변수 g는 " << g;
}

