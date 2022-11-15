#include <iostream>
using namespace std;

int main() {
	int i, a, b, sum = 0;
	cout << "두 개의 정수 입력 ->";
	cin >> a >> b;

	i = a;
	while (i <= b) {
		sum += i;
		i++;
	}

	cout << a << "에서 " << b << "까지 합은 " << sum;
}