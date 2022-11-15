#include <iostream>
using namespace std;

int main() {
	int n[10];
	int i;
	int* p;

	for (i = 0; i < 10; i++) {
		*(n + i) = i * 3;         // 배열 이름 = 주소
	}

	p = n;                        // 포인터 p 에 배열의 주소값 부여
	for (i = 0; i < 10; i++) {
		cout << *(p + i) << ' ';  // p로 배열의 원소에 접근
	}
	cout << '\n';

	for (i = 0; i < 10; i++) {
		*p = *p + 2;              // 배열의 원소값에 2를 더해주기
		p++;                      // 다음 주소로 이동
	}

	for (i = 0; i < 10; i++) {
		cout << n[i] << ' ';
	}

}