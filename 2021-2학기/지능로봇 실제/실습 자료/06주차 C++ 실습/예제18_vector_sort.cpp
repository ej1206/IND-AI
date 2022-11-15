#include <iostream>
#include <vector>
#include <algorithm>

int main() {
	std::vector<int> v;

	std::cout << "정수 5개를 입력하세요 -> ";

	for (int i = 0; i < 5; i++) {
		int n;
		std::cin >> n;
		v.push_back(n);
	}

	std::sort(v.begin(), v.end());  // sort 함수를 이용해서 v.begin() ~ v.end() 사이의 값을 오름차순으로 정렬

	std::vector<int>::iterator it;  // 벡터 내의 원소를 탐색하는 iterator 변수

	for (it = v.begin(); it != v.end(); it++) {
		std::cout << *it << ' ';    // 벡터의 모든 원소 출력
	}
	std::cout << std::endl;
}