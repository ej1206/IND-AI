#include <iostream>
using namespace std;

int main() {

	int width;
	int heihgt;

	cout << "너비를 입력하세요 -> ";
	cin >> width;

	cout << "높이를 입력하세요 -> ";
	cin >> heihgt;

	int area = width * heihgt;

	cout << "면적은 " << area;

}
