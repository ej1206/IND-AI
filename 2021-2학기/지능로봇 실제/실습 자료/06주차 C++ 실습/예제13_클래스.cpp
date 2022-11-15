#include <iostream>
using namespace std;

// 클래스 선언
class Circle {
public:
	int radius;
	double getArea();
};

// 클래스 구현
double Circle::getArea() {
	return 3.14 * radius * radius;
}

int main() {

	Circle donut;                   // donut 객체 생성
	donut.radius = 1;               // donut 멤버 변수 접근
	double area = donut.getArea();  // donut 멤버 함수 호출
	cout << "donut 면적은 " << area << endl;

	Circle pizza;
	pizza.radius = 30;
	area = pizza.getArea();
	cout << "pizza 면적은 " << area << endl;

}
