#include <iostream>
using namespace std;

class Circle {
public:
	int radius;
	Circle();      // 매개변수가 없는 생성자 // 객체가 생성되는 시점에서 자동으로 호출되는 멤버함수,클래스 이름과 동일
	Circle(int r); // 매개변수가 있는 생성자
	double getArea();
};

Circle::Circle() {
	radius = 1;
	cout << "반지름 : " << radius << ", 원 생성" << endl;
}

Circle::Circle(int r) {
	radius = r;
	cout << "반지름 : " << radius << ", 원 생성" << endl;
}

double Circle::getArea() {
	return 3.14 * radius * radius;
}

int main() {

	Circle donut;    // 매개변수가 없는 생성자 호출
	double area = donut.getArea();
	cout << "donut 면적은 " << area << endl;

	Circle pizza(30); // 매개변수가 있는 생성자 호출
	area = pizza.getArea();
	cout << "pizza 면적은 " << area << endl;

}
