#include <iostream>
using namespace std;

class Circle {
public:
	int radius;
	Circle();      // 매개변수가 없는 생성자
	Circle(int r); // 매개변수가 있는 생성자
	~Circle();     // 소멸자 객체가 소멸되는 시점에서 자동으로 호출 / 하나만 존재
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

Circle:: ~Circle() {
	cout << "반지름 : " << radius << ", 원 소멸" << endl;
}

double Circle::getArea() {
	return 3.14 * radius * radius;
}

int main() {

	Circle donut;     // 매개변수가 없는 생성자 호출
	Circle pizza(30); // 매개변수가 있는 생성자 호출

	return 0;         // 메인함수 종료 시 donut, pizza 객체 소멸 / 생성의 반대순으로 소멸

}