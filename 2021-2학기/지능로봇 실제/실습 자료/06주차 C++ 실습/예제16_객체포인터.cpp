#include <iostream>
using namespace std;

class Circle {
	int radius;
public:
	Circle() { radius = 1; }        // 클래스 선언 및 구현?
	Circle(int r) { radius = r; }
	double getArea();
};

double Circle::getArea() {
	return 3.14 * radius * radius;
}

int main() {

	Circle donut;
	Circle pizza(30);

	// 객체 이름으로 멤버 접근
	cout << donut.getArea() << endl;

	// 객체 포인터로 멤버 접근
	Circle* p;
	p = &donut;
	cout << p->getArea() << endl;
	cout << (*p).getArea() << endl;

	p = &pizza;
	cout << p->getArea() << endl;
	cout << (*p).getArea() << endl;

}