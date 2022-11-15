#include <iostream>
using namespace std;

bool equal(int* p, int* q);

int main() {
	int a = 5, b = 6;

	if (equal(&a, &b)) cout << "equal";
	else cout << "not equal";

}

bool equal(int* p, int* q) {
	if (*p == *q) return true;
	else return false;
}