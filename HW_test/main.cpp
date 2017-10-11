#include "main_kernel.cuh"
#include <iostream>

int main()
{
	std::cout << "Hello World!" << std::endl;
	evaluateComponents(NULL, NULL, NULL, 0);
	return 0;
}