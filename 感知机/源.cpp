
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include<iostream>
using namespace std;
#define db double
double z7[10];
double a7[10];
double output[10];
db tFun()
{
	db SUM = 0;
	db tMax = -999999;
	//找出z7最大的
	for (int i = 0; i < 10; i++)
		if (z7[i] > tMax)
			tMax = z7[i];
	for (int i = 0; i < 10; i++)
	{
		z7[i] -= tMax;
		SUM += exp(z7[i]);
	}

	for (int i = 0; i < 10; i++)
		a7[i] = exp(z7[i]) / SUM;
	db j = 0;
	for (int i = 0; i < 10; i++)
		j += (a7[i] - output[i]) * (a7[i] - output[i]) / 2;
	return j;
}
int main()
{
	int x = log10(21);
	cout << pow(10, x);
	return 0;
}