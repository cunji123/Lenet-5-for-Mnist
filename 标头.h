/*
         CNN-Lenet5卷积神经网络手写数字识别
		 CaiJun
		 由于参数10M，放在堆区需要拓展堆大小，所以采用面向过程设计，所有参数放在全局区
		 有任何问题请联系hub主
		 目前正确率0.66，待解决问题：局部最优
*/
#pragma once
#include<opencv2/opencv.hpp>
#include <windows.h>
#include<iostream>
#include<time.h>
#include<math.h>
#include <thread>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
using namespace std;
using namespace cv;
#define db double
int showPD;

//隐藏层神经元，z代表激活前，a代表激活后
//所有设置和Lenet-5论文中一模一样，实在是注释不清晰的地方可以完全参考Lenet-5（除了激活函数和损失函数以外，参考个人毕设）
db input[32][32];       //32*32输入
db z1[6][28][28];         //6个28*28矩阵
db a1[6][28][28];         //6个28*28矩阵

db z2[6][14][14];      //6个14*14矩阵

db z3[16][10][10];     //16个10*10矩阵
db a3[16][10][10];     //16个10*10矩阵

db z4[16][5][5];      //16个5*5矩阵
db z4_dimReduc[400];  //一维化，与后面全连接

db z5[120];
db a5[120];

db z6[84];
db a6[84];

db a7[10];
db z7[10];

db output[10];
int maxPi;

db w1[6][5][5];        //6个5*5卷积核
db b1[6];

db w3[16][5][5];      //16个5*5卷积核
db b3[16];

db w5[120][400];   //120*400个参数
db b5[120];

db w6[84][120];      //84*120个参数
db b6[84];

db w7[10][84];       //10*84个参数


//反向传播中间值   
db J;//损失函数
db pd7[10];
db pd6[84];
db pd5[120];
db pd4[16][5][5];
db pd3[16][10][10];
db pd2[6][14][14];
db pd1[6][28][28];


db pd7_w[10][84];

db pd6_w[84][120];
db pd6_b[84];

db pd5_w[120][400];
db pd5_b[120];

db pd3_w[16][5][5];
db pd3_b[16];

db pd1_w[6][5][5];
db pd1_b[6];

double LearnProcess;//学习率
db weaky = 0.2;//Relu Puls参数
//函数

void lenet();//初始化参数 w，b
void init();//初始化除参数外所有： z，a，pd，pd_w,pd_b....
void setInput(int _input[28][28]);//设置输入矩阵，将图片中获取的28*28矩阵复制到32*32输入矩阵


void forward();//向前传播总函数
void backward();//反向传播求偏导总函数

//part
void fw_INtoL1();//正向传播：输入层到L1层，以下以此类推
void fw_L1toL2();
void fw_L2toL3();
void fw_L3toL4();
void fw_L4toL5();
void fw_L5toL6();
void fw_L6toL7();

void bw_L7();//求L7层神经元激活前偏导，以及该层参数偏导，以下以此类推
void bw_L6();
void bw_L5();
void bw_L4();
void bw_L3();
void bw_L2();
void bw_L1();


void update();//根据偏导更新参数

void func(int inputN, int corN);//由于L2到L3层是半无规律传播，所以设计此函数用于卷积求和累加
void func1(int inputN, int corN);//由于L2到L3层是半无规律传播，所以设计此函数用于求出L3到L2偏导
void visual();//测试函数
db Max(db x1, db x2, db x3, db x4);//输入4个数，返回最大数的值
int maxP(db x1, db x2, db x3, db x4);//输入4个数，返回最大数的序号，如x1最大，返回0




void lenet()
{
	

	//随机初始化权值
	for (int i = 0; i < 6; i++)
	{
		for (int j = 0; j < 5; j++)
			for (int k = 0; k < 5; k++)
				w1[i][j][k] = rand() * (2. / RAND_MAX) - 1;
		b1[i] = rand() * (2. / RAND_MAX) - 1;
	}

	for (int i = 0; i < 16; i++)
	{
		for (int j = 0; j < 5; j++)
			for (int k = 0; k < 5; k++)
				w3[i][j][k] = rand() * (2. / RAND_MAX) - 1;
		b3[i] = rand() * (2. / RAND_MAX) - 1;
	}

	for (int i = 0; i < 120; i++)
	{
		for (int j = 0; j < 400; j++)
			w5[i][j] = rand() * (2. / RAND_MAX) - 1;
		b5[i] = rand() * (2. / RAND_MAX) - 1;
	}

	for (int i = 0; i < 84; i++)
	{
		for (int j = 0; j < 120; j++)
			w6[i][j] = rand() * (2. / RAND_MAX) - 1;
		b6[i] = rand() * (2. / RAND_MAX) - 1;
	}

	for (int i = 0; i < 10; i++)
		for (int j = 0; j < 84; j++)
			w7[i][j] = rand() * (2. / RAND_MAX) - 1;


	LearnProcess = 0.01;

}
void init()
{

	for (int i = 0; i < 6; i++)
		for (int j = 0; j < 28; j++)
			for (int k = 0; k < 28; k++)
			{
				z1[i][j][k] = 0;
				pd1[i][j][k] = 0;
			}
	for (int i = 0; i < 6; i++)
		for (int j = 0; j < 14; j++)
			for (int k = 0; k < 14; k++)
				pd2[i][j][k] = 0;


	for (int i = 0; i < 16; i++)
		for (int j = 0; j < 10; j++)
			for (int k = 0; k < 10; k++)
			{
				z3[i][j][k] = 0;
				pd3[i][j][k] = 0;
			}
	for (int i = 0; i < 120; i++)
	{
		z5[i] = 0;
		pd5[i] = 0;
	}

	for (int i = 0; i < 84; i++)
	{
		z6[i] = 0;
		pd6[i] = 0;
	}

	for (int i = 0; i < 10; i++)
	{
		z7[i] = 0;
		pd7[i] = 0;
	}

	//初始化参数中间值

	for (int i = 0; i < 6; i++)
	{
		for (int j = 0; j < 5; j++)
			for (int k = 0; k < 5; k++)
				pd1_w[i][j][k] = 0;
		pd1_b[i] = 0;
	}

	for (int i = 0; i < 16; i++)
	{
		for (int j = 0; j < 5; j++)
			for (int k = 0; k < 5; k++)
				pd3_w[i][j][k] = 0;
		pd3_b[i] = 0;
	}

	for (int i = 0; i < 120; i++)
	{
		for (int j = 0; j < 400; j++)
			pd5_w[i][j] = 0;
		pd5_b[i] = 0;
	}

	for (int i = 0; i < 84; i++)
	{
		for (int j = 0; j < 120; j++)
			pd6_w[i][j] = 0;
		pd6_b[i] = 0;
	}

	for (int i = 0; i < 10; i++)
		for (int j = 0; j < 84; j++)
			pd7_w[i][j] = 0;

}

void forward()
{
	init();//初始中间值
	fw_INtoL1();
	fw_L1toL2();
	fw_L2toL3();
	fw_L3toL4();
	fw_L4toL5();
	fw_L5toL6();
	fw_L6toL7();


	//找到10个L7层神经元值最大的值，作为最终输出标签
	db max = 0;
	for (int i = 0; i < 10; i++)
		if (a7[i] > max)
		{
			max = a7[i];
			maxPi = i;
		}

}
void fw_INtoL1()
{
	for (int i = 0; i < 6; i++)
		for (int j = 0; j < 28; j++)
			for (int k = 0; k < 28; k++)
			{
				for (int m = 0; m < 5; m++)
					for (int n = 0; n < 5; n++)
					{
						z1[i][j][k] += input[j + m][k + n] * w1[i][m][n];
					}
				z1[i][j][k] += b1[i];
				if (z1[i][j][k] > 0)
					a1[i][j][k] = z1[i][j][k];
				else
					a1[i][j][k] = weaky * z1[i][j][k];
			}
}
void fw_L1toL2()
{
	for (int i = 0; i < 6; i++)
		for (int j = 0; j < 14; j++)
			for (int k = 0; k < 14; k++)
			{
				z2[i][j][k] = Max(a1[i][j * 2][k * 2], a1[i][j * 2][k * 2 + 1], a1[i][j * 2 + 1][k * 2], a1[i][j * 2 + 1][k * 2 + 1]);
			}
}
void fw_L2toL3()
{

	//0： 0 4 5 6 9 10 11 12 14 15
	func(0, 0); func(0, 4); func(0, 5); func(0, 6); func(0, 9);
	func(0, 10); func(0, 11); func(0, 12); func(0, 14); func(0, 15);
	//1：0 1 5 6 7 10 11 12 13 15
	func(1, 0); func(1, 1); func(1, 5); func(1, 6); func(1, 7); func(1, 10);
	func(1, 11); func(1, 12); func(1, 13); func(1, 15);
	//2：0 1 2 6 7 8 11 13 14 15
	func(2, 0); func(2, 1); func(2, 2); func(2, 6); func(2, 7);
	func(2, 8); func(2, 11); func(2, 13); func(2, 14); func(2, 15);
	//3：1 2 3 6 7 8 9 12 14 15
	func(3, 1); func(3, 2); func(3, 3); func(3, 6); func(3, 7);
	func(3, 8); func(3, 9); func(3, 12); func(3, 14); func(3, 15);
	//4：2 3 4 7 8 9 10 12 13 15
	func(4, 2); func(4, 3); func(4, 4); func(4, 7); func(4, 8);
	func(4, 9); func(4, 10); func(4, 12); func(4, 13); func(4, 15);
	//5：3 4 5 8 9 10 11 13 14 15
	func(5, 3); func(5, 4); func(5, 5); func(5, 8); func(5, 9);
	func(5, 10); func(5, 11); func(5, 13); func(5, 14); func(5, 15);

	for (int i = 0; i < 16; i++)
		for (int j = 0; j < 10; j++)
			for (int k = 0; k < 10; k++)
			{
				z3[i][j][k] += b3[i];

				if (z3[i][j][k] > 0)
					a3[i][j][k] = z3[i][j][k];
				else
					a3[i][j][k] = weaky * z3[i][j][k];
			}
}
void fw_L3toL4()
{
	for (int i = 0; i < 16; i++)
		for (int j = 0; j < 5; j++)
			for (int k = 0; k < 5; k++)
			{
				z4[i][j][k] = Max(a3[i][j * 2][k * 2], a3[i][j * 2][k * 2 + 1], a3[i][j * 2 + 1][k * 2], a3[i][j * 2 + 1][k * 2 + 1]);
			}
}
void fw_L4toL5()
{
	//展开z4，得到400个神经元
	int p = 0;
	for (int i = 0; i < 16; i++)
		for (int j = 0; j < 5; j++)
			for (int k = 0; k < 5; k++)
			{
				z4_dimReduc[p++] = z4[i][j][k];
			}
	//传播
	for (int i = 0; i < 120; i++)
	{
		for (int d = 0; d < 400; d++)
		{
			z5[i] += z4_dimReduc[d] * w5[i][d];
		}
		z5[i] += b5[i];
		if (z5[i] > 0)
			a5[i] = z5[i];
		else
			a5[i] = weaky * z5[i];
	}

}
void fw_L5toL6()
{

	for (int i = 0; i < 84; i++)
	{
		for (int d = 0; d < 120; d++)
			z6[i] += a5[i] * w6[i][d];

		z6[i] += b6[i];
	}
	db tMax = z6[0];
	for (int i = 0; i < 84; i++)
		if (a6[i] > tMax)
			tMax = a6[i];
	int Pow = log10(tMax);
	Pow = pow(10, Pow+1);
	for (int i = 0; i < 84; i++)
	{
		a6[i] = z6[i] / Pow;
		if (a6[i] > 0)
		{
			db ep = exp(-2 * a6[i]);
			a6[i] = 1.7159 * (1 - ep) / (1 + ep);
		}
		else
		{
			db ep = exp(2 * a6[i]);
			a6[i] = 1.7159 * (ep - 1) / (ep + 1);
		}
	}
}
void fw_L6toL7()
{
	for (int i = 0; i < 10; i++)
		for (int d = 0; d < 84; d++)
			z7[i] += (a6[d] - w7[i][d]) * (a6[d] - w7[i][d]) / 2;


	db SUM = 0;
	db tMax = 0;
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
}


void backward()
{
	bw_L7();
	bw_L6();
	bw_L5();
	bw_L4();
	bw_L3();
	bw_L2();
	bw_L1();

}
void bw_L7()
{

	for (int i = 0; i < 10; i++)
		pd7[i] =(a7[i]-output[i]) * (a7[i] - a7[i]* a7[i]);

	for (int i = 0; i < 10; i++)
		for (int j = 0; j < 84; j++)
			pd7_w[i][j] = -pd7[i] * (a6[j] - w7[i][j]);


	//查看中间结果
	if (showPD)
	{
		cout << "pd7:" << endl;
		for (int i = 0; i < 10; i++)
			cout << pd7[i] << endl;
		cout << "pd7_w:" << endl;
		for (int i = 0; i < 10; i++)
		{
			for (int j = 0; j < 84; j++)
				cout << pd7_w[i][j] << "  ";
			cout << endl;
		}


	}

}
void bw_L6()
{
	db _tPd6[84] = { 0 };
	for (int j = 0; j < 84; j++)
		for (int i = 0; i < 10; i++)
			_tPd6[j] += pd7[i] * (a6[j] - w7[i][j]);

	db tMax = z6[0];
	for (int i = 0; i < 84; i++)
		if (a6[i] > tMax)
			tMax = a6[i];
	int Pow = log10(tMax);
	Pow = pow(10, Pow+1);
	for (int j = 0; j < 84; j++)
	{
		int kjkj = -1;
		if (z6[j] > 0)
			kjkj = 1;
		pd6[j] = _tPd6[j] * (1.7159 - a6[j] * a6[j] / 1.7159) / Pow;

	}

	for (int i = 0; i < 84; i++)
	{
		for (int j = 0; j < 120; j++)
			pd6_w[i][j] = pd6[i] * a5[j];
		pd6_b[i] = pd6[i];
	}

	//查看中间结果
	if (showPD)
	{
		cout << "_tPd6:" << endl;
		for (int j = 0; j < 84; j++)
			cout << _tPd6[j] << endl;
		cout << "pd6:" << endl;
		for (int j = 0; j < 84; j++)
			cout << pd6[j] << endl;

		cout << "pd6_w:" << endl;
		for (int i = 0; i < 84; i++)
		{
			for (int j = 0; j < 120; j++)
				cout << pd6_w[i][j] << " ";
			cout << endl;
		}
	}
}
void bw_L5()
{
	db _tPd5[120] = { 0 };
	for (int j = 0; j < 120; j++)
		for (int i = 0; i < 84; i++)
			_tPd5[j] += pd6[i] * w6[i][j];


	for (int j = 0; j < 120; j++)
	{
		if (z5[j] > 0)
			pd5[j] = _tPd5[j];
		else
			pd5[j] = _tPd5[j] * weaky;
	}
	for (int i = 0; i < 120; i++)
	{
		for (int j = 0; j < 400; j++)
			pd5_w[i][j] = pd5[i] * z4_dimReduc[j];
		pd5_b[i] = pd5[i];
	}

	//查看中间结果
	if (showPD)
	{
		cout << "_tPd5:" << endl;
		for (int j = 0; j < 120; j++)
			cout << _tPd5[j] << endl;
		cout << "pd5:" << endl;
		for (int j = 0; j < 120; j++)
			cout << pd5[j] << endl;

		/*cout << "pd5_w:" << endl;
		for (int i = 0; i < 120; i++)
		{
			for (int j = 0; j < 400; j++)
				cout << setprecision(20) << pd5_w[i][j] << endl;
			cout << endl;
		}*/
	}
}
void bw_L4()
{
	db _tPd4[400] = { 0 };
	for (int j = 0; j < 400; j++)
		for (int i = 0; i < 120; i++)
			_tPd4[j] += pd5[i] * w5[i][j];

	int p = 0;
	for (int i = 0; i < 16; i++)
		for (int j = 0; j < 5; j++)
			for (int k = 0; k < 5; k++)
				pd4[i][j][k] = _tPd4[p++];
}
void bw_L3()
{
	db _tPd3[16][10][10] = { 0 };
	for (int i = 0; i < 16; i++)
	{
		for (int j = 0; j < 5; j++)
			for (int k = 0; k < 5; k++)
			{
				int p = maxP(a3[i][j * 2][k * 2], a3[i][j * 2][k * 2 + 1], a3[i][j * 2 + 1][k * 2], a3[i][j * 2 + 1][k * 2 + 1]);
				_tPd3[i][j * 2 + p / 2][k * 2 + p % 2] = pd4[i][j][k];
			}
	}
	for (int i = 0; i < 16; i++)
	{
		for (int j = 0; j < 10; j++)
			for (int k = 0; k < 10; k++)
			{
				if (z3[i][j][k] > 0)
					pd3[i][j][k] = _tPd3[i][j][k];
				else
					pd3[i][j][k] = weaky * _tPd3[i][j][k];
			}
	}
	for (int i = 0; i < 16; i++)
		for (int j = 0; j < 10; j++)
			for (int k = 0; k < 10; k++)
				pd3_b[i] += pd3[i][j][k];


	
	

}
void bw_L2()
{
	//0： 0 4 5 6 9 10 11 12 14 15
	func1(0, 0); func1(0, 4); func1(0, 5); func1(0, 6); func1(0, 9);
	func1(0, 10); func1(0, 11); func1(0, 12); func1(0, 14); func1(0, 15);
	//1：0 1 5 6 7 10 11 12 13 15
	func1(1, 0); func1(1, 1); func1(1, 5); func1(1, 6); func1(1, 7); func1(1, 10);
	func1(1, 11); func1(1, 12); func1(1, 13); func1(1, 15);
	//2：0 1 2 6 7 8 11 13 14 15
	func1(2, 0); func1(2, 1); func1(2, 2); func1(2, 6); func1(2, 7);
	func1(2, 8); func1(2, 11); func1(2, 13); func1(2, 14); func1(2, 15);
	//3：1 2 3 6 7 8 9 12 14 15
	func1(3, 1); func1(3, 2); func1(3, 3); func1(3, 6); func1(3, 7);
	func1(3, 8); func1(3, 9); func1(3, 12); func1(3, 14); func1(3, 15);
	//4：2 3 4 7 8 9 10 12 13 15
	func1(4, 2); func1(4, 3); func1(4, 4); func1(4, 7); func1(4, 8);
	func1(4, 9); func1(4, 10); func1(4, 12); func1(4, 13); func1(4, 15);
	//5：3 4 5 8 9 10 11 13 14 15
	func1(5, 3); func1(5, 4); func1(5, 5); func1(5, 8); func1(5, 9);
	func1(5, 10); func1(5, 11); func1(5, 13); func1(5, 14); func1(5, 15);
	

	if (showPD)
	{
		cout << endl << "pd3:" << endl;
		for (int i = 0; i < 16; i++)
		{
			for (int j = 0; j < 10; j++)
			{
				for (int k = 0; k < 10; k++)
					cout << pd3[i][j][k] << " ";
				cout << endl;
			}
			cout << endl;
		}
		cout << endl << "pd3_w:" << endl;
		for (int i = 0; i < 16; i++)
		{
			for (int j = 0; j < 5; j++)
			{
				for (int k = 0; k < 5; k++)
					cout << pd3_w[i][j][k] << " ";
				cout << endl;
			}
			cout << endl;
		}
	}
}
void bw_L1()
{
	db _tPd1[6][28][28] = { 0 };
	for (int i = 0; i < 6; i++)
		for (int j = 0; j < 14; j++)
			for (int k = 0; k < 14; k++)
			{
				int p = maxP(a1[i][j * 2][k * 2], a1[i][j * 2][k * 2 + 1], a1[i][j * 2 + 1][k * 2], a1[i][j * 2 + 1][k * 2 + 1]);
				_tPd1[i][j * 2 + p / 2][k * 2 + p % 2] = pd2[i][j][k];
			}

	for (int i = 0; i < 6; i++)
		for (int j = 0; j < 28; j++)
			for (int k = 0; k < 28; k++)
			{
				if (z1[i][j][k] > 0)
					pd1[i][j][k] = _tPd1[i][j][k];
				else
					pd1[i][j][k] = _tPd1[i][j][k] * weaky;
			}
	for (int i = 0; i < 6; i++)
		for (int j = 0; j < 28; j++)
			for (int k = 0; k < 28; k++)
			{
				for (int m = 0; m < 5; m++)
					for (int n = 0; n < 5; n++)
						pd1_w[i][m][n] += pd1[i][j][k] * input[j + m][k + n];

				pd1_b[i] += pd1[i][j][k];
			}
	if (showPD)
	{
		cout << endl << "pd1:" << endl;
		for (int i = 0; i < 6; i++)
		{
			for (int j = 0; j < 28; j++)
			{
				for (int k = 0; k < 28; k++)
					cout << pd1[i][j][k] << " ";
				cout << endl;
			}
			cout << endl;
		}
		cout << endl << "pd1_w:" << endl;
		for (int i = 0; i < 6; i++)
		{
			for (int j = 0; j < 5; j++)
			{
				for (int k = 0; k < 5; k++)
					cout << pd1_w[i][j][k] << " ";
				cout << endl;
			}
			cout << endl;
		}
	}
}

void update()
{
	//更新参数梯度
	for (int i = 0; i < 10; i++)
		for (int j = 0; j < 84; j++)
			w7[i][j] -= LearnProcess * pd7_w[i][j];

	for (int i = 0; i < 84; i++)
	{
		for (int j = 0; j < 120; j++)
			w6[i][j] -= LearnProcess * pd6_w[i][j];
		b6[i] -= LearnProcess * pd6_b[i];
	}
	for (int i = 0; i < 120; i++)
	{
		for (int j = 0; j < 400; j++)
			w5[i][j] -= LearnProcess * pd5_w[i][j];
		b5[i] -= LearnProcess * pd5_b[i];
	}
	for (int i = 0; i < 16; i++)
	{
		for (int m = 0; m < 5; m++)
			for (int n = 0; n < 5; n++)
				w3[i][m][n] -= LearnProcess * pd3_w[i][m][n];

		b3[i] -= LearnProcess * pd3_b[i];
	}
	for (int i = 0; i < 6; i++)
	{
		for (int j = 0; j < 5; j++)
			for (int k = 0; k < 5; k++)
			{
				w1[i][j][k] -= pd1_w[i][j][k];
			}
		b1[i] -= LearnProcess * pd1_b[i];
	}
}

//功能函数
void visual()
{
	//输入层
	cout << "输入层：" << endl;
	for (int i = 0; i < 32; i++)
	{
		for (int j = 0; j < 32; j++)
			cout << input[i][j] << " ";
		cout << endl;
	}cout << endl;
	cout << endl;

	//L1层
	cout << "L1层激活前:" << endl;
	for (int i = 0; i < 6; i++)
	{
		cout << "第" << i << "个：" << endl;
		for (int j = 0; j < 28; j++)
		{
			for (int k = 0; k < 28; k++)
				cout << z1[i][j][k] << " ";
			cout << endl;
		}
	}
	cout << "L1层激活后:" << endl;
	for (int i = 0; i < 6; i++)
	{
		cout << "第" << i << "个：" << endl;
		for (int j = 0; j < 28; j++)
		{
			for (int k = 0; k < 28; k++)
				cout << a1[i][j][k] << " ";
			cout << endl;
		}
	}
	cout << endl;

	//L2层
	cout << "L2层:" << endl;
	for (int i = 0; i < 6; i++)
	{
		cout << "第" << i << "个：" << endl;
		for (int j = 0; j < 14; j++)
		{
			for (int k = 0; k < 14; k++)
				cout << z2[i][j][k] << " ";
			cout << endl;
		}
	}
	cout << endl;

	//L3层
	cout << "L3层激活前:" << endl;
	for (int i = 0; i < 16; i++)
	{
		cout << "第" << i << "个：" << endl;
		for (int j = 0; j < 10; j++)
		{
			for (int k = 0; k < 10; k++)
				cout << z3[i][j][k] << " ";
			cout << endl;
		}
		cout << endl;
	}
	cout << "L3层激活后:" << endl;
	for (int i = 0; i < 16; i++)
	{
		cout << "第" << i << "个：" << endl;
		for (int j = 0; j < 10; j++)
		{
			for (int k = 0; k < 10; k++)
				cout << a3[i][j][k] << " ";
			cout << endl;
		}
		cout << endl;
	}
	cout << endl;

	//L4层
	cout << "L4层:" << endl;
	for (int i = 0; i < 16; i++)
	{
		cout << "第" << i << "个：" << endl;
		for (int j = 0; j < 5; j++)
		{
			for (int k = 0; k < 5; k++)
				cout << z4[i][j][k] << " ";
			cout << endl;
		}
		cout << endl;
	}
	cout << endl;

	//L5层
	cout << "L5层激活前：" << endl;
	for (int i = 0; i < 120; i++)
		cout << z5[i] << endl;
	cout << "L5层激活后：" << endl;
	for (int i = 0; i < 120; i++)
		cout << a5[i] << endl;
	cout << endl;

	//L6层
	cout << "L6层激活前：" << endl;
	for (int i = 0; i < 84; i++)
		cout << z6[i] << endl;
	cout << "L6层激活后：" << endl;
	for (int i = 0; i < 84; i++)
		cout << a6[i] << endl;
	cout << endl;

	//L7层
	cout << "L7层归一前：" << endl;
	for (int i = 0; i < 10; i++)
		cout << z7[i] << endl;

	cout << "最终结果：" << endl;
	for (int i = 0; i < 10; i++)
		cout << a7[i] << " ";
	cout << endl;
}


void setInput(int _input[28][28])
{

	for (int i = 0; i < 10; i++)
		output[i] = 0;

	for (int i = 0; i < 32; i++)
		for (int j = 0; j < 32; j++)
			input[i][j] = 0;
	for (int i = 0; i < 28; i++)
		for (int j = 0; j < 28; j++)
			input[i + 2][j + 2] = (db)_input[i][j] / 255;
}
void func(int inputN, int corN)
{
	for (int i = 0; i < 10; i++)
		for (int j = 0; j < 10; j++)
		{
			for (int m = 0; m < 5; m++)
				for (int n = 0; n < 5; n++)
				{
					z3[corN][i][j] += z2[inputN][i + m][j + n] * w3[corN][m][n];
				}

		}
}
void func1(int inputN, int corN)
{
	for(int i=0;i<14;i++)
		for(int j=0;j<14;j++)
			for(int m=0;m<5;m++)
				for (int n = 0; n < 5; n++)
				{
					if((i-m)>-1&&(j-n)>-1&&(i-m)<10&&(j-n)<10)
					  pd2[inputN][i][j] += pd3[corN][i - m][j - n] * w3[corN][m][n];
				}
	for (int m = 0; m < 5; m++)
for (int n = 0; n < 5; n++)
	for (int j = 0; j < 10; j++)
		for (int k = 0; k < 10; k++)
		
					pd3_w[corN][m][n] += pd3[corN][j][k] * z2[inputN][j + m][k + n];


}

db Max(db x1, db x2, db x3, db x4)
{
	db tMax = -999999;
	if (x1 > tMax)
		tMax = x1;
	if (x2 > tMax)
		tMax = x2;
	if (x3 > tMax)
		tMax = x3;
	if (x4 > tMax)
		tMax = tMax;
	return tMax;
}
int maxP(db x1, db x2, db x3, db x4)
{
	int p = 0;
	db tMax = -999999;
	if (x1 > tMax)
	{
		tMax = x1;
		p = 0;
	}
	if (x2 > tMax)
	{
		tMax = x2;
		p = 1;
	}
	if (x3 > tMax)
	{
		tMax = x3;
		p = 2;
	}
	if (x4 > tMax)
	{
		tMax = x4;
		p = 3;
	}
	return p;
}
