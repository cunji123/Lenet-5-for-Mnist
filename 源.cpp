#include"标头.h"
#include"导入.h"

typedef byte unsignedchar;
int bbb=0;
void chakan()
{
	string str;
	while (1)
	{
		cin >> str;
		if (bbb)
			bbb = 0;
		else
			bbb = 1;
	}

}

int main()
{
	srand((unsigned)time(NULL));
	thread t(chakan);
	cout << "weaky:" << endl;
	weaky = 0.2;
	//cin >> weaky;

	cout << "LearnProcess:" << endl;
	LearnProcess = 0.1;
	//cin >> LearnProcess;

	showPD =0;
	lenet();//初始化参数及其它
	int maxN = 0;
		for (int i = 0; i < 1000; i++)
		{
			readImages();//读取图像
			int arr[28][28];
			int label;
			int right1 = 0;
			int right2 = 0;
			for(int num=0;num<5999;num++)
			{
				//设置输入以及标签
				label = func(arr);//获取图像矩阵及标签
				setInput(arr);//设置输入
				output[label] = 1;//设置标签，其余为0


				forward();//向前传播
				if (bbb)
				{
					
					visual();
					showPD = 1;
					bbb = 0;
				}
				backward();//反向传播
				update();//更新参数

				if (num < 1000 && label == maxPi)
					right1++;
				if (num > 5000 && label == maxPi)
					right2++;
				showPD = 0;
			}
			if (right1 > maxN)
				maxN = right1;
			if (right2 > maxN)
				maxN = right2;
			cout << "第:"<<i<<"轮，正确率：" << (db)right1/1000 << ",   " << (db)right2/1000<< ",   ";
			cout << "最大优解:" << (db)maxN/1000 << endl;
		}



	return 0;
	
	
}
/*
	//Mat mat(28,28, CV_8UC1);
	//Lenet.visual();
	//imshow("1", mat);
	//waitKey(0);
namedWindow("Example2", WINDOW_AUTOSIZE);
	VideoCapture cap;
	cap.open("D:\\12.mp4");
	Mat frame;
	while (1)
	{
		cap >> frame;
		if (frame.empty()) break;
		imshow("Exameple2", frame);
		if (waitKey(400) >= 0) break;
	}

	return 0;*/