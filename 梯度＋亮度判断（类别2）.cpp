#include<opencv2\opencv.hpp>
#include<iostream>

using namespace std;
using namespace cv;

//int类型转换为字符串
string intTostring(int i) {
	stringstream ss;
	string out;
	ss << i;
	out = ss.str();
	return out;
}

double calDistance(Point p1, Point p2, Point p) {
	double a, b, c, dis;
	// 化简两点式为一般式
	// 两点式公式为(y - y1)/(x - x1) = (y2 - y1)/ (x2 - x1)
	// 化简为一般式为(y2 - y1)x + (x1 - x2)y + (x2y1 - x1y2) = 0
	// A = y2 - y1
	// B = x1 - x2
	// C = x2y1 - x1y2
	a = p1.y - p2.y;
	b = p2.x - p1.x;
	c = p1.x*p2.y - p2.x*p1.y;
	// 距离公式为d = |A*x0 + B*y0 + C|/√(A^2 + B^2)
	if (a != 0 && b != 0)
		dis = abs(a*p.x + b*p.y + c) / sqrt(a*a + b*b);
	else if (a == 0)
	{
		dis = abs(p.y - p1.y);
	}
	else
	{
		dis = abs(p.x - p1.x);
	}
	return dis;
	//cout << "距离：" << dis << endl;
}

//透视变换
void perspectiveTransform(Mat srcImage, Mat&perspectiveImage, 
	static vector<Point2f>&tmplatepoints, static int heigh, static int width);//输入图像，变换后的输出图像，模板的四个点坐标

//旋转变换
void rotateRow(Mat& img, Mat &img_rotate, double degree);//输入图像，输出图像，旋转角度

//自动获取非信号灯区域模板上下行
void tmplateUpAndDown(Mat roImage, int &upRow, int &downRow);//输入图像，模板首行，模板末行

//模板匹配
void tmplateMatch(Mat src, Mat &roImage, vector<int>&rowNonSignal, int upRow, int downRow);//输入图像,匹配结果图像，每个非信号灯区域首行位置，模板首行，模板末行

//亮暗判断
void judgelightORblack(Mat resultRoiImage, Mat&dst, vector<int>signalRow);//输入图像，输出图像，信号灯区域上下行

void sobel_segement(Mat roImage);
//线性拟合
bool polynomial_curve_fit(std::vector<cv::Point>& key_point, int n, cv::Mat& A);
int main() 
{
	
	string resultpath = "result2\\";
	for (int imagenameNum = 0; imagenameNum < 31; imagenameNum++) {
		cout << "image: " << imagenameNum << endl;
		string filepath = "四点图片//";
		string referImage = "2(" + intTostring(imagenameNum)+").jpg";
		Mat srcImage = imread(filepath + referImage, 1);
		if (srcImage.empty()) {
			cout << "could not find the image..." << endl;
			return 0;
		}
		imshow("srcImage", srcImage);
		imwrite(resultpath + referImage + "1.jpg", srcImage);
		//将横着排列的信号灯统一变换为竖着排列
		if (srcImage.cols > srcImage.rows) {
			rotateRow(srcImage, srcImage, 270);
		}
		//透视变换
		Mat perspectiveImage;
		static vector<Point2f>tmplatepoints;//变换模板的四个点位置
		static int heigh, width;
		if (imagenameNum == 0) {
			heigh = srcImage.rows;
			width = srcImage.cols;
		}
		perspectiveTransform(srcImage, perspectiveImage, tmplatepoints,heigh,width);
		cout << "透视变换完成" << endl;
		if (!perspectiveImage.empty())//判断是否正确进行变换
		{
			imwrite(resultpath + referImage + "2.jpg", perspectiveImage);
			
			int roiwidth = abs(tmplatepoints[0].x - tmplatepoints[1].x) > abs(tmplatepoints[2].x - tmplatepoints[3].x) ?
				abs(tmplatepoints[0].x - tmplatepoints[1].x) : abs(tmplatepoints[2].x - tmplatepoints[3].x);
			int roiheigh = abs(tmplatepoints[0].y - tmplatepoints[3].y) > abs(tmplatepoints[2].y - tmplatepoints[1].y) ?
				abs(tmplatepoints[0].y - tmplatepoints[3].y) : abs(tmplatepoints[2].y - tmplatepoints[1].y);
			double widratio = 1.2;//提取宽度比
			int labelheigh = 0;//标签高度
			char referfirst = referImage[0];
			if (referfirst == '1') {
				widratio = 1.2;
				labelheigh = 20;
			}
			if (referfirst == '2') {
				widratio = 1.4;
				
			}
			if (referfirst == '3')widratio = 1.6;
			if ((tmplatepoints[0].x - (widratio - 1) / 2 *roiwidth > 0) && (tmplatepoints[0].x + widratio*roiwidth < perspectiveImage.cols))
			{
				//Mat resultRoiImagetmp = perspectiveImage(Rect(tmplatepoints[0].x , tmplatepoints[0].y + labelheigh / 2,
				//	roiwidth, roiheigh - labelheigh));//精确定位掩膜区域;
				
				Mat roImage = perspectiveImage(Rect(tmplatepoints[0].x - (widratio - 1) / 2 * roiwidth, tmplatepoints[0].y + labelheigh / 2,
					widratio*roiwidth, roiheigh - labelheigh));//精确定位掩膜区域

				Mat resultRoiImage = roImage.clone();
				//Mat resultRoiImage = tmpRoiImage(Rect(0.2*tmpRoiImage.cols, 0, 0.6*tmpRoiImage.cols,tmpRoiImage.rows));
				imshow("roImage", roImage);
				imwrite(resultpath + referImage + "3.jpg", roImage);
				
				/*Mat segeroImage = roImage.clone();
				sobel_segement(segeroImage);*/

				//模板匹配分割
				int upRow, downRow;//模板的上下行位置
				tmplateUpAndDown(roImage, upRow, downRow);
				//cout << "down:" << downRow << " up:" << upRow << endl;
				if (upRow > 0 && downRow < roImage.rows&&upRow < downRow) {
					Mat dstImage = roImage.clone();
					vector<int>rowNonSignal;//非信号灯区域的首行位置
					tmplateMatch(roImage, dstImage, rowNonSignal, upRow, downRow);//模板匹配
					imshow("dstImage", dstImage);
					imwrite(resultpath + referImage + "4.jpg", dstImage);
					vector<int>signalRow;//信号灯的上下行依次排列
					if (rowNonSignal.size() > 0) {
						//画出信号灯区域
						signalRow.push_back(1);
						signalRow.push_back(rowNonSignal[0]);
						rectangle(roImage, Rect(1, 1, roImage.cols - 2, rowNonSignal[0]), Scalar(0, 0, 255), 2, 8);
						for (int i = 0; i < rowNonSignal.size() - 1; i++) {
							rectangle(roImage, Rect(1, rowNonSignal[i] + downRow - upRow, roImage.cols - 2, 
								rowNonSignal[i + 1] - (rowNonSignal[i] + downRow - upRow)),Scalar(0, 0, 255), 2, 8);
							signalRow.push_back(rowNonSignal[i] + downRow - upRow);
							signalRow.push_back(rowNonSignal[i + 1]);
							
						}
						rectangle(roImage, Rect(1, rowNonSignal[rowNonSignal.size() - 1] + downRow - upRow, roImage.cols - 2,
							roImage.rows - (rowNonSignal[rowNonSignal.size() - 1] + downRow - upRow)), Scalar(0, 0, 255), 2, 8);

						signalRow.push_back(rowNonSignal[rowNonSignal.size()-1] + downRow - upRow);
						signalRow.push_back(roImage.rows - 1);
						imshow("result", roImage);
						imwrite(resultpath + referImage + "5.jpg", roImage);

						//亮暗判断
						if (!signalRow.empty()) 
						{
							Mat dst;
							judgelightORblack(resultRoiImage, dst, signalRow);
							imshow("dst", dst);
							imwrite(resultpath + referImage + "6.jpg", dst);
						}
					}
				}
			}
		}
		cout << endl;
		if(waitKey(0)==27)destroyAllWindows();
	}
	return 0;
}

//透视变换
void perspectiveTransform(Mat srcImage, Mat&perspectiveImage, static vector<Point2f>&tmplatepoints,static int heigh,static int width) {

	Mat grayImage;
	cvtColor(srcImage, grayImage, CV_BGR2GRAY);
	//equalizeHist(grayImage, grayImage);
	Mat gaussImage;
	blur(grayImage, gaussImage, Size(3, 3));
	imshow("median", gaussImage);

	Mat binaryImage;
	threshold(gaussImage, binaryImage, 5, 255, THRESH_BINARY);
	imshow("binary", binaryImage);

	//闭操作去除黑洞
	Mat morphImage;
	int kernelSizeValue = 1;
	Mat element = getStructuringElement(MORPH_RECT, Size(kernelSizeValue * 2 + 1,
		kernelSizeValue * 2 + 1));
	morphologyEx(binaryImage, morphImage, MORPH_CLOSE, element);
	
	//imwrite("result\\" + referImage + ".jpg", morphImage);

	//反二值化
	threshold(binaryImage, morphImage, 0, 255, THRESH_BINARY_INV);
	imshow("morphImage", morphImage);
	Mat edgeImage = Mat::zeros(morphImage.size(), morphImage.type());
	/*Canny(morphImage, edgeImage, 50, 100, 3);
	imshow("edgeImage", edgeImage);*/

	//寻找所有轮廓
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours(morphImage, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);
	cout <<"轮廓数量："<< contours.size() << endl;

	//排除掉狭长轮廓
	int tmpsize = contours.size();
	for (int i = 0; i < tmpsize; ) {
		double area = contourArea(contours[i]);
		RotatedRect rect = minAreaRect(contours[i]);
		Point2f P[4];//最小外接矩形四个端点
		rect.points(P);
		//计算矩形长宽比
		double rect_ratio = sqrtf((powf((P[1].x) - (P[0].x), 2) + powf((P[1].y) - (P[0].y), 2)) /
			(powf((P[2].x) - (P[1].x), 2) + powf((P[2].y) - (P[1].y), 2)));
		if (rect_ratio > 3 || rect_ratio < 0.33||contours[i].size()<5) {
			swap(contours[i], contours[tmpsize - 1]);
			contours.pop_back();
			tmpsize--;

		}
		else
		{
			i++;
		}

	}
	//获取标注圆心位置
	vector<Point2f>centerpoints;
	
	for (int i = 0; i < tmpsize; ) {
		drawContours(edgeImage, contours, i, 255, 1, 8);

		RotatedRect temp = fitEllipse(contours[i]);
		centerpoints.push_back(temp.center);
		i++;
		//cout << i << ": ( " << centerpoints[i].x << ", " << centerpoints[i].y << " )" << endl;
	}
	imshow("edgeImage", edgeImage);
	cout << "点的数量：" << centerpoints.size() << endl;
	if (centerpoints.size() == 4) {
		//对标注的四个点进行排序
		for (int i = 0; i < centerpoints.size() - 1; i++) {
			for (int j = i + 1; j < centerpoints.size(); j++) {
				if (centerpoints[i].y > centerpoints[j].y) {
					swap(centerpoints[i], centerpoints[j]);
				}
			}

		}
		if (centerpoints[0].x > centerpoints[1].x)swap(centerpoints[0], centerpoints[1]);
		if (centerpoints[2].x < centerpoints[3].x)swap(centerpoints[2], centerpoints[3]);
		//模板点位置
		if (tmplatepoints.empty()) { 
			tmplatepoints = centerpoints; }

	/*	for (int i = 0; i < centerpoints.size(); i++) {
			cout << i << ": ( " << centerpoints[i].x << ", " << centerpoints[i].y << " )" << endl;

		}*/

	}
	
	
	
	if (tmplatepoints.size() == 4 && centerpoints.size() == 4) {
		
		//获取透视变换矩阵
		Mat transform = getPerspectiveTransform(tmplatepoints, centerpoints);
		//透视变换
		cv::warpPerspective(srcImage, perspectiveImage, transform, Size(width,heigh),
			cv::INTER_LINEAR | cv::WARP_INVERSE_MAP);
		imshow("perspectiveImage", perspectiveImage);
	}
	/*imwrite("result\\" + referImage + "1.jpg", morphImage);
	imwrite("result\\" + referImage + "2.jpg", perspectiveImage);*/
}

//旋转变换
void rotateRow(Mat& img, Mat &img_rotate, double degree)
{
	//旋转中心为图像中心
	CvPoint2D32f center;
	center.x = float(img.cols / 2.0 + 0.5);
	center.y = float(img.rows / 2.0 + 0.5);
	//计算二维旋转的仿射变换矩阵
	float m[6];
	Mat M = Mat(2, 3, CV_32F, m);
	M = getRotationMatrix2D(center, degree, 1.0);
	double scale = 1.0;
	//变换图像，并用黑色填充其余值
	Rect bbox;
	bbox = RotatedRect(center, Size(scale*img.cols, scale*img.rows), degree).boundingRect();

	// 对变换矩阵的最后一列做修改，重新定义变换的 中心

	M.at<double>(0, 2) += bbox.width / 2 - center.x;
	M.at<double>(1, 2) += bbox.height / 2 - center.y;

	warpAffine(img, img_rotate, M, bbox.size());
}

//自动获取非信号灯区域模板上下行
void tmplateUpAndDown(Mat roImage, int &upRow, int &downRow) {
	//获取模板位置
	Mat sobel_y, medianImage, grayroImage;
	cvtColor(roImage, grayroImage, CV_BGR2GRAY);
	imshow("gray", grayroImage);
	//equalizeHist(grayroImage, grayroImage);
	medianBlur(grayroImage, medianImage, 3);
	imshow("upandroweuqal",medianImage);
	
	Sobel(medianImage, sobel_y, -1, 1, 0, 3);
	imshow("sobel_y", sobel_y);
	//Mat cannyImage;
	//Canny(sobel_y, cannyImage, 80, 160);
	//imshow("cannyImage", cannyImage);

	//计算每行的平均梯度值
	vector<double>colSums(sobel_y.rows);
	for (int i = 0; i < sobel_y.rows; i++) {
		double coltmp = 0.0;
		for (int j = 0; j < sobel_y.cols; j++) {
			coltmp += sobel_y.at<uchar>(i, j);
		}
		colSums[i] = coltmp / sobel_y.cols;
	}
	vector<double>mincolSums(colSums);//最小平均梯度值
	sort(mincolSums.begin(), mincolSums.end());
	Mat blackRow = Mat::zeros(Size(colSums.size(), 100), sobel_y.type());
	vector<int>rowNum;
	for (int i = 0; i < colSums.size() - 1; i++) {
		Point pre, cur;
		pre.x = i;
		pre.y = (1 - (colSums[i]- mincolSums[0]) / (mincolSums[mincolSums.size() - 1] - mincolSums[0])) * 100;
		cur.x = i + 1;
		cur.y = (1 - (colSums[i + 1] - mincolSums[0]) / (mincolSums[mincolSums.size() - 1] - mincolSums[0])) * 100;
		line(blackRow, pre, cur, Scalar(255), 1, 8);//画出梯度图
		//取出小的平均梯度值的行数
		if (colSums[i] < mincolSums[0] + 4) {
			//cout << i << endl;
			rowNum.push_back(i);
		}
	}
	imshow("梯度均值", blackRow);
	//imwrite("result\\" + referImage + "6.jpg", blackRow);
	struct tmplaterow
	{
		int uprow;
		int length;
	};
	vector<tmplaterow>tmplateupAnddown;
	if (rowNum.size() > 0) {
		
		tmplaterow tmpRow;
		tmpRow.length = 0;
		tmpRow.uprow = rowNum[0];
		//计算连续行数的长度以及
		for (int i = 0; i < rowNum.size() - 1; i++) {
			if (rowNum[i + 1] - rowNum[i] == 1) {
				tmpRow.length++;

			}
			else
			{
				tmplateupAnddown.push_back(tmpRow);
				//记录对应的第一行
				tmpRow.uprow = rowNum[i + 1];
				tmpRow.length = 0;
			}
		}
		//取出行数最大的作为模板
		for (int i = 0; i < tmplateupAnddown.size(); i++) {
			if (tmplateupAnddown[0].length < tmplateupAnddown[i].length)swap(tmplateupAnddown[0], tmplateupAnddown[i]);
		}
		/*for (int i = 1; i < tmplateupAnddown.size(); i++) {
			if (tmplateupAnddown[1].length < tmplateupAnddown[i].length)swap(tmplateupAnddown[1], tmplateupAnddown[i]);
		}*/
	}
	if (tmplateupAnddown.size() > 0) {
		//模板的上下行
		upRow = tmplateupAnddown[0].uprow;
		
		downRow = upRow + tmplateupAnddown[0].length;
	}

}

//模板匹配
void tmplateMatch(Mat src, Mat &roImage, vector<int>&rowNonSignal, int upRow, int downRow) {
	Mat grayroImage;
	cvtColor(src, grayroImage, CV_BGR2GRAY);
	//equalizeHist(grayroImage, grayroImage);
	blur(grayroImage, grayroImage, Size(3,3));
	imshow("equaltmplate", grayroImage);
	if (downRow<grayroImage.rows&&upRow>0&&downRow>upRow)
	{
		//画出模板区域
		Point v1, v2, v3, v4;
		v1.x = v2.x = 0;
		v1.y = v4.y = upRow;
		v3.x = v4.x = grayroImage.cols;
		v2.y = v3.y = downRow;
		line(roImage, v1, v4, Scalar(0, 255, 0), 2, 8, 0);
		line(roImage, v2, v3, Scalar(0, 255, 0), 2, 8, 0);
		imshow("line", roImage);
		//imwrite("result\\6.jpg"  , roImage);

		Mat tmplateImage = grayroImage(Rect(v1, v3));//模板图片
		imshow("tmplateImage", tmplateImage);

		vector<double>tmplateScore(grayroImage.rows - tmplateImage.rows);//定义图像中每行模板匹配得分
																		 //模板匹配得分计算
		for (int i = 0; i < grayroImage.rows - tmplateImage.rows; i++) {
			double equal = 0;
			double tmpl = 0;
			double diss = 0;
			for (int k = 0; k < tmplateImage.rows; k++) {
				uchar* srcPtr = grayroImage.ptr<uchar>(i + k);
				uchar*tmpPtr = tmplateImage.ptr<uchar>(k);
				for (int j = 0; j < tmplateImage.cols; j++) {
					equal += srcPtr[j] * srcPtr[j];
					tmpl += tmpPtr[j] * tmpPtr[j];
					diss += (srcPtr[j] - tmpPtr[j])*(srcPtr[j] - tmpPtr[j]);
				}
			}
			//tmplateScore[i] = diss / (tmplateImage.rows*tmplateImage.cols*255*255);
			tmplateScore[i] = diss / sqrt(equal*tmpl);//模板匹配得分（得分越高匹配程度越低）

		}
		//画出匹配得分情况
		Mat image_tmp = Mat::zeros(Size(tmplateScore.size() + 50, 150), CV_8UC1);
		for (int i = 0; i < tmplateScore.size() - 1; i++) {
			Point x1, x2;
			x1.x = i + 10;
			x2.x = i + 11;
			x1.y = 150 - (1.0 - tmplateScore[i]) * 100;
			x2.y = 150 - (1.0 - tmplateScore[i + 1]) * 100;
			line(image_tmp, x1, x2, 255, 1, 8);
			//cout << tmplateScore[i] << endl;
		}
		// 获取匹配最大得分
		vector<double>scoreMax(tmplateScore);
		sort(scoreMax.begin(), scoreMax.end());
		double score_max = 1.0 - scoreMax[0];
		/*Point tmp_d1,tmp_d2;
		tmp_d1.y = tmp_d2.y=150- score_max*85;
		tmp_d1.x = 0;
		tmp_d2.x = tmplateScore.size() + 10;
		line(image_tmp, tmp_d1, tmp_d2, 255, 1, 8);*/

		imshow("imag_score", image_tmp);
		//imwrite("result\\" + referImage + "5.jpg", image_tmp);
		//获取特定区域内得分大于一定值的峰值
		//vector<int>rowNonSignal;
		
		int borden = 15;//匹配上下边界
		//if (src.rows > 300)borden = 20;
		if (tmplateScore.size() > 2 * borden) {
			
			for (int i = 0; i < tmplateScore.size(); i++) {
				int judge = 0;
				
				//前面边界
				//if (i < borden)
				//{
				//	for (int j = 1; j < borden; j++) {

				//		if (((1.0 - tmplateScore[i]) > score_max / 3.0) && ((1.0 - tmplateScore[i]) > (1.0 - tmplateScore[i + j]))) {
				//			//bordenNum++;
				//			judge = 1;
				//			
				//		}
				//		else
				//		{
				//			judge = 0;
				//			break;

				//		}
				//	}
				//	if (judge == 1) {
				//		for (int j = 1; j < i +1; j++) {

				//			if (((1.0 - tmplateScore[i]) < score_max / 3.0) || ((1.0 - tmplateScore[i]) <= (1.0 - tmplateScore[i - j]))) {
				//				//bordenNum++;
				//				judge = 0;
				//				break;
				//			}
				//			else
				//			{
				//				judge = 1;

				//			}
				//		}
				//		
				//	}
				//	
				//	

				//}
				//中间
				if (i >= borden&&i <= tmplateScore.size() - borden)
				{

					for (int j = 1; j < borden; j++) {

						if (((1.0 - tmplateScore[i]) < score_max / 3.0) || ((1.0 - tmplateScore[i]) <= (1.0 - tmplateScore[i + j])) ||
							((1.0 - tmplateScore[i]) <= (1.0 - tmplateScore[i - j]))) {
							//bordenNum++;
							judge = 0;
							break;
						}
						else
						{
							judge = 1;

						}
					}
					//if (((1.0 - tmplateScore[i]) > score_max *0.85))judge = 1;
					//if (judge == 1)rowNonSignal.push_back(i);
				}
			
				//后边界
				//if (i > tmplateScore.size() - borden) {
				//	//int judge = 0;
				//	//int tmp = i;
				//	//int bordenNum = 0;
				//	for (int j = 1; j < borden; j++) {

				//		if (((1.0 - tmplateScore[i]) < score_max / 3.0) || ((1.0 - tmplateScore[i]) <= (1.0 - tmplateScore[i - j]))) {
				//			//bordenNum++;
				//			judge = 0;
				//			break;
				//		}
				//		else
				//		{
				//			judge = 1;

				//		}
				//	}
				//	if (judge == 1) {
				//		for (int j = 1; j < tmplateScore.size() - i; j++) {

				//			if (((1.0 - tmplateScore[i]) < score_max / 3.0) || ((1.0 - tmplateScore[i]) <= (1.0 - tmplateScore[i + j]))) {
				//				//bordenNum++;
				//				judge = 0;
				//				break;
				//			}
				//			else
				//			{
				//				judge = 1;

				//			}
				//		}
				//	}
				//}
				//if (((1.0 - tmplateScore[i]) > score_max *0.85))judge = 1;
				if (judge == 1)rowNonSignal.push_back(i);
			}
		}
		//if (rowNonSignal.size() > 1) 
		//{
		//	cout << 1-tmplateScore[rowNonSignal[rowNonSignal.size() - 1]] <<"  "<< 1-tmplateScore[rowNonSignal[ rowNonSignal.size() - 2]] << endl;
		//	if ((1.0-tmplateScore[rowNonSignal[rowNonSignal.size() - 1]]) < (1.0-tmplateScore[rowNonSignal[rowNonSignal.size() - 2]]) * 0.95){
		//		//swap(rowNonSignal[0], rowNonSignal[rowNonSignal.size() - 1]);
		//		rowNonSignal.pop_back();

		//	}
		//	cout << 1-tmplateScore[rowNonSignal[0]] << "  "<<1-tmplateScore[rowNonSignal[1]] << endl;
		//	if ((1.0-tmplateScore[rowNonSignal[0]]) < (1-tmplateScore[rowNonSignal[1]]) * 0.95) {
		//		
		//		swap(rowNonSignal[0], rowNonSignal[rowNonSignal.size() - 1]);
		//		rowNonSignal.pop_back();
		//		for (int i = 0; i < rowNonSignal.size() - 1; i++) {
		//			int tmp = rowNonSignal[i ];
		//			rowNonSignal[i ] = rowNonSignal[i+1];
		//			rowNonSignal[i+1] = tmp;
		//		}
		//	}
		//}
		//struct tmplaterow
		//{
		//	int uprow;
		//	int length;
		//};
		//vector<tmplaterow>tmplateupAnddown;
		//if (rowNonSignal.size() > 0) {

		//	tmplaterow tmpRow;
		//	tmpRow.length = 0;
		//	tmpRow.uprow = rowNonSignal[0];
		//	//计算连续行数的长度以及
		//	for (int i = 0; i < rowNonSignal.size() - 1; i++) {
		//		if (rowNonSignal[i + 1] - rowNonSignal[i] == 1) {
		//			tmpRow.length++;

		//		}
		//		else
		//		{
		//			tmplateupAnddown.push_back(tmpRow);
		//			//记录对应的第一行
		//			tmpRow.uprow = rowNonSignal[i + 1];
		//			tmpRow.length = 0;
		//		}
		//	}
		//	
		//}
		//在图中画出峰值位置（行数）
		for (int i = 0; i < rowNonSignal.size(); i++) {
			//cout << "row: " << rowNonSignal[i] << "  " << tmplateScore[rowNonSignal[i]] << endl;
			/*Point nonP;
			nonP.x = grayroImage.cols / 2;
			nonP.y = rowNonSignal[i] + (downRow - upRow) / 2;
			circle(resultimage, nonP, 4, Scalar(255, 0, 255), 2, 8);*/
			rectangle(roImage, Rect(0, rowNonSignal[i], roImage.cols, downRow-upRow), Scalar(255, 0, 255), -1, 8);

		}
		/*if (rowNonSignal.size() > 1) {
			rectangle(grayroImage, Rect(0, 0, roImage.cols, 2 * rowNonSignal[0] - rowNonSignal[1] + downRow - upRow), Scalar(255, 0, 255), -1, 8);
			int rectHeigh = 2 * rowNonSignal[rowNonSignal.size() - 1] - rowNonSignal[rowNonSignal.size() - 2];
			rectangle(roImage, Rect(0, rectHeigh, roImage.cols, roImage.rows - rectHeigh),
				Scalar(255, 0, 255), -1, 8);
		}*/
		//imshow("center", roImage);
		//imwrite("result\\" + referImage + "6.jpg", roImage);
	}
}

//信号灯亮暗判断
void judgelightORblack(Mat resultRoiImage, Mat&dst, vector<int>signalRow) {
	//Mat resultImage = resultRoiImage(Rect(Point(0.2*resultRoiImage.cols, 0), Point(0.8*resultRoiImage.cols, resultRoiImage.rows)));
	dst = resultRoiImage.clone();
	/*for (int i = 0; i < signalRow.size(); i+=2) {
		rectangle(dst, Rect(Point(0, signalRow[i]), Point(dst.cols, signalRow[i + 1])), Scalar(0, 0, 255), 2, 8);
	}*/
	Mat grayRoiImage;
	cvtColor(resultRoiImage, grayRoiImage, CV_BGR2GRAY);
	//equalizeHist(grayRoiImage, grayRoiImage);
	medianBlur(grayRoiImage, grayRoiImage, 3);
	imshow("grayRoiImage", grayRoiImage);

	////直方图
	//const int channels[1] = { 0 };
	//const int histSize[1] = { 256 };
	//float hranges[2] = { 0,255 };
	//const float* ranges[1] = { hranges };
	//MatND hist;
	//calcHist(&grayRoiImage, 1, channels, Mat(), hist, 1, histSize, ranges);
	//double maxVal = 0;
	//double minVal = 0;
	////找到直方图中的最大值和最小值
	//minMaxLoc(hist, &minVal, &maxVal, 0, 0);
	//int histSize1 = hist.rows;
	//Mat histImg(histSize1, histSize1, CV_8U, Scalar(255));
	//// 设置最大峰值为图像高度的90%
	//int hpt = static_cast<int>(0.9*histSize1);
	//for (int h = 0; h<histSize1; h++)
	//{
	//	float binVal = hist.at<float>(h);
	//	int intensity = static_cast<int>(binVal*hpt / maxVal);
	//	line(histImg, Point(h, histSize1), Point(h, histSize1 - intensity), Scalar::all(0));
	//}
	//imshow("hisImg", histImg);//显示直方图
	Mat sobel_y;
	Sobel(grayRoiImage, sobel_y, -1, 1, 0, 3);
	imshow("sobel_y", sobel_y);
	//计算每行的平均梯度值
	vector<double>colSums(sobel_y.rows);
	for (int i = 0; i < sobel_y.rows; i++) {
		double coltmp = 0.0;
		for (int j = 0; j < sobel_y.cols; j++) {
			coltmp += sobel_y.at<uchar>(i, j);
		}
		colSums[i] = coltmp / sobel_y.cols;
	}

	vector<double>sortcolSums(colSums);//最小平均梯度值
	sort(sortcolSums.begin(), sortcolSums.end());
	double maxcoSums = sortcolSums[sortcolSums.size() - 1];//最大平均梯度值
	double mincoSums = sortcolSums[0];
	//cout << "minGrad:" << mincoSums << " maxGrad:" << maxcoSums << endl;
	Mat blackRow = Mat::zeros(Size(colSums.size(), 255), CV_32FC3);
	//vector<int>rowNum;
	Scalar gradColor = Scalar(0, 100, 255);//梯度显示颜色
	Scalar valueColor = Scalar(255, 100, 0);//亮度显示颜色
	for (int i = 0; i < colSums.size() - 1; i++) {
		Point pre, cur;
		pre.x = i;
		pre.y = 255-colSums[i]  ;
		cur.x = i + 1;
		cur.y = 255-colSums[i + 1] ;
		line(blackRow, pre, cur, gradColor, 1, 8);//画出梯度图

		//取出小的平均梯度值的行数
		//if (colSums[i] < mincolSums[0] + 4) {
		//	//cout << i << endl;
		//	rowNum.push_back(i);
		//}
	}
	
	/*----------------根据梯度值较高区域进行亮暗判断------------------*/
	sort(signalRow.begin(), signalRow.end());
	vector<int>rowSobelTop;//梯度峰值
	//寻找梯度值峰值
	for (int i = 0; i < signalRow.size() - 1; i += 2) {
		for (int j = 0; j < grayRoiImage.rows; j++) {
			int num = 16;//信号灯个数
			int borden = colSums.size()/(num*2);
			int judge = 0;
			
			//前段
			if (j < borden)
			{
				for (int k = 1; k < borden; k++) {
					if (colSums[j] < colSums[j + k] || colSums[j] < maxcoSums / 3.0) {
						judge = 0;
						break;

					}
					else
					{
						judge = 1;

					}

				}
				if (judge == 1) {
					for (int k = 1; k < j; k++) {
						if (colSums[j] <colSums[j - k] || colSums[j] < maxcoSums / 3.0) {
							judge = 0;
							break;

						}
						else
						{
							judge = 1;

						}
					}
				}
			}
			//中间段
			if ((j>=borden)&&(j<=grayRoiImage.rows - borden))
			{
				for (int k = 1; k < borden; k++) {
					if (colSums[j] < colSums[j - k] || colSums[j] < (maxcoSums / 3.0)|| colSums[j] < colSums[j + k]) 
{
						judge = 0;
						break;

					}
					else
					{
						judge = 1;

					}
					
				}
			}
			//后段
			if ( j >grayRoiImage.rows - borden)
			{
				for (int k = 1; k < grayRoiImage.rows - j; k++) {
					if (colSums[j] < colSums[j + k] || colSums[j] < maxcoSums / 3.0) {
						judge = 0;
						break;

					}
					else
					{
						judge = 1;

					}
				}

				
				if (judge == 1) {
					for (int k = 1; k < borden; k++) {
						if (colSums[j] < colSums[j - k] || colSums[j] < maxcoSums / 3.0) {
							judge = 0;
							break;

						}
						else
						{
							judge = 1;
							
						}
					}
				}
			}
			
			if (judge == 1) {
				rowSobelTop.push_back(j);
				//cout << j << endl;
				line(blackRow, Point(j, 0), Point(j, 255), Scalar(255, 255, 255), 1, 8);
			}
		}
	}
	sort(rowSobelTop.begin(), rowSobelTop.end());

	//计算每行亮度均值
	vector<double>rowValue(grayRoiImage.rows);
	for (int i = 0; i < grayRoiImage.rows; i++) {
		for (int j = 0; j < grayRoiImage.cols; j++) {
			rowValue[i] += grayRoiImage.at<uchar>(i, j);
		}
		rowValue[i] /= grayRoiImage.cols;
	}
	vector<double>rowValuecopy(rowValue);
	sort(rowValuecopy.begin(), rowValuecopy.end());

	double rowValueMax = rowValuecopy[rowValuecopy.size() - 1];//最大亮度均值
	double rowValueMin = rowValuecopy[0];
	//cout << "minValue:" << rowValueMin << " maxValue:" << rowValueMax << endl;
	//画出每行亮度情况
	Mat image_tmp = Mat::zeros(Size(rowValue.size(), 255), CV_8UC1);
	vector<Point>rowValue_points;
	for (int i = 0; i < rowValue.size() - 1; i++) {
		Point x1, x2;
		x1.x = i;
		x2.x = i;
		x1.y = 255 - rowValue[i];
		x2.y = 255 - rowValue[i + 1];
		line(image_tmp, x1, x2, 255, 1, 8);
		line(blackRow, x1, x2, valueColor,1,8);
		//cout << tmplateScore[i] << endl;
		rowValue_points.push_back(Point(i, rowValue[i]));
	}
	imshow("亮度均值", image_tmp);
	putText(blackRow, "value:-", Point(blackRow.cols - 80, 15), 1, 1, valueColor, 1);
	putText(blackRow, "grad:-", Point(blackRow.cols - 80, 35), 1, 1, gradColor, 1);

	//-------对亮度均值进行线性拟合------
	Mat A;
	polynomial_curve_fit(rowValue_points, 1, A);//线性拟合
	std::vector<cv::Point> points_fitted;
	for (int x = 0; x < rowValue.size(); x++)
	{
		double y = A.at<double>(0, 0) + A.at<double>(1, 0) * x;
		points_fitted.push_back(cv::Point(x, 255-y));
	}
	cv::polylines(blackRow, points_fitted, false, valueColor, 1, 8, 0);
	imshow("梯度、亮度均值", blackRow);
	
	//判断亮暗
	if (rowValueMax < 80) {
		for (int i = 0; i < signalRow.size() - 1; i += 2)
		{
			Point site;
			site.x = dst.cols / 2 - 2;
			site.y = (signalRow[i] + signalRow[i + 1]) / 2 + 2;
			putText(dst, "0", site, 1, 1, Scalar(0, 0, 255), 2, 8);


			//imshow("dst", dst);
			//imwrite(resultpath + referImage + "6.jpg", resultRoiImage);
		}
	}
	else
	{
		/*for (int i = 0; i < rowSobelTop.size(); i++) {

			Point site;
			site.x = dst.cols / 2 - 2;
			site.y = rowSobelTop[i] + 5;
			if (rowValue[rowSobelTop[i]] > 60 && rowValue[rowSobelTop[i]] > rowValueMax / 1.5) {

				putText(dst, "1", site, 1, 1, Scalar(0, 255, 0), 2, 8);
			}

		}*/
		int lampNum = 16;//信号灯数量
		int gradThresh = 15;//梯度阈值
		vector<int>high_gradRow;//高梯度的行
		for (int i = 0; i < rowSobelTop.size(); i++)
		{

			if (rowSobelTop[i] < colSums.size() / (4 * lampNum)) {
				for (int k = 0; k < rowSobelTop[i] + colSums.size() / (4 * lampNum); k++) {
					if (colSums[k] > colSums[rowSobelTop[i]] - gradThresh)
						high_gradRow.push_back(k);
				}
			}
			else if (rowSobelTop[i] > (colSums.size() - colSums.size() / (4 * lampNum))) {
				for (int k = rowSobelTop[i] - colSums.size() / (4 * lampNum); k < colSums.size(); k++) {
					if (colSums[k] > colSums[rowSobelTop[i]] - gradThresh)
						high_gradRow.push_back(k);
				}
			}
			else
			{
				for (int k = rowSobelTop[i] - colSums.size() / (4 * lampNum); k < rowSobelTop[i] + colSums.size() / (4 * lampNum); k++) {
					if (colSums[k] > colSums[rowSobelTop[i]] - gradThresh)
						high_gradRow.push_back(k);
				}
			}
		}
		sort(high_gradRow.begin(), high_gradRow.end());
		/*int tmp = 0;
		vector<int>high_continuegrad;
		for (int j = 1; j < high_gradRow.size(); j++) {
			if (abs(high_gradRow[j] - rowSobelTop[tmp]) > colSums.size() / (4 * lampNum)) {
				tmp++;
				high_continuegrad.push_back(high_gradRow[j - 1]);
			}
		}
		high_continuegrad.push_back(high_gradRow[0]);
		high_continuegrad.push_back(high_gradRow[high_gradRow.size() - 1]);
		sort(high_continuegrad.begin(), high_continuegrad.end());*/

		//----求梯度峰值附近的亮度均值
		vector<double>meanValue;//每个最高剃度峰值附近的亮度均值
		vector<int>meanRow;//梯度峰值的均值行
		vector<int>diffnum;
		double tmp = 0;
		int num = 0;
		int tmp2 = 0;
		for (int i = 0; i < high_gradRow.size() - 1; i++) {
			
			//cout << high_gradRow[i] << endl;
			if (high_gradRow[i + 1] - high_gradRow[i] < colSums.size() / 50) {
				tmp += rowValue[high_gradRow[i]];
				num++;
				tmp2 += high_gradRow[i];
			}
			else
			{
				meanValue.push_back(tmp);
				diffnum.push_back(num);
				meanRow.push_back(tmp2);
				tmp=0;
				num = 0;
				tmp2 = 0;
			}
			if (i == high_gradRow.size() - 2) {
				meanValue.push_back(tmp);
				diffnum.push_back(num);
				meanRow.push_back(tmp2);
			}
		}
		for (int i = 0; i < meanValue.size(); i++) {
			if (diffnum[i] > 0) {
				meanValue[i] /= diffnum[i];
				meanRow[i] /= diffnum[i];
			}
			double y =A.at<double>(0, 0) + A.at<double>(1, 0) * meanValue[i];
			if (meanValue[i] > y + 10) {
				Point site;
				site.x = dst.cols / 2 - 2;
				site.y = meanRow[i] + 5;
				putText(dst, "1", site, 1, 1, Scalar(0, 255, 0), 2, 8);

			}
			cout << i<<"	亮度均值:"<< meanValue[i] << "	hang:" << meanRow[i] << "	拟合亮度值:" << y << endl;
		}

		//for (int i = 0; i < high_gradRow.size(); ++i) {
		//	line(blackRow, Point(high_gradRow[i], 0), Point(high_gradRow[i], 255), Scalar(0, 0, 255), 1, 8);
		//}
		//imshow("梯度、亮度均值2", blackRow);
	

	}
	
	/*------------根据均值判断灯的状态--------------*/
	/*Mat  stdev;
	Mat m(Size(1, 0), CV_32FC1);
	meanStdDev(grayRoiImage, m, stdev);
	double mean = m.at<double>(0, 0);
	cout << "均值：" << mean << endl;
	Mat binaryRoiImage = grayRoiImage.clone();
	vector<double>signalMeanValue(signalRow.size() / 2);
	//vector<vector<int>>histCal;//直方图统计
	for (int i = 0; i < signalRow.size() - 1; i += 2)
	{
		//vector<int>tmpHist(256);
		for (int j = signalRow[i]; j < signalRow[i + 1]; j++) {
			for (int k = 0; k < grayRoiImage.cols; k++) {
				int tmp = grayRoiImage.at<uchar>(j, k);
				//tmpHist[tmp]++;
				signalMeanValue[i / 2] += grayRoiImage.at<uchar>(j, k);
				if (binaryRoiImage.at<uchar>(j, k) > 120)binaryRoiImage.at<uchar>(j, k) = 255;
				else
				{
				binaryRoiImage.at<uchar>(j, k) = 0;
				}
			}
		}
		//histCal.push_back(tmpHist);
		signalMeanValue[i / 2] /= ((signalRow[i + 1] - signalRow[i])*grayRoiImage.cols);
		//对图片进行均值判断
	
		Point site;
		site.x = dst.cols / 2 - 2;
		site.y = (signalRow[i] + signalRow[i + 1]) / 2 + 2;
		if (signalMeanValue[i / 2] > mean)
			putText(dst, "1", site, 1, 1, Scalar(255, 0, 0), 2, 8);
		else
		{
			putText(dst, "0", site, 1, 1, Scalar(0, 255,0 ), 2, 8);
		}
	
		//imshow("dst", dst);
		//imwrite(resultpath + referImage + "6.jpg", resultRoiImage);
	}*/

	/*sort(signalMeanValue.begin(), signalMeanValue.end());
	for (int i = 0; i < signalMeanValue.size(); i++) {
		cout << i + 1 << ": " << signalMeanValue[i] << endl;
	}
	threshold(binaryRoiImage, binaryRoiImage, 250, 255, THRESH_BINARY);
	imshow("binaryRoiImage", binaryRoiImage);
	Mat openImage;
	Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
	morphologyEx(binaryRoiImage, openImage, MORPH_OPEN, element);
	imshow("openImage", openImage);*/

}

//最小梯度进行分割
void sobel_segement(Mat roImage) {
	//利用背景梯度值低的特点对信号灯分割
	Mat sobel_y, medianImage, grayroImage;
	cvtColor(roImage, grayroImage, CV_BGR2GRAY);
	//imshow("gray", grayroImage);
	//equalizeHist(grayroImage, grayroImage);
	medianBlur(grayroImage, medianImage, 3);
	//imshow("upandroweuqal", medianImage);

	Sobel(medianImage, sobel_y, -1, 1, 0, 3);
	//imshow("sobel_y", sobel_y);
	//Mat cannyImage;
	//Canny(sobel_y, cannyImage, 80, 160);
	//imshow("cannyImage", cannyImage);
	//计算每行的平均梯度值
	vector<double>colSums(sobel_y.rows);
	for (int i = 0; i < sobel_y.rows; i++) {
		double coltmp = 0.0;
		for (int j = 0; j < sobel_y.cols; j++) {
			coltmp += sobel_y.at<uchar>(i, j);
		}
		colSums[i] = coltmp / sobel_y.cols;
	}
	vector<double>mincolSums(colSums);//最小平均梯度值
	sort(mincolSums.begin(), mincolSums.end());
	Mat blackRow = Mat::zeros(Size(colSums.size(), 100), sobel_y.type());
	vector<int>rowNum;
	for (int i = 0; i < colSums.size() - 1; i++) {
		Point pre, cur;
		pre.x = i;
		pre.y = (1 - colSums[i] / (mincolSums[mincolSums.size() - 1] - mincolSums[0])) * 100;
		cur.x = i + 1;
		cur.y = (1 - colSums[i + 1] / (mincolSums[mincolSums.size() - 1] - mincolSums[0])) * 100;
		line(blackRow, pre, cur, Scalar(255), 1, 8);//画出梯度图
													//取出小的平均梯度值的行数
		if (colSums[i] < mincolSums[0] + 4) {
			//cout << i << endl;
			rowNum.push_back(i);
		}
	}
	//imshow("梯度均值", blackRow);
	//imwrite("result\\" + referImage + "6.jpg", blackRow);
	int borden = 15;//匹配上下边界
					//if (src.rows > 300)borden = 20;
	vector<int>rowNonSignal;
	if (colSums.size() > 2 * borden) {

		//中间
		for (int i = borden; i < colSums.size() - borden; i++) {
			int judge = 0;
			//int bordenNum = 0;
			for (int j = 1; j < borden; j++) {

				if (((colSums[i]) > mincolSums[0] * 3.0) || ((colSums[i]) >(colSums[i + j])) ||
					((colSums[i]) > (colSums[i - j]))) {
					//bordenNum++;
					judge = 0;
					break;
				}
				else
				{
					judge = 1;

				}
			}
			//if (((1.0 - tmplateScore[i]) > score_max *0.85))judge = 1;
			if (judge == 1) {
				rowNonSignal.push_back(i);
				cout << i << " ";
			}
		}
		cout << endl;

	}
	vector<int>allNonSignal;
	for (int i = 0; i < rowNonSignal.size(); i++)
	{
		for (int j = -borden; j < borden; j++)
		{
			if (colSums[rowNonSignal[i]] + 2 > colSums[rowNonSignal[i] + j]) {
				allNonSignal.push_back(rowNonSignal[i] + j);
			}
		}
	}
	if (allNonSignal.size() > 0) {
		sort(allNonSignal.begin(), allNonSignal.end());
		map<int, int>rowNonsignalMap;
		for (int i = 0; i < allNonSignal.size() - 1; i++) {
			static int tmp = allNonSignal[0];
			if (allNonSignal[i + 1] - allNonSignal[i] > 1) {
				rowNonsignalMap.insert(pair<int, int>(tmp, allNonSignal[i]));
				tmp = allNonSignal[i + 1];
			}
			if (i == allNonSignal.size() - 2)rowNonsignalMap.insert(pair<int, int>(tmp, allNonSignal[i + 1]));
		}
		map<int, int>::iterator iter;
		for (iter = rowNonsignalMap.begin(); iter != rowNonsignalMap.end(); iter++) {
			rectangle(roImage, Rect(Point(0, iter->second), Point(roImage.cols, iter->first)), Scalar(0, 0, 255), 2, 8);
		}
		imshow("rowNOnsignal", roImage);

	}
}

//进行线性拟合
bool polynomial_curve_fit(std::vector<cv::Point>& key_point, int n, cv::Mat& A)
{
	//Number of key points
	int N = key_point.size();

	//构造矩阵X
	cv::Mat X = cv::Mat::zeros(n + 1, n + 1, CV_64FC1);
	for (int i = 0; i < n + 1; i++)
	{
		for (int j = 0; j < n + 1; j++)
		{
			for (int k = 0; k < N; k++)
			{
				X.at<double>(i, j) = X.at<double>(i, j) +
					std::pow(key_point[k].x, i + j);
			}
		}
	}

	//构造矩阵Y
	cv::Mat Y = cv::Mat::zeros(n + 1, 1, CV_64FC1);
	for (int i = 0; i < n + 1; i++)
	{
		for (int k = 0; k < N; k++)
		{
			Y.at<double>(i, 0) = Y.at<double>(i, 0) +
				std::pow(key_point[k].x, i) * key_point[k].y;
		}
	}

	A = cv::Mat::zeros(n + 1, 1, CV_64FC1);
	//求解矩阵A
	cv::solve(X, Y, A, cv::DECOMP_LU);
	return true;
}


