#include <iostream>
#include <valarray>
#include <ctime>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <chrono>

using namespace std;

#define max_kmeans 13
#define MAX_ITERATION 30

valarray<float> increment(const valarray<float>& v) { return v*(-2); }

valarray<valarray<float>> prodMat(valarray<valarray<float>> &mat1, valarray<valarray<float>> &mat2,int upto) {
	int m = mat1.size();
	int k = mat2.size();
	valarray<valarray<float>> res(m);

	for (int i = 0; i < m; i++) {
		res[i] = valarray<float>(k);
		for (int j = 0; j < upto; j++) {
			res[i][j] = (mat1[i] * mat2[j]).sum();
		}
		
	}
	return move(res);
}

valarray<valarray<float>> distMat(valarray<valarray<float>> &dataMatrix, valarray<valarray<float>> &centers,int upto) {
	int data_rows = dataMatrix.size();
	int center_rows = centers.size();

	auto productMat = prodMat(dataMatrix, centers, upto);

	valarray<valarray<float>> matX(productMat.size());
	valarray<valarray<float>> matY(productMat.size());

	valarray<valarray<float>> temp = dataMatrix * dataMatrix;

	for (int i = 0; i < dataMatrix.size(); i++)
		matX[i] = valarray<float>(temp[i].sum(), productMat[0].size());

	temp = centers * centers;
	valarray<float> sumY(centers.size());
	for (int i = 0; i < upto; i++)
		sumY[i] = temp[i].sum();

	for (int i = 0; i < productMat.size(); i++)
		matY[i] = sumY;

	productMat = productMat.apply(increment);
	return move(matX) + productMat + matY;
}

valarray<int> findCenters(valarray<valarray<float>> &dataMatrix, valarray<valarray<float>> &centers,int k) {
	valarray<int> indexMat(dataMatrix.size());
	auto distmat = distMat(dataMatrix, centers,k);

	for (int i = 0; i < distmat.size(); i++) {
		float minval = distmat[i][0];
		indexMat[i] = 0;
		for (int j = 1; j < distmat[0].size(); j++)
			if (distmat[i][j] < minval) {
				minval = distmat[i][j];
				indexMat[i] = j;
				}
	}
	return move(indexMat);
}

int probability_index(valarray<float>& distances) {
	long int sum = distances.sum();
	long int random_sum = rand() % sum;

	for (int i = 0; i < distances.size(); i++) 
	{
		random_sum -= distances[i];
		if (random_sum < 0)
			return i;
	}
}

valarray<valarray<float>> initCenters(valarray<valarray<float>> &dataMatrix, int k) {
	int randomNum = rand() % dataMatrix.size(); //Generates number between 0 - 99,999
	valarray<valarray<float>> centers(k);
	centers[0] = dataMatrix[randomNum];
	valarray<float> minDistMat(dataMatrix.size());
	
	for (int i = 0; i < k-1; i++) {
		auto distmat = distMat(dataMatrix, centers,i+1);
		
		for (int j = 0; j < dataMatrix.size(); j++) {
			valarray<float> temp = distmat[j][slice(0, i + 1, 1)];
			minDistMat[j] = temp.min();
		}
		
		centers[i+1] = dataMatrix[probability_index(minDistMat)];
	}
	return move(centers);
}

valarray<int> kmeansPP(valarray<valarray<float>> &dataMatrix, int k) {
	int data_rows = dataMatrix.size();
	auto centers = initCenters(dataMatrix, k);
	valarray<int> old_clusters(-1, dataMatrix.size());
	valarray<valarray<float>> c(k);
	int count = 0;
	while (true) {
		count++;
		if (count >= MAX_ITERATION)
			return move(old_clusters);
		auto new_clusters = findCenters(dataMatrix, centers,k);
		if ((new_clusters != old_clusters).sum() == 0)
			return move(new_clusters);
		old_clusters = move(new_clusters);

		for (int i = 0; i < k; i++) {
			valarray<valarray<float>> tmp = dataMatrix[old_clusters == i];
			c[i] = tmp.sum()/tmp.size();
		}
		centers = c;
	}
}

float silhouette(valarray<valarray<float>>& dataMatrix, valarray<int>& clusters, int k) {
	int data_rows = dataMatrix.size();
	valarray<valarray<float>> centers(k);
	int count = 0;

	for (int i = 0; i < k; i++) {
		valarray<valarray<float>> tmp = dataMatrix[clusters == i];
		centers[i] = tmp.sum() / tmp.size();
	}

	auto dist = distMat(dataMatrix, centers,k);
	valarray<float> a(0.0, data_rows);
	valarray<float> b(0.0, data_rows);
	valarray<float> temp(k - 1);
	for (int i = 0; i < data_rows; i++) {
		a[i] = dist[i][clusters[i]];

		int jindex = 0;
		for (int j = 0;j < k ;j++)
			if (clusters[i] != j)
				temp[jindex++] = dist[i][j];

		b[i] = temp.min();
	}

	valarray<float> c(0.0, data_rows);
	for (int i = 0;i < c.size();i++)
		c[i] = max(a[i], b[i]);

	auto silo = (((b - a) / c).sum()) / data_rows;
	return silo;
}

int main() {
	srand(time(NULL));
	ifstream rFile;
	rFile.open("data_1_3.txt");
	valarray<valarray<float>> dataMat(100000);
	string s = "";
	int j = 0;

	while (getline(rFile, s))
	{
		stringstream lineStream(s);
		string bit;
		valarray<float> vector(100);
		int i = 0;
		while (getline(lineStream, bit, ','))
		{
			if (bit.compare("\n") != 0) {
			vector[i] = stof(bit);
			i++;
			}
		}

		getline(lineStream, bit, '\n');
		vector[99] = stof(bit);
		dataMat[j] = vector;
		j++;
	}
	auto start = chrono::high_resolution_clock::now();
	valarray<int> x = kmeansPP(dataMat, 10);
	cout << "silhouette score for k = " << 10 << " is: " << silhouette(dataMat, x, 10) << endl;
	auto stop = chrono::high_resolution_clock::now();
	auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);
	cout << "time taken: " << duration.count() / 1000000.0 << " seconds" << endl;
	for (int i = 2; i <= 10;i++) {
		valarray<int> x = kmeansPP(dataMat, i);
		cout << "silhouette score for k = " << i <<" is: " << silhouette(dataMat, x, i) << endl;
	}
}