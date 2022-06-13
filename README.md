# K-means-and-Silhouette-Algorithm-cpp
This algorithm is an improvement to the k-means algorithm by using vectorization methods.
## K-means algorithem:
1. Decide on a value for k.	
2. Initialize the k cluster centers.	
3. Decide the class memberships of the N objects by assigning them to the nearest cluster center.	
4. Re-estimate the k cluster centers, by assuming the memberships found above are correct.	
5. If none of the N objects changed membership in the last iteration, exit. Otherwise goto 3.

## K-means++ Initialization:
1. Randomly select the first centroid from the data points.
2. For each data point compute its distance from the nearest, previously chosen centroid.
3. Select the next centroid from the data points such that the probability of choosing a point as centroid is directly proportional to its distance from the nearest, previously chosen centroid. (i.e. the point having maximum distance from the nearest centroid is most likely to be selected next as a centroid)
4. Repeat steps 2 and 3 until k centroids have been sampled

## K-means++ optimization:
1. Valarray
2. Move semantics
3. References
4. Vectorization:

![image](https://user-images.githubusercontent.com/74405706/173426461-bd4e57e5-86c6-41d2-9a17-7147d87f16d4.png)
![image](https://user-images.githubusercontent.com/74405706/173426521-de9eeb43-9365-48e3-b106-bceeadd6984e.png)

## Results
Our implementation’s speed is approximately: 43 seconds (on my machine)
This is worse than the python implementation because we didnt use AVX instructions so we could not achive NumPy level of optimization.



