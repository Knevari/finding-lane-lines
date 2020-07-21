# **Finding Lane Lines**
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
### Usage
```
py main.py videos/video_name.mp4
```

### Project
This is the first project of Udacity's self driving car engineer nanodegree. I had to code a pipeline for detecting lane lines on road videos and images.

### Pipeline
1. Convert image to grayscale
2. Adjust the grayscale image gamma
2. Convert original image to hsl channel space
3. Create yellow and white masks to filter the hsl image from unnecessary details
4. Apply the created mask to the gray image
5. Apply gaussian blur to facilitate the use of edge detection algorithm
6. Retrieve the image edges using canny edge detection algorithm
7. Create a region of interest (roi) mask to boil out some more unnecessary details
8. Apply Hough Transform algorithm to find lines in the image
9. Get the average slope and y-intercept of all lines all generate one single line for each lane

