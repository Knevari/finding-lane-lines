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

#### Converting the image to grayscale
<img src="examples/grayscale_img.jpg" width="480" alt="Grayscale Image" />
```python
image = mpimg.imread("images/example.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
```
It's easier for us to work with grayscale images because they have fewer details than color images, after converting from BGR to gray we then adjust (optional) the image gamma to facilitate the feature extraction from the image

#### After adjusting the grayscale image gamma
<img src="examples/dark_grayscale_img.jpg" width="480" alt="Dark Grayscale Image" />

#### Convert original BGR image to HLS
<img src="examples/hls_img.jpg" width="480" alt="HLS Image" />
```python
hls_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
```
We convert from BGR color space to HLS because we can retrieve color information more easily than in plain BGR

#### Create masks to filter out unnecessary details
<img src="examples/masks.jpg" width="480" alt="Masks" />
```python
lower_white = np.array([0, 210, 0], dtype=np.uint8)
upper_white = np.array([210, 255, 255], dtype=np.uint8)
white_mask = cv2.inRange(hls, lower_white, upper_white)

lower_yellow = np.array([10, 0, 100], dtype=np.uint8)
upper_yellow = np.array([30, 255, 255], dtype=np.uint8)
yellow_mask = cv2.inRange(hls, lower_yellow, upper_yellow)

combined_masks = cv2.bitwise_or(white_mask, yellow_mask)
```
We try to find parts of the image which are within our white and yellow color ranges and then create a single mask using OpenCV bitwise_or function

#### Apply the created mask to our gray image
<img src="examples/masks.jpg" width="480" alt="Masks" />
```python
masked_image = cv2.bitwise_and(gray, combined_masks)
```
#### Apply Gaussian Blur
<img src="examples/gaussian_blur.jpg" width="480" alt="Gaussian Blur" />
```python
kernel_size = 5
blur = cv2.GaussianBlur(masked_image, (kernel_size, kernel_size), 0)
```
[Gaussian Blur](https://en.wikipedia.org/wiki/Gaussian_blur#:~:text=In%20image%20processing%2C%20a%20Gaussian,image%20noise%20and%20reduce%20detail) is a widely used effect in computer graphics to reduce the image noise and detail, after running our image through the OpenCV GaussianBlur function the output is a smoothed image that we need to feed the edge detection algorithm. This function receives 3 parameters, the image, kernel_size and the standard deviation

#### Canny Edge Detection
There are a handful of edge detection algorithms out there, [Canny](https://en.wikipedia.org/wiki/Canny_edge_detector#:~:text=The%20Canny%20edge%20detector%20is,explaining%20why%20the%20technique%20works.) does it based on gradient changes. It is important to notice that Canny also applies blurring in the beginning of the function, but we apply Gaussian Blur to smooth even more before detecting the edges