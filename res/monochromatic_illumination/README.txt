README

Monochromatic Illumination - data analysis:

2022.06.02 Test 2:

To find best illumination, need to find highest difference in luminance of dura & bone.
1.	Save 1 frame from one of the videos and open it with GIMP. 
2.	Find coordinates of a rectangle that covers each desired tissue-type (both rectangles with the same size).
3.	In python code (luminance_extract): insert coordinates & size of rectangles. And run code for each video. This will produce large csv file with luma values for each tissue-type and each test-type.
4.	In matlab (luma_analysis): load the .csv file and save it as .mat
5.	Run code, which will reorganize data & perform One-Way ANOVA test.
