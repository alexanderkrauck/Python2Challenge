# Python2Challenge
My implementation for our Python Challenge (of the JKU python2 course)

## Dataset Folder
This folder contains the dataset collected for the Programming in Python II Image Extrapolation Challenge 2021. In total, it contains \~40,000 files.

The `.zip` files each contain up to 100 submissions, with each submission containing \~100 images. Each folder in the `.zip` files corresponds to the submission of a student, their order is random.

All image files were converted to grayscale using the function `rgb2gray` from file `04_solutions.py`, task 01, with the following weighting of the color channels: r=0.2989, g=0.5870, b=0.1140 .

Folder analysis contains the outputs of file `04_data_analysis.py`, computed for the whole dataset (this took around 1-2h on our servers).
