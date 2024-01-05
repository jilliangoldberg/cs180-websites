# Computational Photography Projects
## UC Berkeley CS 180 | Fall 2023

These are my projects from [CS 180](https://inst.eecs.berkeley.edu/~cs180/fa23/), a class taught by [Alexei (Alyosha) Efros](http://www.eecs.berkeley.edu/~efros/) and [Angjoo Kanazawa](http://people.eecs.berkeley.edu/~kanazawa/). The class focuses on computing with visual data, which includes the image processing, computer vision, computational photography, and the use of convolutional neural networks.
### Final Project
My final project was comprised of 3 different projects demonstrating different techniques for image manipulation.
1. **Gradient Domain Fusion:** This project implemented a technique used for blending, tone-mapping, and non-photorealistic rendering, which we used to seamlessly blend objects into target images. [Project Spec](https://inst.eecs.berkeley.edu/~cs194-26/sp20/hw/gradient-domain-fusion/) | [My Project Website](https://htmlpreview.github.io/?https://github.com/jilliangoldberg/cs180-websites/blob/main/finalproject/index.html)
2. **Image Quilting:** This project involves creating a larger texture image from a small sample by sampling patches and ensuring similarities in overlapping regions. [Project Spec](https://yxw.cs.illinois.edu/course/CS445/F21/projects/quilting/ComputationalPhotography_ProjectQuilting.html) | [My Project Website](https://htmlpreview.github.io/?https://github.com/jilliangoldberg/cs180-websites/blob/main/finalproject/proj2-index.html)
3. **Lightfield Camera**: This project mimics depth refocusing and aperature adjustment by capturing an array of images over a plane orthogonal to the object and using simple techniques such as shifting and averaging. [Project Spec](https://inst.eecs.berkeley.edu/~cs194-26/fa17/hw/proj5/) | [My Project Website](https://htmlpreview.github.io/?https://github.com/jilliangoldberg/cs180-websites/blob/main/finalproject/proj3-index.html)

### Project 1: Colorizing the [Prokudin-Gorskii](https://www.loc.gov/collections/prokudin-gorskii/) photo collection 📸
[Project Spec](https://inst.eecs.berkeley.edu/~cs180/fa23/hw/proj1/) | [My Project Website](https://htmlpreview.github.io/?https://github.com/jilliangoldberg/cs180-websites/blob/main/proj1/proj1.html)
<br> This project focused on image processing to extract color channels and align them to form a single RGB color image.

### Project 2: Fun with Filters and Frequencies! 🍎
[Project Spec](https://inst.eecs.berkeley.edu/~cs180/fa23/hw/proj2/) | [My Project Website](https://htmlpreview.github.io/?https://github.com/jilliangoldberg/cs180-websites/blob/main/proj2/proj2.html)
<br> This project builds off of project 1 by introducing gaussians, 2d convolutions, filtering, and multiresolution blending.

### Project 3: Face Morphing 🎭
[Project Spec](https://inst.eecs.berkeley.edu/~cs180/fa23/hw/proj3/) | [My Project Website](https://htmlpreview.github.io/?https://github.com/jilliangoldberg/cs180-websites/blob/main/proj3/proj3.html)
<br> The goal of this project was to produce a morphing sequence of 2 faces, which can be done by defining corresponding points, computing a midway face through triangularization and averaging, and changing alpha values.

### Project 4: Image Warping and Mosaicing 🌅
[Project Spec](https://inst.eecs.berkeley.edu/~cs180/fa23/hw/proj4/) | [My Project Website](https://htmlpreview.github.io/?https://github.com/jilliangoldberg/cs180-websites/blob/main/proj4/proj4.html)
<br> This project focused on image mosaicing by taking two or more photographs and creating an image mosaic by registering, projective warping, resampling, and compositing them. This includes computing homographies and using them to warp images.

### Project 5: Neural Radiance Field! (NeRFs!) 🎞️
[Project Spec](https://inst.eecs.berkeley.edu/~cs180/fa23/hw/proj5/) | [My Project Website](https://htmlpreview.github.io/?https://github.com/jilliangoldberg/cs180-websites/blob/main/proj5/proj5.html)
<br> The goal of this assignment is to learn how to implement both 2D neural fields and 3D Neural Radiance Fields.
