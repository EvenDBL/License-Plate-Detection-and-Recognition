# license plate detection and recognition

## environment

- pytorch 1.4.0
- python 3.6

## demo

It is supported to infer a image or a set of  images.  Just run "demo.py"

## train

​	Run "seg_train.py" for training SegNet and "reg_train.py" for RegNet after you have prepared your dataset that contains images and label text file.  All images' label is written in a label text file. The format of label text file is organized like this:

```
file_name x1 y1 x2 y2 x3 y3 x4 y4 lp_chars
...
...
```

For example:

```
27.jpg 109 197 186 197 186 219 109 219 7866KR
```

The license area is located by four points. They around license area from upper left with clockwise direction. As following picture:

<img src="图片1.png" style="zoom:50%;" />

## statement

 This code will only be used for research purposes. 

