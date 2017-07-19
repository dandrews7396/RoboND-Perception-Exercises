#!/usr/bin/python

import pcl

p = pcl.load("filename.pcd")

fil = p.make_statistical_outlier_filter()
fil.set_mean_k(50)
fil.set_std_dev_mul_thresh(1.0)

pcl.save(fil.filter(), "filename.pcd")

fil.set_negative(True)
pcl.save(fil.filter(), "filename.pcd")
