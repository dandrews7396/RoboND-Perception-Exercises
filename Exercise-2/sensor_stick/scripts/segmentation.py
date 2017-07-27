#!/usr/bin/env python

# Import modules
from pcl_helper import *

# Define functions as required

# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):

    #Convert ROS msg to PCL data
    cloud = ros_to_pcl(pcl_msg)

    # Voxel Grid Downsampling
    vox = cloud.make_voxel_grid_filter()
    # create leaf size
    LEAF_SIZE = 0.005
    # Set leaf size for voxel filter
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
    # Apply Voxel filter
    vox_filtered = vox.filter()

    # PassThrough Filter
    filter_axis = 'z' ## Set filter axis
    passthrough = vox_filtered.make_passthrough_filter()
    passthrough.set_filter_field_name(filter_axis)
    # Set filter range
    axis_min = 0.759 # Taken from Exercise 1
    axis_max = 2
    passthrough.set_filter_limits(axis_min, axis_max)
    # Apply Passthrough filter
    pt_filtered = passthrough.filter()

    # RANSAC Plane Segmentation
    seg = pt_filtered.make_segmenter()

    # Set model for fit
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    # Max Distance for point fitting models.
    max_distance = 0.0185
    seg.set_distance_threshold(max_distance)

    # Extract inliers and outliers
    inliers, coefficients = seg.segment()
    cloud_table = pt_filtered.extract(inliers, negative = False)
    cloud_objects = pt_filtered.extract(inliers, negative = True)

    # Euclidean Clustering
    white_cloud = XYZRGB_to_XYZ(cloud_objects)
    tree = white_cloud.make_kdtree()

    # Create cluster extraction object
    ec = white_cloud.make_EuclideanClusterExtraction()
    # Set tolerances for distance threshold
    # and min and max cluster sizes
    ec.set_ClusterTolerance(0.01)
    ec.set_MinClusterSize(25)
    ec.set_MaxClusterSize(10000)

    # Search KD tree for clusters
    ec.set_SearchMethod(tree)
    # Extract co-ords for each discovery
    cluster_indices = ec.Extract()

    # Create Cluster-Mask Point Cloud to visualize each cluster separately
    cluster_color = get_color_list(len(cluster_indices))

    color_cluster_point_list = []

    for j, indices in enumerate(cluster_indices):
        for i, indice in enumerate(indices):
            color_cluster_point_list.append([white_cloud[indice][0],
                                            white_cloud[indice][1],
                                            white_cloud[indice][2],
                                             rgb_to_float(cluster_color[j])])

    # Create new cloud with uniquely coloured items
    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)

    # Convert PCL data to ROS messages
    ros_cloud_objects = pcl_to_ros(cloud_objects)
    ros_cloud_table = pcl_to_ros(cloud_table)
    ros_cluster_cloud = pcl_to_ros(cluster_cloud)
     
    # Publish ROS messages
    pcl_objects_pub.publish(ros_cloud_objects)
    pcl_table_pub.publish(ros_cloud_table)
    pcl_cluster_pub.publish(ros_cluster_cloud)


if __name__ == '__main__':

    # ROS node initialization
    rospy.init_node('clustering', anonymous = True)

    # Create Subscribers
    pcl_sub = rospy.Subscriber("/sensor_stick/point_cloud", pc2.PointCloud2, pcl_callback, queue_size = 1)
    
    # Create Publishers
    pcl_objects_pub = rospy.Publisher("/pcl_objects", PointCloud2, queue_size=1)
    pcl_table_pub = rospy.Publisher("/pcl_table", PointCloud2, queue_size=1)
    pcl_cluster_pub = rospy.Publisher("/pcl_cluster", PointCloud2, queue_size=1)
    
    # Initialize color_list
    get_color_list.color_list = []

    # Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.spin()
        
