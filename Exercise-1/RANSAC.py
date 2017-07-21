#Import PCL module
import pcl

# Load Point Cloud file
cloud = pcl.load_XYZRGB('tabletop.pcd')

# Voxel Grid filter
## Create voxelgrid filter object
vox = cloud.make_voxel_grid_filter()
## Choose voxel (leaf) size .. 1 is bad
## This basically sets the distance between filtered pixels.
## Obviously 1 would mean that there is a gap between pixels, and you wouldn't
## Achieve meaningful data. But if the number is too small,
# then you would be processing too much data, and render the filter ineffective.
LEAF_SIZE = 0.005
## Set the leaf size in the filter
vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)

# Save pcd for table
cloud_filtered = vox.filter()
filename = 'voxel_downsampled.pcd'
pcl.save(cloud_filtered, filename)

# PassThrough filter:- This basically reduces the volume of the data that
## is being filtered, here we've only acted in the 'z' axis,
##  because there is no surrounding data that we need to block out.
## Create Passthrough filter object
passthrough = cloud_filtered.make_passthrough_filter()
# Assign axis and range to the filter
filter_axis = 'z'
passthrough.set_filter_field_name(filter_axis)
axis_min = 0.759    ## Course suggests 0.8, but this allows for 
		  ## bottom of objects too
axis_max = 2 
passthrough.set_filter_limits(axis_min,axis_max)

# Save pcd for table
cloud_filtered = passthrough.filter()
filename = 'pass_through_filtered.pcd'
pcl.save(cloud_filtered, filename)

# RANSAC plane segmentation
seg = cloud_filtered.make_segmenter()

## Set model for fit
seg.set_model_type(pcl.SACMODEL_PLANE)
seg.set_method_type(pcl.SAC_RANSAC)
# Max Distance considered for a point fitting model, make smaller for greater
# resolution
max_distance = 0.0185
seg.set_distance_threshold(max_distance) 

# Extract inliers
inliers, coefficients = seg.segment()
extracted_inliers = cloud_filtered.extract(inliers, negative = False)
filename = 'extracted_inliers.pcd'
pcl.save(extracted_inliers, filename)
# Extract outliers
extracted_outliers = cloud_filtered.extract(inliers, negative=True)
filename = 'extracted_outliers.pcd'
pcl.save(extracted_outliers, filename)

# Save pcd for tabletop objects


