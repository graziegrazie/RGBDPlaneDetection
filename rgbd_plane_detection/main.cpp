#include "plane_detection.h"
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/LaserScan.h>
#include <image_transport/image_transport.h>
#include <camera_info_manager/camera_info_manager.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include <random>
#include <cmath>
#include "rgbd_plane_detection/rgbd_plane_detection.h"
#include "rgbd_plane_detection/rgbd_plane_detection_utils.h"
#include <geometry_msgs/Point.h>
#include <algorithm>
#include <ros/assert.h>
#include <sensor_msgs/CameraInfo.h>
#include <image_geometry/pinhole_camera_model.h>

#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>

using namespace cv;

PlaneDetection plane_detection;
image_transport::Publisher pub;
ros::Subscriber camera_info_sub;
image_geometry::PinholeCameraModel pinhole_camera_model;
typedef pcl::SampleConsensusModelPlane<pcl::PointXYZ>::Ptr SampleConsensusModelPlanePtr;

//#define DEBUG
//#define DEBUG_VIEW
#define NUM_OF_POINTS_FOR_CENTER_CALCULATION	(10)

struct normal_dev_info
{
	Eigen::Vector3d normal;
	double square_dev;
};

//-----------------------------------------------------------------
// MRF energy functions
MRF::CostVal dCost(int pix, int label)
{
	return plane_detection.dCost(pix, label);
}

MRF::CostVal fnCost(int pix1, int pix2, int i, int j)
{
	return plane_detection.fnCost(pix1, pix2, i, j);
}

void runMRFOptimization()
{
	DataCost *data = new DataCost(dCost);
	SmoothnessCost *smooth = new SmoothnessCost(fnCost);
	EnergyFunction *energy = new EnergyFunction(data, smooth);
	int width = kDepthWidth, height = kDepthHeight;
	MRF* mrf = new Expansion(width * height, plane_detection.plane_num_ + 1, energy);
	// Set neighbors for the graph
	for (int row = 0; row < height; row++)
	{
		for (int col = 0; col < width; col++)
		{
			int pix = row * width + col;
			if (col < width - 1) // horizontal neighbor
				mrf->setNeighbors(pix, pix + 1, 1);
			if (row < height - 1) // vertical
				mrf->setNeighbors(pix, pix + width, 1);
			if (row < height - 1 && col < width - 1) // diagonal
				mrf->setNeighbors(pix, pix + width + 1, 1);
		}
	}
	mrf->initialize();
	mrf->clearAnswer();
	float t;
	mrf->optimize(5, t);  // run for 5 iterations, store time t it took 
	MRF::EnergyVal E_smooth = mrf->smoothnessEnergy();
	MRF::EnergyVal E_data = mrf->dataEnergy();
	cout << "Optimized Energy: smooth = " << E_smooth << ", data = " << E_data << endl;
	cout << "Time consumed in MRF: " << t << endl;

	// Get MRF result
	for (int row = 0; row < height; row++)
	{
		for (int col = 0; col < width; col++)
		{
			int pix = row * width + col;
			plane_detection.opt_seg_img_.at<cv::Vec3b>(row, col) = plane_detection.plane_colors_[mrf->getLabel(pix)];
			plane_detection.opt_membership_img_.at<int>(row, col) = mrf->getLabel(pix);
		}
	}
	delete mrf;
	delete energy;
	delete smooth;
	delete data;
}
/*****************************************************************************************************************************************/
Result separate_region(const cv::Mat& img, std::vector<PlaneCandidateInfo>& plane_candidate_info)
{
	Result result = Succeeded;

	RNG rng(12345);
	ROS_ASSERT_MSG(nullptr != &img,                  "null image is passed as img");
	ROS_ASSERT_MSG(nullptr != &plane_candidate_info, "null image is passed as plane_candidate_info");

	for(int i = 0; i < colors.size(); i++)
	{
		PlaneCandidateInfo temp_plane_candidate_info;
		vector<vector <Point> > contours;

		inRange(img, colors[i], colors[i], temp_plane_candidate_info.img);
		findContours(temp_plane_candidate_info.img, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);
		
		// TODO: check this part is needless
		if(0 == contours.size())
		{
			continue;
		}
		else
		{
			// no operation
		}
		
		Rect biggest_rect;
		for( size_t j = 0; j < contours.size(); j++ )
		{
			Rect temp_rect;
			vector<vector <Point> > contours_poly( contours.size() );

			approxPolyDP( contours[j], contours_poly[j], 3, true );
			temp_rect = boundingRect( contours_poly[j] );

			if(temp_rect.area() < AREA_THRESHOLD)
			{
				continue;
			}
			else if(biggest_rect.area() < temp_rect.area())
			{
				biggest_rect       = temp_rect;
			}
			else
			{
				// no operation
			}
		}

		if(biggest_rect.area() < AREA_THRESHOLD)
		{
			continue;
		}
		else
		{
			// no operation
		}
		
		temp_plane_candidate_info.plane.header.frame_id = "/xtion_link";
		temp_plane_candidate_info.plane.info.id         = i;
		temp_plane_candidate_info.plane.info.width      = biggest_rect.width;
		temp_plane_candidate_info.plane.info.height     = biggest_rect.height;

		cv::Point2d temp_pose;
		temp_pose.x     = biggest_rect.x;
		temp_pose.y     = biggest_rect.y;
		temp_plane_candidate_info.top_left_pose = temp_pose;

		plane_candidate_info.push_back(temp_plane_candidate_info);
	}
	return result;
}

/*
 * @brief this function convert the pose in image to point in 3D coordinate.
 *
 */
Result get_3d_point_from_pointcloud2(const sensor_msgs::PointCloud2ConstPtr& pointcloud2_ptr, Eigen::Vector2i& pose, Eigen::Vector3d& point)
{
	Result result = Succeeded;

	// check if pose is located beyond image and pointcloud region
	if(pointcloud2_ptr->width <= pose[0] || pointcloud2_ptr->height <= pose[1])
	{
		point[0] = std::numeric_limits<double>::quiet_NaN();
		point[1] = std::numeric_limits<double>::quiet_NaN();
		point[2] = std::numeric_limits<double>::quiet_NaN();
		
		return UnexpectedPointEntered;
	}
	else
	{
		// keep going
	}

	int arrayPosition = pose[1] * pointcloud2_ptr->row_step + pose[0] * pointcloud2_ptr->point_step;
	int arrayPosX = arrayPosition + pointcloud2_ptr->fields[0].offset; // X has an offset of 0
	int arrayPosY = arrayPosition + pointcloud2_ptr->fields[1].offset; // Y has an offset of 4
	int arrayPosZ = arrayPosition + pointcloud2_ptr->fields[2].offset; // Z has an offset of 8

	float x, y, z;

	memcpy(&x, &(pointcloud2_ptr->data[arrayPosX]), sizeof(float));
	memcpy(&y, &(pointcloud2_ptr->data[arrayPosY]), sizeof(float));
	memcpy(&z, &(pointcloud2_ptr->data[arrayPosZ]), sizeof(float));

	point[0] = x;
	point[1] = y;
	point[2] = z;

	return result;
}

/*
 * @brief this function find white pixels within plane caididate
 * @param[in]
 * @param[out] pose		vector contains all white pixel positions within candidate plane region
 */
Result get_white_points_from_gray_scale_image(PlaneCandidateInfo& plane_candidate_info, std::vector<Eigen::Vector2i>& poses)
{
	Result result = Succeeded;

	int plane_width  = plane_candidate_info.plane.info.width;
	int plane_height = plane_candidate_info.plane.info.height;

	int plane_top_left_x = plane_candidate_info.top_left_pose.x;
	int plane_top_left_y = plane_candidate_info.top_left_pose.y;

	for(int col = plane_top_left_x; col < (plane_width + plane_top_left_x); col++)
	{
		for(int row = plane_top_left_x; row < (plane_height + plane_top_left_y); row++)
		{
			unsigned char *src = plane_candidate_info.img.ptr<unsigned char>(row); // Pointer to 1st element on Jth row.
			if(IS_WHITE_PIEXL(src[col]))
			{
				Eigen::Vector2i temp(col, row);
				poses.push_back(temp);
			}
			else
			{
				continue;
			}
		}
	}
	return result;
}

Result calc_plane_normal_ransac(const sensor_msgs::PointCloud2ConstPtr& pointcloud2_ptr, PlaneCandidateInfo& plane_candidate_info)
{
	Result result = Succeeded;

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ (new pcl::PointCloud<pcl::PointXYZ> ());
	std::vector<int> indices_;
	std::vector<Eigen::Vector2i> poses;
	Eigen::Vector3d normalized_normal(0.0, 0.0, 0.0);

	get_white_points_from_gray_scale_image(plane_candidate_info, poses);

	for(auto itr: poses)
	{
		Eigen::Vector3d point;

		get_3d_point_from_pointcloud2(pointcloud2_ptr, itr, point);
		if(IS_NAN_FOR_POINT(point))
		{
			continue;
		}
		else
		{
			pcl::PointXYZ temp(point[0], point[1], point[2]);
			cloud_->push_back(temp);
		}
	}

	if(cloud_->size() == 0)
	{
		normalized_normal(0) = std::numeric_limits<double>::quiet_NaN();
		normalized_normal(1) = std::numeric_limits<double>::quiet_NaN();
		normalized_normal(2) = std::numeric_limits<double>::quiet_NaN();
	}
	else
	{
		SampleConsensusModelPlanePtr model (new pcl::SampleConsensusModelPlane<pcl::PointXYZ> (cloud_));
		// Create the RANSAC object
		pcl::RandomSampleConsensus<pcl::PointXYZ> sac (model, 0.03);
		bool ransac_result = sac.computeModel ();
		Eigen::VectorXf coeff;
		sac.getModelCoefficients (coeff);

		normalized_normal << coeff[0], coeff[1], coeff[2];
		if(coeff[2] < 0)
		{
			ROTATE_180_DEG_AROUND_Y_AXIS(normalized_normal);
		}
		else
		{
			// no operation
		}
	}

	plane_candidate_info.plane.pose.pi1 = normalized_normal(0);
	plane_candidate_info.plane.pose.pi2 = normalized_normal(1);
	plane_candidate_info.plane.pose.pi3 = normalized_normal(2);

	return result;
}

/*
 * @brief This function assumes that plane is orhogonal to floor, and some points on plane center column have segmented.
 */
Result calc_plane_distance_from_camera(const sensor_msgs::PointCloud2ConstPtr& pointcloud2_ptr, PlaneCandidateInfo& plane_candidate_info)
{
	Result result = Succeeded;

	std::mt19937 mt{ std::random_device{}() };
	std::uniform_real_distribution<double> dist(0, 1);

	int plane_height     = plane_candidate_info.plane.info.height;
	int plane_top_left_y = plane_candidate_info.top_left_pose.y;

	Eigen::Vector2i pose((plane_candidate_info.plane.info.width / 2.0) + plane_candidate_info.top_left_pose.x, 0);
	double average_distance = 0.0;
	unsigned int num_of_detected_points = 0;

	for(int i = 0; i <= NUM_OF_POINTS_FOR_CENTER_CALCULATION; i++)
	{
		Eigen::Vector3d point;

		pose[1] = dist(mt) * plane_height + plane_top_left_y;
		get_3d_point_from_pointcloud2(pointcloud2_ptr, pose, point); // Enter the pose on plane center column
		if(IS_NAN_FOR_POINT(point))
		{
			// no operation
		}
		else
		{
			plane_candidate_info.plane.pose.pi4 += point[2];
			num_of_detected_points++;
		}
	}

	if(0 == num_of_detected_points)
	{
		result = Failure;
	}
	else
	{
		plane_candidate_info.plane.pose.pi4 /= num_of_detected_points;
	}
	return result;
}

/*
 * @brief This function estimate plane normal vector. The function resolves least square problem with SVD.
 */
Result calc_plane_2d_coordinate_on_camera_coordinate(const sensor_msgs::PointCloud2ConstPtr& pointcloud2_ptr, std::vector<PlaneCandidateInfo>& plane_candidate_info)
{
	Result result = Succeeded;

	ROS_ASSERT_MSG(nullptr != pointcloud2_ptr,        "null pointcloud2 is entered");

	if(0 == plane_candidate_info.size())
	{
		result = ConditionNotSatisfied;
		return result;
	}
	else
	{
		// no operation
	}

	for(unsigned char plane_index = 0; plane_index < plane_candidate_info.size() ; plane_index++)
	{
		PlaneCandidateInfo* plane_candidate_info_ptr = &(plane_candidate_info[plane_index]);
		std::vector<Eigen::Vector3d> normalized_normals;
		Eigen::Vector3d average_normal;

		calc_plane_normal_ransac(pointcloud2_ptr, *plane_candidate_info_ptr);
		calc_plane_distance_from_camera(pointcloud2_ptr, *plane_candidate_info_ptr);
	}
	std::cout << "" << std::endl;
	return result;
}

Result extract_walls_from_candidate(std::vector<PlaneCandidateInfo>& plane_candidate_info)
{
	Result result = Succeeded;

	ROS_ASSERT(nullptr != plane_candidate_info);
	ROS_ASSERT(nullptr != wall_info);

	auto itr = plane_candidate_info.begin();
	while(itr != plane_candidate_info.end())
	{
		if( IS_NAN_FOR_4D_POINT(itr->plane.pose) )
		{
			// avoid invalid planes
			itr = plane_candidate_info.erase(itr);
			continue;
		}
		else
		{
			// no operation
		}

		plane_msgs::PlanePose& plane_pose = itr->plane.pose;
		Eigen::Vector3d plane_normal(plane_pose.pi1, plane_pose.pi2, plane_pose.pi3);

		ROS_INFO("Plane Nornal:%+1.3lf %+1.3lf %+1.3lf", plane_normal[0], plane_normal[1], plane_normal[2]);
		// TODO: Magic number should be defind as MACRO!
		if( false == CHECK_RANGE_NOT_INCL_EQUAL(Y_AXIS.dot(plane_normal), 0.15) )
		{
			// leave the planes perpendicular to floor and ceil.
			itr = plane_candidate_info.erase(itr);
			continue;
		}
		else
		{
			// no operation
		}
		itr++;
	}

	return result;
}

Result calc_plane_3d_width_and_height(PlaneCandidateInfo& wall_info)
{
	Result result = Succeeded;

	ROS_ASSERT(nullptr != wall_info);

	cv::Point2d temp_right_bottom_2d(wall_info.plane.info.width  + wall_info.top_left_pose.x,
									 wall_info.plane.info.height + wall_info.top_left_pose.y);

	cv::Point3d left_top     = pinhole_camera_model.projectPixelTo3dRay(wall_info.top_left_pose);
	cv::Point3d right_bottom = pinhole_camera_model.projectPixelTo3dRay(temp_right_bottom_2d);

	cv::Point3d plane_3d_width_and_height = wall_info.plane.pose.pi4 * (right_bottom - left_top);

	wall_info.plane.info_real.width  = plane_3d_width_and_height.x;
	wall_info.plane.info_real.height = plane_3d_width_and_height.y;

	return result;
}

Result refine_scan_with_wall_information(std::vector<PlaneCandidateInfo>& wall_info,
										 const sensor_msgs::LaserScanConstPtr& original_scan_ptr,
										 sensor_msgs::LaserScan& refined_scan)
{
	Result result = Succeeded;

	ROS_ASSERT(nullptr != wall_info);

	for(auto itr: wall_info)
	{
		calc_plane_3d_width_and_height(itr);
	}

	return result;
}

#if 1
void callback(const sensor_msgs::ImageConstPtr& depth_ptr,
			  const sensor_msgs::PointCloud2ConstPtr& pointcloud2_ptr,
			  const sensor_msgs::LaserScanConstPtr& laserscan_ptr)
#else
void callback(const sensor_msgs::ImageConstPtr& depth_ptr,
			  const sensor_msgs::PointCloud2ConstPtr& pointcloud2_ptr)
#endif
{
	ROS_INFO("callback called");

	sensor_msgs::LaserScan refined_scan;

	std::vector<PlaneCandidateInfo> plane_candidate_info;
	if(nullptr == depth_ptr )
	{
		ROS_WARN("depth_ptr is null");
		return;
	}
	else
	{
		plane_detection.readDepthImage(depth_ptr);
	}

	sensor_msgs::ImagePtr img_ptr = plane_detection.runPlaneDetection();
	pub.publish(*img_ptr);

	separate_region(plane_detection.seg_img_, plane_candidate_info);
	calc_plane_2d_coordinate_on_camera_coordinate(pointcloud2_ptr, plane_candidate_info);
	// TODO: some plane has NaN coordinate. Please remove such element in plane_candidate_info
	// TODO: consider if divided segments which are contained in same plane should be merged?

	for(auto itr: plane_candidate_info)
	{
		ROS_INFO("[A%d] %lf %lf %lf %lf", itr.plane.info.id, itr.plane.pose.pi1, itr.plane.pose.pi2, itr.plane.pose.pi3, itr.plane.pose.pi4);

		rectangle( itr.img,
		           itr.top_left_pose,
				   cv::Point(itr.top_left_pose.x + itr.plane.info.width, itr.top_left_pose.y + itr.plane.info.height),
				   cv::Scalar( 126, 126, 126 ),
				   2 );

		ostringstream ostr;
		ostr << "result image" << itr.plane.info.id;
		imshow(ostr.str(), itr.img);
	}
	waitKey(100);

	extract_walls_from_candidate(plane_candidate_info);
	for(auto itr: plane_candidate_info)
	{
		ROS_INFO("[B%d] %lf %lf %lf %lf", itr.plane.info.id, itr.plane.pose.pi1, itr.plane.pose.pi2, itr.plane.pose.pi3, itr.plane.pose.pi4);
	}
	refine_scan_with_wall_information(plane_candidate_info, laserscan_ptr, refined_scan);
}

void camera_info_callback(const sensor_msgs::CameraInfoConstPtr& msg)
{
	ROS_INFO("camera_info_callback called");
	ROS_ASSERT(nullptr != msg);

	pinhole_camera_model.fromCameraInfo(msg);

	// once store camera information. No longer this callback doesn't need to be called.
	camera_info_sub.shutdown();
}

/*****************************************************************************************************************************************/
int main(int argc, char** argv)
{
	// Initialize ROS
    ros::init (argc, argv, "RGBDPlaneDetection");
    ros::NodeHandle nh;

	for(int i=0; i<10; ++i)
	{
		colors.push_back(Scalar(default_colors[i][0], default_colors[i][1], default_colors[i][2]));
	}

	image_transport::ImageTransport it(nh);
	pub = it.advertise ("/RGBDPlaneDetection/result_image", 1);

#if 1
#if 1
	typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::PointCloud2, sensor_msgs::LaserScan> SyncPolicy;
	message_filters::Subscriber<sensor_msgs::Image>       depth_sub(nh, "/xtion/depth_registered/hw_registered/image_rect", 10);
	message_filters::Subscriber<sensor_msgs::PointCloud2> pointcloud2_sub(nh, "/xtion/depth_registered/points", 3);
	message_filters::Subscriber<sensor_msgs::LaserScan>   laserscan_sub(nh, "/xtion/scan", 3);

	message_filters::Synchronizer<SyncPolicy> sync(SyncPolicy(10), depth_sub, pointcloud2_sub, laserscan_sub);
	sync.registerCallback(boost::bind(&callback, _1, _2, _3));
#else
	typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::PointCloud2> SyncPolicy;
	message_filters::Subscriber<sensor_msgs::Image>       depth_sub(nh, "/xtion/depth_registered/hw_registered/image_rect", 10);
	message_filters::Subscriber<sensor_msgs::PointCloud2> pointcloud2_sub(nh, "/xtion/depth_registered/points", 2);

	message_filters::Synchronizer<SyncPolicy> sync(SyncPolicy(10), depth_sub, pointcloud2_sub);
	sync.registerCallback(boost::bind(&callback, _1, _2));
#endif
	//image_transport::Subscriber sub1 = it.subscribe ("depth", 1, &PlaneDetection::readDepthImage, &plane_detection);
	camera_info_sub = nh.subscribe("/xtion/depth_registered/camera_info", 1, camera_info_callback);

	ros::spin();
#else
	image_transport::Subscriber sub = it.subscribe ("depth", 1, &PlaneDetection::readDepthImage, &plane_detection);
	image_transport::Publisher  pub = it.advertise ("/RGBDPlaneDetection/result_image", 1);
	sensor_msgs::ImagePtr msg;

	ros::Rate loop_rate(30);
  	while (nh.ok()) {
    	ros::spinOnce();
    	loop_rate.sleep();

		msg = plane_detection.runPlaneDetection();
		/*
		if (run_mrf)
		{
			msg = plane_detection.prepareForMRF();
			runMRFOptimization();
		}
		else
		{
			// no operation
		}
		*/
    	pub.publish(msg);
  	}
#endif
	return 0;
	
}