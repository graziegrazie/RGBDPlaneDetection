#include "plane_detection.h"
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
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
#include <Eigen/Dense>
#include <algorithm>

using namespace cv;

PlaneDetection plane_detection;
image_transport::Publisher pub;

//#define DEBUG
#define DEBUG_VIEW
#define MAX_FIND_POINT_ITERATION				(100)
#define NUM_OF_NORMAL_VECTOR					(10)
#define NUM_OF_POINTS_FOR_NORMAL_CALCULATION	(20)
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
//-----------------------------------------------------------------
void cameraInfoCallback(sensor_msgs::CameraInfo color_info)
{
	ros::NodeHandle nh;
	color_info.header.frame_id ="plane_detection";
	ros::Publisher pub_info = nh.advertise<sensor_msgs::CameraInfo>("/camera/plane_detection/camera_info", 1);
	pub_info.publish(color_info);
}

void printUsage()
{
	cout << "Usage: rosrun rgbd_plane_detection rgbd_plane_detection color:=color_image_topic depth:=depth_image_topic" << endl;
	// cout << "-o: run MRF-optimization based plane refinement" << endl;
}

/*****************************************************************************************************************************************/
//Result separate_region(const cv::Mat& img, std::vector<cv::Mat>& out_imgs, plane_msgs::PlaneArray plane_array)
Result separate_region(const cv::Mat& img, std::vector<PlaneCandidateInfo>& plane_candidate_info)
{
	Result result = Succeeded;

	RNG rng(12345);
	ROS_ASSERT_MSG(nullptr == &img,      "null image is passed as img");
	ROS_ASSERT_MSG(nullptr == &out_imgs, "null image is passed as out_imgs");

	for(int i = 0; i < colors.size(); i++)
	{
		PlaneCandidateInfo temp_plane_candidate_info;
		vector<vector <Point> > contours;

		inRange(img, colors[i], colors[i], temp_plane_candidate_info.img);
		findContours(temp_plane_candidate_info.img, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);

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

		geometry_msgs::Pose2D temp_pose;
		temp_pose.x     = biggest_rect.x;
		temp_pose.y     = biggest_rect.y;
		temp_pose.theta = 0.0;
		temp_plane_candidate_info.top_left_pose = temp_pose;

		plane_candidate_info.push_back(temp_plane_candidate_info);

#ifdef DEBUG_VIEW
		Scalar color = Scalar( rng.uniform(0, 256), rng.uniform(0,256), rng.uniform(0,256) );
		rectangle( temp_plane_candidate_info.img, biggest_rect.tl(), biggest_rect.br(), color, 2 );

		ostringstream ostr;
		ostr << "result image" << i;
		imshow(ostr.str(), temp_plane_candidate_info.img);
	}
	waitKey(100);
#else
	}
#endif
	return result;
}

/*
 * @param[in]  plane_candidate_info
 * @param[out] pose
 */
Result find_white_point(PlaneCandidateInfo& plane_candidate_info, Eigen::Vector2i& pose)
{
	Result result = Succeeded;

	std::mt19937 mt{ std::random_device{}() };
	std::uniform_real_distribution<double> dist(0, 1);

	int plane_width  = plane_candidate_info.plane.info.width;
	int plane_height = plane_candidate_info.plane.info.height;

	int plane_top_left_x = plane_candidate_info.top_left_pose.x;
	int plane_top_left_y = plane_candidate_info.top_left_pose.y;

	for(unsigned char j = 0; j < MAX_FIND_POINT_ITERATION; j++) // avoid eternal loop
	{
		int px = dist(mt) * plane_width  + plane_top_left_x;
		int py = dist(mt) * plane_height + plane_top_left_y;

		Vec3b *src = plane_candidate_info.img.ptr<cv::Vec3b>(py); // Pointer to 1st element on Jth row.
		if(IS_WHITE_PIEXL(src[px]))
		{
			pose[0] = px;
			pose[1] = py;

			result = Succeeded;
			break;
		}
		else
		{
			// keep searching
			pose[0] = (plane_width  / 2.0) + plane_top_left_x;
			pose[1] = (plane_height / 2.0) + plane_top_left_y;

			result = UnexpectedPointEntered;
		}
	}

	return result;
}

Result get_3d_point_from_pointcloud2(const sensor_msgs::PointCloud2ConstPtr& pointcloud2_ptr, Eigen::Vector2i& pose, Eigen::Vector3d& point)
{
	Result result = Succeeded;

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

Result calc_plane_normal(const sensor_msgs::PointCloud2ConstPtr& pointcloud2_ptr, PlaneCandidateInfo& plane_candidate_info, Eigen::Vector3d& normalized_normal)
{
	Result result = Succeeded;

	Eigen::MatrixXd A = Eigen::MatrixXd::Zero(NUM_OF_POINTS_FOR_NORMAL_CALCULATION, 3);
	Eigen::VectorXd b = Eigen::VectorXd::Zero(NUM_OF_POINTS_FOR_NORMAL_CALCULATION);

	for(int j = 0; j < NUM_OF_POINTS_FOR_NORMAL_CALCULATION; j++)
	{
		Eigen::Vector2i pose;
		Eigen::Vector3d point;

		find_white_point(plane_candidate_info, pose);
		get_3d_point_from_pointcloud2(pointcloud2_ptr, pose, point);
		if(IS_NAN_FOR_POINT(point))
		{
			continue;
		}
		else
		{
			A(j, 0) = point[0];
			A(j, 1) = point[1];
			A(j, 2) = 1;

			b(j) = point[2];
		}
	}
	Eigen::Vector3d normal = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
	normalized_normal = normal.normalized();

	return result;
}

Result sort_normals_with_deviation(std::vector<Eigen::Vector3d>& normalized_normals, Eigen::Vector3d average_normal, std::vector<normal_dev_info>& normal_dev_infos)
{
	Result result = Succeeded;

	double ave_x = average_normal[0];
	double ave_y = average_normal[1];
	double ave_z = average_normal[2];

	for(auto itr: normalized_normals)
	{
		normal_dev_info temp;
		temp.normal     = itr;
		temp.square_dev = std::pow(itr[0] - ave_x, 2) + std::pow(itr[1] - ave_y, 2) + std::pow(itr[2] - ave_z, 2);
		normal_dev_infos.push_back(temp);
	}
	std::sort(normal_dev_infos.begin(), normal_dev_infos.end(), [](normal_dev_info x, normal_dev_info y){return x.square_dev > y.square_dev;});

	return result;
}

/*
 * @brief
 * @param[in]  pointcloud2_ptr
 * @param[in]  plane_candidate_info
 * @param[out] normalized_normals
 * @param[out] average_normal
 */
Result calc_normal_candidates_and_average_normal(const sensor_msgs::PointCloud2ConstPtr& pointcloud2_ptr, 
										 		 PlaneCandidateInfo& plane_candidate_info,
												 std::vector<Eigen::Vector3d>& normalized_normals,
												 Eigen::Vector3d& average_normal)
{
	Result result = Succeeded;

	for(int l = 0; l < NUM_OF_NORMAL_VECTOR; l++)
	{
		Eigen::Vector3d normalized_normal;

		calc_plane_normal(pointcloud2_ptr, plane_candidate_info, normalized_normal);
		if( IS_NAN_FOR_POINT(normalized_normal) )
		{
			continue;
		}
		else
		{
			normalized_normals.push_back( normalized_normal );

			// INFO: This line assumes that normalized normal must be detected at 10 times.
			average_normal[0] += normalized_normal[0];
			average_normal[1] += normalized_normal[1];
			average_normal[2] += normalized_normal[2];
		}
	}
	average_normal[0] /= normalized_normals.size();
	average_normal[1] /= normalized_normals.size();
	average_normal[2] /= normalized_normals.size();

	return result;
}

/*
 * @brief This function assumes that plane is orhogonal to floor, and some points on plane center column have segmented.
 */
Result calc_plane_3d_position_on_camera_coordinate(const sensor_msgs::PointCloud2ConstPtr& pointcloud2_ptr, PlaneCandidateInfo& plane_candidate_info)
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

Result calc_refined_average_normal(std::vector<Eigen::Vector3d>& normalized_normals, Eigen::Vector3d& average_normal, plane_msgs::PlanePose& plane_pose)
{
	Result result = Succeeded;

	std::vector<normal_dev_info> normal_dev_infos;
	sort_normals_with_deviation(normalized_normals, average_normal, normal_dev_infos);
	// TODO: Use Macro or other method not to use magic number
	normal_dev_infos.erase(normal_dev_infos.begin(), normal_dev_infos.begin() + 4);

	for(auto itr: normal_dev_infos)
	{
		plane_pose.pi1 += itr.normal[0];
		plane_pose.pi2 += itr.normal[1];
		plane_pose.pi3 += itr.normal[2];
	}
	plane_pose.pi1 /= normal_dev_infos.size();
	plane_pose.pi2 /= normal_dev_infos.size();
	plane_pose.pi3 /= normal_dev_infos.size();

	return result;
}

/*
 * @brief This function estimate plane normal vector. The function resolves least square problem with SVD.
 */
Result calc_plane_2d_coordinate_on_camera_coordinate(const sensor_msgs::PointCloud2ConstPtr& pointcloud2_ptr, std::vector<PlaneCandidateInfo>& plane_candidate_info)
{
	Result result = Succeeded;

	ROS_ASSERT_MSG(nullptr == pointcloud2_ptr,        "null pointcloud2 is entered");

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

		calc_normal_candidates_and_average_normal(pointcloud2_ptr, *plane_candidate_info_ptr, normalized_normals, average_normal);
		if( IS_NAN_FOR_POINT(average_normal) )
		{
			plane_candidate_info_ptr->plane.pose.pi1 = std::numeric_limits<double>::quiet_NaN();
			plane_candidate_info_ptr->plane.pose.pi2 = std::numeric_limits<double>::quiet_NaN();
			plane_candidate_info_ptr->plane.pose.pi3 = std::numeric_limits<double>::quiet_NaN();
			plane_candidate_info_ptr->plane.pose.pi4 = std::numeric_limits<double>::quiet_NaN();

			continue;
		}
		else
		{
			// no operation
		}
		calc_refined_average_normal(normalized_normals, average_normal, plane_candidate_info_ptr->plane.pose);
		calc_plane_3d_position_on_camera_coordinate(pointcloud2_ptr, *plane_candidate_info_ptr);
#ifdef DEBUG
		double plane_bearing = std::atan(plane_candidate_info_ptr->plane.pose.pi3 / plane_candidate_info_ptr->plane.pose.pi1);
		ROS_INFO("[%d]%+3.3lf[deg] %lf %lf", plane_index, RAD2DEG(plane_bearing), ave_xx, ave_zz);
		{
			double ax = 0, ay = 0, az = 0;
			for(auto itr_: normal_dev_infos)
			{
				Eigen::Vector3d itr = itr_.normal;

				ax = itr[0] / normal_dev_infos.size();
				ay = itr[1] / normal_dev_infos.size();
				az = itr[2] / normal_dev_infos.size();	
			}
			double roll  = std::atan(az / ay);
			double pitch = std::atan(az / ax);
			double yaw   = std::atan(ay / ax);
			ROS_INFO("[%d]%+3.3lf %+3.3lf %+3.3lf",plane_index, RAD2DEG(roll), RAD2DEG(pitch), RAD2DEG(yaw));
		}

		for(auto itr_: normal_dev_infos)
		{
			Eigen::Vector3d itr = itr_.normal;
			double roll  = std::atan(itr[2] / itr[1]);
			double pitch = std::atan(itr[2] / itr[0]);
			double yaw   = std::atan(itr[1] / itr[0]);
			ROS_INFO("%+3.3lf %+3.3lf %+3.3lf", RAD2DEG(roll), RAD2DEG(pitch), RAD2DEG(yaw));
		}
		std::cout << "" << std::endl;
#endif
	}
	std::cout << "" << std::endl;
	return result;
}

Result extract_walls_from_candidate(std::vector<PlaneCandidateInfo> plane_candidate_info, std::vector<PlaneCandidateInfo>& wall_info)
{
	Result result = Succeeded;

	for(auto itr: plane_candidate_info)
	{
		if( IS_NAN_FOR_4D_POINT(itr.plane.pose) )
		{
			continue;
		}
	}

	return result;
}

void callback(const sensor_msgs::ImageConstPtr& depth_ptr, const sensor_msgs::PointCloud2ConstPtr& pointcloud2_ptr)
{
	std::vector<cv::Mat> out_imgs;
	std::vector<PlaneCandidateInfo> plane_candidate_info, wall_info;

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
	extract_walls_from_candidate(plane_candidate_info, wall_info);
}
/*****************************************************************************************************************************************/
int main(int argc, char** argv)
{
	// Initialize ROS
    ros::init (argc, argv, "RGBDPlaneDetection");
    ros::NodeHandle nh;
	image_transport::ImageTransport it(nh);

	//printUsage();

	for(int i=0; i<10; ++i) {
		colors.push_back(Scalar(default_colors[i][0], default_colors[i][1], default_colors[i][2]));
	}

	//image_transport::Subscriber sub1 = it.subscribe ("color", 1, &PlaneDetection::readColorImage, &plane_detection);
	//image_transport::Subscriber sub2 = it.subscribe ("depth", 1, &PlaneDetection::readDepthImage, &plane_detection);


	//ros::Subscriber sub_info = nh.subscribe ("/camera/color/camera_info", 1, cameraInfoCallback);
	//ros::Publisher pub_info = nh.advertise<sensor_msgs::CameraInfo>("/camera/plane_detection/camera_info", 1);
	// sensor_msgs::CameraInfo info;

	//image_transport::Publisher pub = it.advertise ("/RGBDPlaneDetection/result_image", 1);
	//sensor_msgs::ImagePtr msg;
	pub = it.advertise ("/RGBDPlaneDetection/result_image", 1);

	//message_filters::Subscriber<sensor_msgs::Image> color_sub(nh, "color", 1);
	message_filters::Subscriber<sensor_msgs::Image>       depth_sub(nh, "depth", 1);
	message_filters::Subscriber<sensor_msgs::PointCloud2> pointcloud2_sub(nh, "pointcloud2", 1);

	typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::PointCloud2> MySyncPolicy;
 	 // ExactTime takes a queue size as its constructor argument, hence MySyncPolicy(10)
  	message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), depth_sub, pointcloud2_sub);
  	sync.registerCallback(boost::bind(&callback, _1, _2));
  		
	ros::Rate loop_rate(30);
  	while (nh.ok()) {
    	ros::spinOnce();
    	loop_rate.sleep();

/*
		msg = plane_detection.runPlaneDetection();
		if (run_mrf)
		{
			msg = plane_detection.prepareForMRF();
			runMRFOptimization();
		}
		else
		{
			// no operation
		}
    	pub.publish(msg);
*/
  	}
	
	return 0;
	
}