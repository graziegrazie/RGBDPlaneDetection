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
#define NUM_OF_NORMAL_VECTOR					(10)
#define NUM_OF_POINTS_FOR_NORMAL_CALCULATION	(10)

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

			if(biggest_rect.area() < temp_rect.area())
			{
				biggest_rect       = temp_rect;
			}
			else
			{
				// no operation
			}
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

#ifdef DEBUG
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

	for(unsigned char j = 0; j < 100; j++) // avoid eternal loop
	{
		int px = dist(mt) * plane_width  + plane_top_left_x;
		int py = dist(mt) * plane_height + plane_top_left_y;

		Vec3b *src = plane_candidate_info.img.ptr<cv::Vec3b>(py); // Pointer to 1st element on Jth row.
		if(white_color[0] == src[px][0] && 
		   white_color[1] == src[px][1] &&
		   white_color[2] == src[px][2])
		{
			pose[0] = px;
			pose[1] = py;
			break;
		}
		else
		{
			// keep searching
		}
	}

	return result;
}

//Result get_3d_point_from_pointcloud2(const sensor_msgs::PointCloud2ConstPtr& pointcloud2_ptr, cv::Vec2i& pose, cv::Vec3f& point)
Result get_3d_point_from_pointcloud2(const sensor_msgs::PointCloud2ConstPtr& pointcloud2_ptr, Eigen::Vector2i& pose, Eigen::Vector3d& point)
{
	Result result = Succeeded;

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

Result check_3points_distance(cv::Vec2i& pose1, cv::Vec2i& pose2, cv::Vec2i& pose3)
{
	Result result = Succeeded;

	if(CHECK_RANGE_INCL_EQUAL(pose1[0] - pose2[0], 10) || CHECK_RANGE_INCL_EQUAL(pose1[1] - pose2[1], 10))
	{
		result = TooClose;
	}
	else if(CHECK_RANGE_INCL_EQUAL(pose2[0] - pose3[0], 10) || CHECK_RANGE_INCL_EQUAL(pose2[1] - pose3[1], 10))
	{
		result = TooClose;
	}
	else if(CHECK_RANGE_INCL_EQUAL(pose3[0] - pose1[0], 10) || CHECK_RANGE_INCL_EQUAL(pose3[1] - pose1[1], 10))
	{
		result = TooClose;
	}
	else // enough far among 3 points
	{
		result = Succeeded;
	}

	return result;
}

struct normal_dev_info
{
	Eigen::Vector3d normal;
	double square_dev;
};

/*
 * @brief This function estimate plane normal vector. The function resolves least square problem with SVD.
 */
Result calc_plane_normal(const sensor_msgs::PointCloud2ConstPtr& pointcloud2_ptr, std::vector<PlaneCandidateInfo>& plane_candidate_info)
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
		std::vector<Eigen::Vector3d> normalized_normals;
		double ave_x = 0, ave_y = 0, ave_z = 0;

		for(int l = 0; l < NUM_OF_NORMAL_VECTOR; l++)
		{
			Eigen::MatrixXd A = Eigen::MatrixXd::Zero(NUM_OF_POINTS_FOR_NORMAL_CALCULATION, 3);
			Eigen::VectorXd b = Eigen::VectorXd::Zero(NUM_OF_POINTS_FOR_NORMAL_CALCULATION);

			for(int j = 0; j < NUM_OF_POINTS_FOR_NORMAL_CALCULATION; j++)
			{
				Eigen::Vector2i pose;
				Eigen::Vector3d point;

				find_white_point(plane_candidate_info[plane_index], pose);
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
			Eigen::Vector3d normalized_normal = normal.normalized();
			normalized_normals.push_back( normalized_normal );

			// INFO: This line assumes that normalized normal must be detected at 10 times.
			ave_x += normalized_normal[0];
			ave_y += normalized_normal[1];
			ave_z += normalized_normal[2];
		}
		ave_x /= normalized_normals.size();
		ave_y /= normalized_normals.size();
		ave_z /= normalized_normals.size();

		std::vector<normal_dev_info> normal_dev_infos;
		for(auto itr: normalized_normals)
		{
			normal_dev_info temp;
			temp.normal     = itr;
			temp.square_dev = std::pow(itr[0] - ave_x, 2) + std::pow(itr[1] - ave_y, 2) + std::pow(itr[2] - ave_z, 2);
			normal_dev_infos.push_back(temp);
		}
		std::sort(normal_dev_infos.begin(), normal_dev_infos.end(), [](normal_dev_info x, normal_dev_info y){return x.square_dev > y.square_dev;});
		normal_dev_infos.erase(normal_dev_infos.begin(), normal_dev_infos.begin() + 4);

		double ave_xx = 0, ave_yy = 0, ave_zz = 0;
		for(auto itr: normal_dev_infos)
		{
			Eigen::Vector3d temp = itr.normal;
			ave_xx += temp[0];
			ave_yy += temp[1];
		}
		double yaw_angle = std::atan(ave_yy / ave_xx);
		plane_candidate_info[plane_index].pi3 = yaw_angle;

#ifdef DEBUG
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
	return result;
}

void callback(const sensor_msgs::ImageConstPtr& depth_ptr, const sensor_msgs::PointCloud2ConstPtr& pointcloud2_ptr)
{
	std::vector<cv::Mat> out_imgs;
	std::vector<PlaneCandidateInfo> plane_candidate_info;

	if(nullptr == depth_ptr )
	{
		ROS_WARN("depth_ptr is null");
	}
	else
	{
		plane_detection.readDepthImage(depth_ptr);
	}

	sensor_msgs::ImagePtr img_ptr = plane_detection.runPlaneDetection();
	pub.publish(*img_ptr);

	separate_region(plane_detection.seg_img_, plane_candidate_info);
	calc_plane_normal(pointcloud2_ptr, plane_candidate_info);
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