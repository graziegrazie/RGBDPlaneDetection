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

#include "plane_msgs/PlaneArray.h"
#include "rgbd_plane_detection/rgbd_plane_detection_utils.h"
#include <random>
#include "geometry_msgs/Point.h"

using namespace cv;

PlaneDetection plane_detection;
image_transport::Publisher pub;

#define DEBUG

using PlaneCandidateInfo = struct PlaneCandidateInfo_
{
	cv::Mat              imgs;
	geometry_msgs::Point top_left_points;
	plane_msgs::Plane    plane;
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
#define AREA_THRESHOLD (2000)

const unsigned char default_colors[10][3] =
{
	{255, 0, 0},
	{255, 255, 0},
	{100, 20, 50},
	{0, 30, 255},
	{10, 255, 60},
	{80, 10, 100},
	{0, 255, 200},
	{10, 60, 60},
	{255, 0, 128},
	{60, 128, 128}
};
vector<Scalar> colors;

Result separate_region(const cv::Mat& img, std::vector<cv::Mat>& out_imgs, plane_msgs::PlaneArray plane_array)
{
	Result result = Success;

	RNG rng(12345);
	ROS_ASSERT_MSG(nullptr == &img,      "null image is passed as img");
	ROS_ASSERT_MSG(nullptr == &out_imgs, "null image is passed as out_imgs");

	for(int i = 0; i < colors.size(); i++)
	{
		cv::Mat temp_img;
		out_imgs.push_back(temp_img);
		vector<vector <Point> > contours;

		inRange(img, colors[i], colors[i], out_imgs[i]);
		findContours(out_imgs[i], contours, RETR_TREE, CHAIN_APPROX_SIMPLE);

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

		plane_msgs::Plane temp_plane;
		temp_plane.header.frame_id = "/xtion_link";
		temp_plane.info.id         = i;
		temp_plane.info.width      = biggest_rect.width;
		temp_plane.info.height     = biggest_rect.height;
		
		plane_array.planes.push_back(temp_plane);

#ifdef DEBUG
		Scalar color = Scalar( rng.uniform(0, 256), rng.uniform(0,256), rng.uniform(0,256) );
		rectangle( out_imgs[i], biggest_rect.tl(), biggest_rect.br(), color, 2 );

		ostringstream ostr;
		ostr << "result image" << i;
		imshow(ostr.str(), out_imgs[i]);
	}
	waitKey(100);
#else
	}
#endif
	return result;
}

Result calc_plane_normal(const sensor_msgs::PointCloud2ConstPtr& pointcloud2_ptr, std::vector<cv::Mat>& imgs, std::vector<std::vector<double>> top_left_points, plane_msgs::PlaneArray& plane_array)
{
	Result result = Success;

	ROS_ASSERT_MSG(nullptr == pointcloud2_ptr,        "null pointcloud2 is entered");
	ROS_ASSERT_MSG(imgs.size() == plane_array.size(), "different size vectors are entered");

	std::mt19937 mt{ std::random_device{}() };
	std::uniform_real_distribution<double> dist(0, 1);

	for(unsigned char i = 0; i < imgs.size(); i++)
	{
		for(unsigned char j = 0; j < 100; j++)
		{
			double temp_x = dist(mt) * plane_array.planes[i].info.width + 0;
			
		}
	}

	return result;
}

void callback(const sensor_msgs::ImageConstPtr& depth_ptr, const sensor_msgs::PointCloud2ConstPtr& pointcloud2_ptr)
{
	std::vector<cv::Mat> out_imgs;
	plane_msgs::PlaneArray plane_array;
	std::vector<std::vector<double>> top_left_points;

	if(nullptr == depth_ptr )
	{
		ROS_WARN("depth_ptr is null");
	}
	else
	{
		plane_detection.readDepthImage(depth_ptr);
	}

	plane_detection.runPlaneDetection();
	separate_region(plane_detection.seg_img_, out_imgs, plane_array);
	calc_plane_normal(pointcloud2_ptr, out_imgs, top_left_points, plane_array);
}
/*****************************************************************************************************************************************/
int main(int argc, char** argv)
{
	// Initialize ROS
    ros::init (argc, argv, "RGBDPlaneDetection");
    ros::NodeHandle nh;
	image_transport::ImageTransport it(nh);

	printUsage();

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