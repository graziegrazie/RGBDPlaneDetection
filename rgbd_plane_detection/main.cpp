#include "plane_detection.h"
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <image_transport/image_transport.h>

PlaneDetection plane_detection;

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


void printUsage()
{
	cout << "Usage: rosrun RGBDPlaneDetection color:=color_image_topic depth:=depth_image_topic" << endl;
	// cout << "-o: run MRF-optimization based plane refinement" << endl;
}

int main(int argc, char** argv)
{
	// Initialize ROS
    ros::init (argc, argv, "RGBDPlaneDetection");
    ros::NodeHandle nh;
	image_transport::ImageTransport it(nh);
	bool run_mrf = false;

	// if (argc != 4 && argc != 5)
	// {
	// 	printUsage();
	// 	return -1;
	// }
	
	image_transport::Subscriber sub1 = it.subscribe ("/camera/color/image_rect_color", 1000, &PlaneDetection::readColorImage, &plane_detection);
	image_transport::Subscriber sub2 = it.subscribe ("/camera/aligned_depth_to_color/image_raw", 1000, &PlaneDetection::readDepthImage, &plane_detection);

	// plane_detection.runPlaneDetection();

	image_transport::Publisher pub = it.advertise ("camera/plane_detection", 1);
	sensor_msgs::ImagePtr msg;

	ros::Rate loop_rate(5);
  	while (nh.ok()) {
		msg = plane_detection.runPlaneDetection();
    	pub.publish(msg);
    	ros::spinOnce();
    	loop_rate.sleep();
  	}
	
	return 0;
	
}