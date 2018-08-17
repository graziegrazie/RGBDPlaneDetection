#pragma once

#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <Eigen/Dense>
#include "geometry_msgs/Pose2D.h"
#include "plane_msgs/Plane.h"

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
std::vector<cv::Scalar> colors;

using PlaneCandidateInfo = struct PlaneCandidateInfo_
{
	cv::Mat           img;
	cv::Point2d       top_left_pose;
	plane_msgs::Plane plane;
};

#define FILL_PLANE_POSE_WITH_NAN(pose)			{pose.pi1 = std::numeric_limits<double>::quiet_NaN();\
												 pose.pi2 = std::numeric_limits<double>::quiet_NaN();\
												 pose.pi3 = std::numeric_limits<double>::quiet_NaN();\
												 pose.pi4 = std::numeric_limits<double>::quiet_NaN();}

Eigen::Vector3d Z_AXIS(0.0, 0.0, 1.0);