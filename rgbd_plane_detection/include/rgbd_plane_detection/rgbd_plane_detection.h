#pragma once

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include "geometry_msgs/Pose2D.h"
#include "plane_msgs/Plane.h"
#include <vector>

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

cv::Vec3b white_color(255, 255, 255);

using PlaneCandidateInfo = struct PlaneCandidateInfo_
{
	cv::Mat               img;
	geometry_msgs::Pose2D top_left_pose;
	plane_msgs::Plane     plane;
};