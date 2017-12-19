#include "plane_detection.h"
#include <stdint.h>

PlaneDetection::PlaneDetection()
{
	cloud.vertices.resize(kDepthHeight * kDepthWidth);
	cloud.w = kDepthWidth;
	cloud.h = kDepthHeight;
}

PlaneDetection::~PlaneDetection()
{

}

// Temporarily don't need it since we set intrinsic parameters as constant values in the code.
//bool PlaneDetection::readIntrinsicParameterFile(string filename)
//{
//	ifstream readin(filename, ios::in);
//	if (readin.fail() || readin.eof())
//	{
//		cout << "WARNING: Cannot read intrinsics file " << filename << endl;
//		return false;
//	}
//	string target_str = "m_calibrationDepthIntrinsic";
//	string str_line, str, str_dummy;
//	double dummy;
//	bool read_success = false;
//	while (!readin.eof() && !readin.fail())
//	{
//		getline(readin, str_line);
//		if (readin.eof())
//			break;
//		istringstream iss(str_line);
//		iss >> str;
//		if (str == "m_depthWidth")
//			iss >> str_dummy >> width_;
//		else if (str == "m_depthHeight")
//			iss >> str_dummy >> height_;
//		else if (str == "m_calibrationDepthIntrinsic")
//		{
//			iss >> str_dummy >> fx_ >> dummy >> cx_ >> dummy >> dummy >> fy_ >> cy_;
//			read_success = true;
//			break;
//		}
//	}
//	readin.close();
//	if (read_success)
//	{
//		cloud.vertices.resize(height_ * width_);
//		cloud.w = width_;
//		cloud.h = height_;
//	}
//	return read_success;
//}

bool PlaneDetection::readColorImage(string filename)
{
	color_img_ = cv::imread(filename, CV_LOAD_IMAGE_COLOR);
	if (color_img_.empty() || color_img_.depth() != CV_8U)
	{
		cout << "ERROR: cannot read color image. No such a file, or the image format is not 8UC3" << endl;
		return false;
	}
	return true;
}

bool PlaneDetection::readDepthImage(string filename)
{
	cv::Mat depth_img = cv::imread(filename, CV_LOAD_IMAGE_ANYDEPTH);
	if (depth_img.empty() || depth_img.depth() != CV_16U)
	{
		cout << "WARNING: cannot read depth image. No such a file, or the image format is not 16UC1" << endl;
		return false;
	}
	int rows = depth_img.rows, cols = depth_img.cols;
	int vertex_idx = 0;
	for (int i = 0; i < rows; ++i)
	{
		for (int j = 0; j < cols; ++j)
		{
			double z = (double)(depth_img.at<unsigned short>(i, j)) / kScaleFactor;
			if (_isnan(z))
			{
				cloud.vertices[vertex_idx++] = VertexType(0, 0, z);
				continue;
			}
			double x = ((double)j - kCx) * z / kFx;
			double y = ((double)i - kCy) * z / kFy;
			cloud.vertices[vertex_idx++] = VertexType(x, y, z);
		}
	}
	return true;
}

bool PlaneDetection::runPlaneDetection()
{
	seg_img_ = cv::Mat(kDepthHeight, kDepthWidth, CV_8UC3);
	plane_filter.run(&cloud, &plane_vertices_, &seg_img_);
	plane_num_ = (int)plane_vertices_.size();

	// Here we set the plane index of a pixel which does NOT belong to any plane as #planes.
	// This is for using MRF optimization later.
	for (int row = 0; row < kDepthHeight; ++row)
	{
		for (int col = 0; col < kDepthWidth; ++col)
		{
			if (plane_filter.membershipImg.at<int>(row, col) < 0)
			{
				plane_filter.membershipImg.at<int>(row, col) = plane_num_;
			}
		}
	}
	return true;
}

void PlaneDetection::prepareForMRF()
{
	opt_seg_img_ = cv::Mat(kDepthHeight, kDepthWidth, CV_8UC3);
	opt_membership_img_ = cv::Mat(kDepthHeight, kDepthWidth, CV_32SC1);
	pixel_boundary_flags_.resize(kDepthWidth * kDepthHeight, false);
	pixel_grayval_.resize(kDepthWidth * kDepthHeight, 0);

	cv::Mat& mat_label = plane_filter.membershipImg;
	for (int row = 0; row < kDepthHeight; ++row)
	{
		for (int col = 0; col < kDepthWidth; ++col)
		{
			pixel_grayval_[row * kDepthWidth + col] = RGB2Gray(row, col);
			int label = mat_label.at<int>(row, col);
			if ((row - 1 >= 0 && mat_label.at<int>(row - 1, col) != label)
				|| (row + 1 < kDepthHeight && mat_label.at<int>(row + 1, col) != label)
				|| (col - 1 >= 0 && mat_label.at<int>(row, col - 1) != label)
				|| (col + 1 < kDepthWidth && mat_label.at<int>(row, col + 1) != label))
			{
				// Pixels in a fixed range near the boundary pixel are also regarded as boundary pixels
				for (int x = max(row - kNeighborRange, 0); x < min(kDepthHeight, row + kNeighborRange); ++x)
				{
					for (int y = max(col - kNeighborRange, 0); y < min(kDepthWidth, col + kNeighborRange); ++y)
					{
						// If a pixel is not on any plane, then it is not a boundary pixel.
						if (mat_label.at<int>(x, y) == plane_num_)
							continue;
						pixel_boundary_flags_[x * kDepthWidth + y] = true;
					}
				}
			}
		}
	}

	for (int pidx = 0; pidx < plane_num_; ++pidx)
	{
		int vidx = plane_vertices_[pidx][0];
		cv::Vec3b c = seg_img_.at<cv::Vec3b>(vidx / kDepthWidth, vidx % kDepthWidth);
		plane_colors_.push_back(c);
	}
	plane_colors_.push_back(cv::Vec3b(0,0,0)); // black for pixels not in any plane
}

// Note: input filename_prefix is like '/rgbd-image-folder-path/frame-XXXXXX'
void PlaneDetection::writeOutputFiles(string filename_prefix, bool run_mrf)
{
	string plane_filename = filename_prefix + "-plane";
	cv::imwrite(plane_filename + ".png", seg_img_);
	cv::imwrite(plane_filename + "-mapping.png", plane_filter.membershipImg);
	if (run_mrf)
	{
		string opt_plane_filename = filename_prefix + "-plane-opt";
		cv::imwrite(opt_plane_filename + ".png", opt_seg_img_);
		cv::imwrite(plane_filename + "-mapping-opt.png", opt_membership_img_);
	}
	computePlaneSumStats(run_mrf);
	writePlaneData(plane_filename + "-data.txt");
	if (run_mrf)
		writePlaneData(plane_filename + "-data-opt.txt", run_mrf);
}

void PlaneDetection::writePlaneData(string filename, bool run_mrf /* = false */)
{
	ofstream out(filename, ios::out);
	out << "#plane_index number_of_points_on_the_plane plane_color_in_png_image(1x3) plane_normal(1x3) plane_center(1x3) "
		<< "xx yy zz sx sy sz sxx syy szz sxy syz sxz" << endl;

	for (int pidx = 0; pidx < plane_num_; ++pidx)
	{
		out << pidx << " ";

		// Plane color in output image
		int vidx = plane_vertices_[pidx][0];
		cv::Vec3b c = seg_img_.at<cv::Vec3b>(vidx / kDepthWidth, vidx % kDepthWidth);
		out << int(c.val[2]) << " " << int(c.val[1]) << " "<< int(c.val[0]) << " "; // OpenCV uses BGR by default

		// Plane normal and center
		for (int i = 0; i < 3; ++i)
			out << plane_filter.extractedPlanes[pidx]->normal[i] << " ";
		for (int i = 0; i < 3; ++i)
			out << plane_filter.extractedPlanes[pidx]->center[i] << " ";

		// Sum of all points on the plane
		if (run_mrf)
		{
			out << opt_sum_stats_[pidx].sx << " " 
				<< opt_sum_stats_[pidx].sy << " " 
				<< opt_sum_stats_[pidx].sz << " " 
				<< opt_sum_stats_[pidx].sxx << " "
				<< opt_sum_stats_[pidx].syy << " "
				<< opt_sum_stats_[pidx].szz << " "
				<< opt_sum_stats_[pidx].sxy << " "
				<< opt_sum_stats_[pidx].syz << " "
				<< opt_sum_stats_[pidx].sxz << endl;
		}
		else
		{
			out << sum_stats_[pidx].sx << " " 
				<< sum_stats_[pidx].sy << " " 
				<< sum_stats_[pidx].sz << " " 
				<< sum_stats_[pidx].sxx << " "
				<< sum_stats_[pidx].syy << " "
				<< sum_stats_[pidx].szz << " "
				<< sum_stats_[pidx].sxy << " "
				<< sum_stats_[pidx].syz << " "
				<< sum_stats_[pidx].sxz << endl;
		}

		// NOTE: the plane-sum parameters computed from AHC code seems different from that computed from points belonging to planes shown above.
		// Seems there is a plane refinement step in AHC code so points belonging to each plane are slightly changed.
		//ahc::PlaneSeg::Stats& stat = plane_filter.extractedPlanes[pidx]->stats;
		//cout << stat.sx << " " << stat.sy << " " << stat.sz << " " << stat.sxx << " "<< stat.syy << " "<< stat.szz << " "<< stat.sxy << " "<< stat.syz << " "<< stat.sxz << endl;
	}
	out.close();
}

void PlaneDetection::computePlaneSumStats(bool run_mrf /* = false */)
{
	sum_stats_.resize(plane_num_);
	for (int pidx = 0; pidx < plane_num_; ++pidx)
	{
		for (int i = 0; i < plane_vertices_[pidx].size(); ++i)
		{
			int vidx = plane_vertices_[pidx][i];
			const VertexType& v = cloud.vertices[vidx];
			sum_stats_[pidx].sx += v[0];		 sum_stats_[pidx].sy += v[1];		  sum_stats_[pidx].sz += v[2];
			sum_stats_[pidx].sxx += v[0] * v[0]; sum_stats_[pidx].syy += v[1] * v[1]; sum_stats_[pidx].szz += v[2] * v[2];
			sum_stats_[pidx].sxy += v[0] * v[1]; sum_stats_[pidx].syz += v[1] * v[2]; sum_stats_[pidx].sxz += v[0] * v[2];
		}
	}
	if (run_mrf)
	{
		opt_sum_stats_.resize(plane_num_);
		for (int row = 0; row < kDepthHeight; ++row)
		{
			for (int col = 0; col < kDepthWidth; ++col)
			{
				int label = opt_membership_img_.at<int>(row, col); // plane label each pixel belongs to
				if (label != plane_num_) // pixel belongs to some plane
				{
					int vidx = row * kDepthWidth + col;
					const VertexType& v = cloud.vertices[vidx];
					opt_sum_stats_[label].sx += v[0];		  opt_sum_stats_[label].sy += v[1];		    opt_sum_stats_[label].sz += v[2];
					opt_sum_stats_[label].sxx += v[0] * v[0]; opt_sum_stats_[label].syy += v[1] * v[1]; opt_sum_stats_[label].szz += v[2] * v[2];
					opt_sum_stats_[label].sxy += v[0] * v[1]; opt_sum_stats_[label].syz += v[1] * v[2]; opt_sum_stats_[label].sxz += v[0] * v[2];
				}
			}
		}
	}
}

//void PlaneDetection::updatePlaneData()
//{
//	for (int pidx = 0; pidx < plane_num_; ++pidx)
//	{
//		double sx = 0, sy = 0, sz = 0, sxx = 0, syy = 0, szz = 0, sxy = 0, syz = 0, sxz = 0;
//		for (int i = 0; i < plane_vertices_[pidx].size(); ++i)
//		{
//			int vidx = plane_vertices_[pidx][i];
//			VertexType v = cloud.vertices[vidx];
//			sx += v[0];	sy += v[1];	sz += v[2];
//			sxx += v[0] * v[0];	syy += v[1] * v[1]; szz += v[2] * v[2];
//			sxy += v[0] * v[1]; syz += v[1] * v[2]; sxz += v[0] * v[2];
//		}
//		cout << sx << " " << sy << " " << sz << " " << sxx << " "<< syy << " "<< szz << " "<< sxy << " "<< syz << " "<< sxz << endl;
//		ahc::PlaneSeg::Stats& stat = plane_filter.extractedPlanes[pidx]->stats;
//		cout << stat.sx << " " << stat.sy << " " << stat.sz << " " << stat.sxx << " "<< stat.syy << " "<< stat.szz << " "<< stat.sxy << " "<< stat.syz << " "<< stat.sxz << endl;
//	}
//	vector<cv::Vec3b> plane_colors;
//	for (int pidx = 0; pidx < plane_num_; ++pidx)
//	{
//		int vidx = plane_vertices_[pidx][0];
//		cv::Vec3b c = seg_img_.at<cv::Vec3b>(vidx / kDepthWidth, vidx % kDepthWidth);
//		plane_colors.push_back(c);
//	}

//cv::Mat mat(kDepthHeight, kDepthWidth, CV_32SC1);
//for (int x = 0; x < kDepthHeight; ++x)
//{
//	for (int y = 0; y < kDepthWidth; ++y)
//	{
//		int pidx = plane_filter.membershipImg.at<int>(x, y);
//		//cv::Vec3b c = seg_img_.at<cv::Vec3b>(x, y);
//		//cout << pidx << " " << int(c.val[0]) << " " << int(c.val[1]) << " "<< int(c.val[2]) << endl;
//		mat.at<int>(x,y) = (pidx < 0) ? -1 : pidx;
//	}
//	//cout << endl;
//}
//cv::imwrite(plane_filename + "-mapping.png", mat);
//}