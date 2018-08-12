#pragma once

#include <limits>

enum Result
{
    Success = 0,
    Failure = 1,
    ConditionNotSatisfied = 2,
    TooClose = 3,
    NumOfError,
};

#define CHECK_RANGE_INCL_EQUAL(target, value)               (-value <= target && target <= value)
#define CHECK_RANGE_NOT_INCL_EQUAL(target, value)           (-value < target  && target <  value)

#define GET_VALUE_FROM_POINT_CLOUD2A(pointcloud_ptr, index) (((int)pointcloud_ptr->data[index + 0]) <<  0\
                                                           + ((int)pointcloud_ptr->data[index + 1]) <<  8\
                                                           + ((int)pointcloud_ptr->data[index + 2]) << 16\
                                                           + ((int)pointcloud_ptr->data[index + 3]) << 24)

#define GET_VALUE_FROM_POINT_CLOUD2B(pointcloud_ptr, index) (((int)pointcloud_ptr->data[index + 0]) << 24\
                                                           + ((int)pointcloud_ptr->data[index + 1]) << 16\
                                                           + ((int)pointcloud_ptr->data[index + 2]) <<  8\
                                                           + ((int)pointcloud_ptr->data[index + 3]) <<  0)

#define GET_VALUE_FROM_POINT_CLOUD2C(pointcloud_ptr, index) ((float)(pointcloud_ptr->data[index + 0] <<  0)\
                                                           + (float)(pointcloud_ptr->data[index + 1] <<  8)\
                                                           + (float)(pointcloud_ptr->data[index + 2] << 16)\
                                                           + (float)(pointcloud_ptr->data[index + 3] << 24))

#define GET_VALUE_FROM_POINT_CLOUD2D(pointcloud_ptr, index) ((float)(pointcloud_ptr->data[index + 0] << 24)\
                                                           + (float)(pointcloud_ptr->data[index + 1] << 16)\
                                                           + (float)(pointcloud_ptr->data[index + 2] <<  8)\
                                                           + (float)(pointcloud_ptr->data[index + 3] <<  0))

#define IS_NAN_FOR_POINT(point)                            (std::isnan(point[0]) || std::isnan(point[1]) || std::isnan(point[2]))