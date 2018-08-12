#pragma once

#include <limits>
#define _USE_MATH_DEFINES
#include <cmath>

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

#define IS_NAN_FOR_POINT(point)                            (std::isnan(point[0]) || std::isnan(point[1]) || std::isnan(point[2]))

#define DEG2RAD(deg)    (deg / 180.0 * M_PI)
#define RAD2DEG(rad)    (rad / M_PI * 180.0)