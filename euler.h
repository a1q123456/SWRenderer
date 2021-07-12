#pragma once

#include "matrix.h"

template <typename DataType>
struct BasicEulerXYZ
{
    DataType x;
    DataType y;
    DataType z;

    operator BasicMat4x4<DataType>()
    {
        // Calculate rotation about x axis
        BasicMat4x4<DataType> R_x{
                { 1, 0, 0, 0},
                   {0, cos(x), -sin(x), 0},
                   { 0, sin(x), cos(x)), 0},
                   {0, 0, 0, 1}};

        // Calculate rotation about y axis
        BasicMat4x4<DataType> R_y{
            {cos(theta[1]), 0, sin(theta[1]), 0},
            {0, 1, 0, 0},
            {-sin(theta[1]), 0, cos(theta[1]), 0},
            {0, 0, 0, 1}};

        // Calculate rotation about z axis
        BasicMat4x4<DataType> R_z{
            {cos(theta[2]), -sin(theta[2]), 0, 0},
            {sin(theta[2]), cos(theta[2]), 0, 0},
            {0, 0, 1, 0},
            {0, 0, 0, 1}};

        // Combined rotation matrix
        BasicMat4x4<DataType> R = R_z * R_y * R_x;

        return R;
    }
};

using EulerXYZf = BasicEulerXYZ<float>
