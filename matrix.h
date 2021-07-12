#pragma once
#include "vec.h"
#include <array>

template <typename DataType>
struct BasicMat4x4
{
    DataType vals[4][4];

    BasicMat4x4() {}

    BasicMat4x4(std::array<DataType, 4> r0, std::array<DataType, 4> r1, std::array<DataType, 4> r2, std::array<DataType, 4> r3)
    {
        memcpy(vals[0], r0.data(), r0.size() * sizeof(DataType));
        memcpy(vals[1], r1.data(), r1.size() * sizeof(DataType));
        memcpy(vals[2], r2.data(), r2.size() * sizeof(DataType));
        memcpy(vals[3], r3.data(), r3.size() * sizeof(DataType));
    }

    BasicMat4x4 transpose()
    {
        return BasicMat4x4{
            { vals[0][0], vals[1][0], vals[2][0], vals[3][0] },
            { vals[0][1], vals[1][1], vals[2][1], vals[3][1] },
            { vals[0][2], vals[1][2], vals[2][2], vals[3][2] },
            { vals[0][3], vals[1][3], vals[2][3], vals[3][3] },
        };
    }
};

template <typename DataType>
BasicVector3<DataType> operator*(const BasicVector3<DataType> &vec, const BasicMat4x4<DataType> &mat)
{
    return BasicVector3<DataType>{
        mat.vals[0][0] * vec.x + mat.vals[1][0] * vec.y + mat.vals[2][0] * vec.z + mat.vals[3][0] * 1,
        mat.vals[0][1] * vec.x + mat.vals[1][1] * vec.y + mat.vals[2][1] * vec.z + mat.vals[3][1] * 1,
        mat.vals[0][2] * vec.x + mat.vals[1][2] * vec.y + mat.vals[2][2] * vec.z + mat.vals[3][2] * 1,
    };
}

using Mat4x4f = BasicMat4x4<float>;
