#pragma once

template <typename DataType>
struct BasicVector3
{
    DataType x;
    DataType y;
    DataType z;

    DataType length()
    {
        return std::sqrt(x * x + y * y + z * z);
    }

    DataType dot(const BasicVector3 &vec)
    {
        return x * vec.x + y * vec.y + z * vec.z;
    }

    BasicVector3 cross(const BasicVector3 &vec)
    {
        return BasicVector3{y * vec.z - z * vec.y, z * vec.x - x * vec.z, x * vec.y - y * vec.x};
    }

    BasicVector3 &operator+(const BasicVector3 &vec)
    {
        x += vec.x;
        y += vec.y;
        z += vec.z;
        return *this;
    }

    BasicVector3 &operator-(const BasicVector3 &vec)
    {
        x -= vec.x;
        y -= vec.y;
        z -= vec.z;
        return *this;
    }

    DataType operator*(DataType l)
    {
        x *= l;
        y *= l;
        z *= l;
        return *this;
    }

    DataType operator/(DataType l)
    {
        x /= l;
        y /= l;
        z /= l;
        return *this;
    }

    bool operator==(const BasicVector3 &vec)
    {
        return x == vec.x && y == vec.y && z == vec.z;
    }

    bool operator!=(const BasicVector3 &vec)
    {
        return x != vec.x && y != vec.y && z != vec.z;
    }
};

using Vector3f = BasicVector3<float>;
