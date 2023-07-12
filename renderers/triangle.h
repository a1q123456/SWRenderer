#pragma once
#include <cuda.h>
#include "utils.h"

struct Triangle
{
    glm::vec3 p0, p1, p2;
    glm::vec3 min, max;

    ProgramDataPack vsOutput[3];

    bool isValid = false;

    float avg = 0.0;

    __host__ __device__ Triangle() = default;
    __host__ __device__ Triangle(const glm::vec3& a, const glm::vec3& b, const glm::vec3& c,
                                 ProgramDataPack vsOutput[3])
        : p0(a), p1(b), p2(c), isValid(true),
          min(glm::vec3{std::min(std::min(a.x, b.x), c.x), std::min(std::min(a.y, b.y), c.y),
                        std::min(std::min(a.z, b.z), c.z)}),
          max(glm::vec3{std::max(std::max(a.x, b.x), c.x), std::max(std::max(a.y, b.y), c.y),
                        std::max(std::max(a.z, b.z), c.z)})

    {
        for (int i = 0; i < 3; i++)
        {
            this->vsOutput[i] = vsOutput[i];
        }
        auto avgTri = (a + b + c);
        avgTri /= 3.0;

        avg = (avgTri.x + avgTri.y + avgTri.z) / 3.0;
    }

    __host__ __device__ bool InRange(const glm::vec3& pt) const noexcept
    {
        return pt.x >= min.x && pt.y >= min.y && pt.x <= max.x && pt.y <= max.y;
    }

    __host__ __device__ glm::vec3 Barycentric(const glm::vec3& pt) const
    {
        glm::vec3 a{p0};
        glm::vec3 b{p1};
        glm::vec3 c{p2};

        glm::vec3 tmp{1, 1, 0};
        a *= tmp;
        b *= tmp;
        c *= tmp;

        glm::vec3 l0 = b - a;
        glm::vec3 l1 = c - b;
        glm::vec3 l2 = a - c;

        glm::vec3 pa = pt - a;
        glm::vec3 pb = pt - b;
        glm::vec3 pc = pt - c;

        auto ca = glm::cross(l0, pa);
        auto cb = glm::cross(l1, pb);
        auto cc = glm::cross(l2, pc);
        auto ct = glm::cross(l0, l1);
        auto ret = glm::vec3{cb.z, cc.z, ca.z} / ct.z;

        return ret;
    }

    __host__ __device__ bool PointInTriangle(const glm::vec3& pt) const
    {
        auto weight = Barycentric(pt);
        return weight.x >= 0 && weight.y >= 0 && weight.z >= 0;
    }

    __host__ __device__ bool Interect(const Ray& ray, glm::vec3& intersection) const
    {
        return false;
    }
};
