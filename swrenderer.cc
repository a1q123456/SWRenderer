#include "swrenderer.h"
#include <glm/mat4x4.hpp>
#include <glm/vec4.hpp>
#include <glm/vec3.hpp>
#include <glm/common.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/euler_angles.hpp>
#define _USE_MATH_DEFINES
#include <cmath>
#include <vector>
#include <algorithm>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

SWRenderer::SWRenderer(HDC hdc, int w, int h) : hdc(hdc), width(w), height(h)
{
    memDc = ::CreateCompatibleDC(hdc);

    textureData.reset(stbi_load("D:\\Desktop\\1059544.jpg", &textureW, &textureH, &textureChannels, STBI_rgb_alpha), stbi_image_free);
}

void SWRenderer::ClearZBuffer()
{
    for (int i = 0; i < width * height; i++)
    {
        zBuffer[i] = std::numeric_limits<float>::infinity();
    }
}

void SWRenderer::CreateBuffer(int pixelFormat)
{
    constexpr auto nBuffer = sizeof(buffer) / sizeof(buffer[0]);

    BITMAPINFO bm = {sizeof(BITMAPINFOHEADER),
                     width,
                     height, 1, 32,
                     BI_RGB, width * height * 4, 0, 0, 0, 0};

    for (int i = 0; i < nBuffer; i++)
    {
        bitmaps[i] = CreateDIBSection(memDc, &bm, DIB_RGB_COLORS, (void **)&buffer[i], 0, 0);
        canvas[i] = std::make_unique<Canvas>(memDc, bitmaps[i], width, height);
    }
    zBuffer = new float[width * height];
}

void SWRenderer::SwapBuffer()
{
    InterlockedExchange(&bufferIndex, !bufferIndex);
}

HBITMAP SWRenderer::GetBitmap() const
{
    return bitmaps[bufferIndex];
}

// clang-format off
float vertexList[] = {
    // front
    0, 0, 0, 0, 0, 0,
    0, 1, 0, 0, 1, 0,
    1, 1, 0, 1, 1, 0,
    1, 0, 0, 1, 0, 0,

    // left
    0, 0, 1, 0, 0, 0,
    0, 1, 1, 0, 1, 0,
    0, 1, 0, 1, 1, 0,
    0, 0, 0, 1, 0, 0,

    // back
    1, 0, 1, 0, 0, 0,
    1, 1, 1, 0, 1, 0,
    0, 1, 1, 1, 1, 0,
    0, 0, 1, 1, 0, 0,

    // right
    1, 0, 0, 0, 0, 0,
    1, 1, 0, 0, 1, 0,
    1, 1, 1, 1, 1, 0,
    1, 0, 1, 1, 0, 0,

    // bottom
    1, 0, 0, 0, 0, 0,
    1, 0, 1, 0, 1, 0,
    0, 0, 1, 1, 1, 0,
    0, 0, 0, 1, 0, 0,

    // top
    0, 1, 0, 0, 0, 0,
    0, 1, 1, 0, 1, 0,
    1, 1, 1, 1, 1, 0,
    1, 1, 0, 1, 0, 0,
    };

// clang-format on
enum VertexElementType
{
    VET_POSITION,
    VET_UV,
    VET_NORMAL
};

enum DataType
{
    DT_FLOAT3 = sizeof(float) * 3,
    DT_FLOAT2 = sizeof(float) * 2,
    DT_FLOAT1 = sizeof(float) * 1
};

const int vertexLayout[]{
    VET_POSITION, DT_FLOAT3,
    VET_UV, DT_FLOAT3};

int indexList[] = {
    // front
    0, 1, 2,
    0, 2, 3,

    // left
    4, 5, 6,
    4, 6, 7,

    // back
    8, 9, 10,
    8, 10, 11,

    // right
    12, 13, 14,
    12, 14, 15,

    // bottom
    16, 17, 18,
    16, 18, 19,

    // top
    20, 21, 22,
    20, 22, 23};

struct Triangle
{
    glm::vec3 p0, p1, p2;
    glm::vec3 uv0, uv1, uv2;

    glm::vec3 min, max;

    float avg;

    Triangle() = default;
    Triangle(
        const glm::vec3 &a,
        const glm::vec3 &b,
        const glm::vec3 &c,
        const glm::vec3 &uv0,
        const glm::vec3 &uv1,
        const glm::vec3 &uv2) : p0(a),
                                p1(b),
                                p2(c),
                                uv0(uv0),
                                uv1(uv1),
                                uv2(uv2),
                                min(glm::vec3{
                                    std::min(std::min(a.x, b.x), c.x),
                                    std::min(std::min(a.y, b.y), c.y),
                                    std::min(std::min(a.z, b.z), c.z)}),
                                max(glm::vec3{
                                    std::max(std::max(a.x, b.x), c.x),
                                    std::max(std::max(a.y, b.y), c.y),
                                    std::max(std::max(a.z, b.z), c.z)})

    {
        auto avgTri = (a + b + c);
        avgTri /= 3.0;

        avg = (avgTri.x + avgTri.y + avgTri.z) / 3.0;
    }

    bool inRange(const glm::vec3 &pt) const noexcept
    {
        return pt.x >= min.x && pt.y >= min.y && pt.x <= max.x && pt.y <= max.y;
    }

    glm::vec3 barycentric(const glm::vec3 &pt)
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
};

void SWRenderer::Render(float timeElapsed)
{

    float fov = 50;
    float aspectRatio = (float)width / (float)height;
    float zNear = 0.01;
    float zFar = 1000;

    float canvasWidth = 1;
    float canvasHeight = 1;

    static float x = 0;
    static float y = 0;
    static float z = 0;

    x += timeElapsed * 0.5;
    //y += timeElapsed * 0.1;
    z += timeElapsed * 0.3;

    glm::vec4 omniLight{10, 10, 10, 1};
    glm::vec3 lightColor{1, 1, 1};
    float lightIntensity = 2.0;
    float lightFadeFactor = 1500;

    auto viewTransform = glm::lookAt(glm::vec3{0, 0, -7}, glm::vec3{0, 0, 0}, glm::vec3{0, 1, 0});
    auto projectionMatrix = glm::perspective(glm::radians(55.f), aspectRatio, zNear, zFar);
    auto scaleMatrix = glm::scale(glm::identity<glm::mat4>(), glm::vec3{1, 1, 1});
    auto translateOriginMatrix = glm::translate(glm::identity<glm::mat4>(), glm::vec3{-0.5, -0.5, -0.5});
    auto translateBackMatrix = glm::identity<glm::mat4>();
    auto translateMatrix = glm::translate(glm::identity<glm::mat4>(), glm::vec3{0, 0, 0});
    auto rotationMatrix = glm::eulerAngleXYZ(x, y, z);
    auto modelTransform = translateMatrix * translateBackMatrix * rotationMatrix * scaleMatrix * translateOriginMatrix;
    auto projWorld = projectionMatrix * glm::inverse(viewTransform) * modelTransform;

    constexpr int nbIndices = sizeof(indexList) / sizeof(indexList[0]);
    constexpr int nbLayoutElement = sizeof(vertexLayout) / sizeof(vertexLayout[0]);
    int szElement = 0;
    for (int i = 0; i < 4; i += 2)
    {
        szElement += vertexLayout[(i + 1)];
    }
    szElement /= sizeof(float);
    std::vector<Triangle> triangleList;

    for (int i = 0; i < nbIndices; i += 3)
    {
        glm::vec3 v0;
        glm::vec3 v1;
        glm::vec3 v2;

        glm::vec3 uv0;
        glm::vec3 uv1;
        glm::vec3 uv2;

        int offset = 0;
        for (int j = 0; j < nbLayoutElement; j += 2)
        {
            if (vertexLayout[j] == VET_POSITION)
            {
                v0 = glm::vec3{vertexList[indexList[i + 0] * szElement + offset], vertexList[indexList[i + 0] * szElement + 1 + offset], vertexList[indexList[i + 0] * szElement + 2 + offset]};
                v1 = glm::vec3{vertexList[indexList[i + 1] * szElement + offset], vertexList[indexList[i + 1] * szElement + 1 + offset], vertexList[indexList[i + 1] * szElement + 2 + offset]};
                v2 = glm::vec3{vertexList[indexList[i + 2] * szElement + offset], vertexList[indexList[i + 2] * szElement + 1 + offset], vertexList[indexList[i + 2] * szElement + 2 + offset]};
                offset += vertexLayout[j + 1];
            }
            else if (vertexLayout[j] == VET_UV)
            {
                uv0 = glm::vec3{vertexList[indexList[i + 0] * szElement + offset], vertexList[indexList[i + 0] * szElement + 1 + offset], vertexList[indexList[i + 0] * szElement + 2 + offset]};
                uv1 = glm::vec3{vertexList[indexList[i + 1] * szElement + offset], vertexList[indexList[i + 1] * szElement + 1 + offset], vertexList[indexList[i + 1] * szElement + 2 + offset]};
                uv2 = glm::vec3{vertexList[indexList[i + 2] * szElement + offset], vertexList[indexList[i + 2] * szElement + 1 + offset], vertexList[indexList[i + 2] * szElement + 2 + offset]};
            }
        }

        auto pv0 = projWorld * glm::vec4{v0, 1};
        auto pv1 = projWorld * glm::vec4{v1, 1};
        auto pv2 = projWorld * glm::vec4{v2, 1};

        pv0 = projectionMatrix * pv0;
        pv1 = projectionMatrix * pv1;
        pv2 = projectionMatrix * pv2;

        auto ov0 = pv0;
        auto ov1 = pv1;
        auto ov2 = pv2;

        auto sv0 = glm::vec3{pv0 / pv0.z};
        auto sv1 = glm::vec3{pv1 / pv1.z};
        auto sv2 = glm::vec3{pv2 / pv2.z};

        // backface culling
        auto t0 = sv1 - sv0;
        auto t1 = sv2 - sv1;

        if (backFaceCulling && glm::cross(t0, t1).z < 0)
        {
            continue;
        }

        // // frustum culling
        // if ()
        // {
        //     continue;
        // }

        sv0 = glm::vec3{(sv0.x + canvasWidth / 2.f) / canvasWidth, (sv0.y + canvasHeight / 2.f) / canvasHeight, ov0.z};
        sv1 = glm::vec3{(sv1.x + canvasWidth / 2.f) / canvasWidth, (sv1.y + canvasHeight / 2.f) / canvasHeight, ov1.z};
        sv2 = glm::vec3{(sv2.x + canvasWidth / 2.f) / canvasWidth, (sv2.y + canvasHeight / 2.f) / canvasHeight, ov2.z};

        glm::vec3 rv0{sv0.x * width, (1.0 - sv0.y) * height, -sv0.z};
        glm::vec3 rv1{sv1.x * width, (1.0 - sv1.y) * height, -sv1.z};
        glm::vec3 rv2{sv2.x * width, (1.0 - sv2.y) * height, -sv2.z};

        uv0 /= rv0.z;
        uv1 /= rv1.z;
        uv2 /= rv2.z;

        triangleList.emplace_back(rv0, rv1, rv2, uv0, uv1, uv2);

        // canvas[bufferIndex]->LineTo(std::round(rv0.x), std::round(rv0.y), std::round(rv1.x), std::round(rv1.y), 0xFFFFFFFF);
        // canvas[bufferIndex]->LineTo(std::round(rv1.x), std::round(rv1.y), std::round(rv2.x), std::round(rv2.y), 0xFFFFFFFF);
        // canvas[bufferIndex]->LineTo(std::round(rv2.x), std::round(rv2.y), std::round(rv0.x), std::round(rv0.y), 0xFFFFFFFF);

        // texturing
    }

    omniLight = projectionMatrix * omniLight;
    omniLight = glm::vec4{omniLight.x / omniLight.z, omniLight.y / omniLight.z, omniLight.z, 1};
    omniLight = glm::vec4{(omniLight.x + canvasWidth / 2.f) / canvasWidth, (omniLight.y + canvasHeight / 2.f) / canvasHeight, omniLight.z, 1.0};
    omniLight = glm::vec4{omniLight.x * width, (1.0 - omniLight.y) * height, -omniLight.z, 1.0};

    canvas[bufferIndex]->Clear(0);
    ClearZBuffer();

    // for cache friendly
    std::sort(std::begin(triangleList), std::end(triangleList), [](const Triangle &a, const Triangle &b)
              { return a.avg < b.avg; });
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            glm::vec3 pt{x, y, 0};

            for (auto &&tri : triangleList)
            {
                if (!tri.inRange(pt))
                {
                    continue;
                }
                auto weight = tri.barycentric(pt);
                if (weight.x < 0 || weight.y < 0 || weight.z < 0)
                {
                    continue;
                }

                auto fragx = 1.0 / (1.0 / tri.p0.x * weight.x + 1.0 / tri.p1.x * weight.y + 1.0 / tri.p2.x * weight.z);
                auto fragy = 1.0 / (1.0 / tri.p0.y * weight.x + 1.0 / tri.p1.y * weight.y + 1.0 / tri.p2.y * weight.z);
                auto fragz = 1.0 / (1.0 / tri.p0.z * weight.x + 1.0 / tri.p1.z * weight.y + 1.0 / tri.p2.z * weight.z);

                glm::vec4 fragPos{fragx, fragy, fragz, 1};

                if (depthTestEnabled && zBuffer[y * width + x] < fragz)
                {
                    continue;
                }
                if (depthWriteEnabled)
                {
                    zBuffer[y * width + x] = fragz;
                }
                auto uvw = tri.uv0 * weight.x + tri.uv1 * weight.y + tri.uv2 * weight.z;
                uvw *= fragz;
                uvw.x = 1.0 - uvw.x;
                auto imgX = std::clamp((int)std::round(uvw.x * textureW), 0, textureW - 1);
                auto imgY = std::clamp((int)std::round(uvw.y * textureH), 0, textureH - 1);
                auto lightDistance = glm::distance(omniLight, fragPos);
                auto lightValue = (1.0 - glm::clamp(lightDistance, 0.f, lightFadeFactor) / lightFadeFactor) * lightIntensity;

                buffer[bufferIndex][y * width * 4 + x * 4 + 0] = std::clamp<std::uint8_t>(textureData[imgY * textureW * 4 + imgX * 4 + 2] * lightValue, 0, 255);
                buffer[bufferIndex][y * width * 4 + x * 4 + 1] = std::clamp<std::uint8_t>(textureData[imgY * textureW * 4 + imgX * 4 + 1] * lightValue, 0, 255);
                buffer[bufferIndex][y * width * 4 + x * 4 + 2] = std::clamp<std::uint8_t>(textureData[imgY * textureW * 4 + imgX * 4 + 0] * lightValue, 0, 255);
            }
        }
    }
}
