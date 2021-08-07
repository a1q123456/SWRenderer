#include "swrenderer.h"

#define _USE_MATH_DEFINES
#include <cmath>
#include <vector>
#include <algorithm>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#include "utils.h"
#include <Windows.h>
void SWRenderer::SetHWND(HWND hwnd)
{
    this->hwnd = hwnd;
}
void SWRenderer::MouseDown()
{
    if (hwnd != 0)
    {
        SetCapture(hwnd);
    }
    mouseCaptured = true;
}

void SWRenderer::MouseUp()
{
    if (hwnd != 0)
    {
        ReleaseCapture();
    }
    mouseCaptured = false;
    lastMouseX = -1;
    lastMouseY = -1;
}

void SWRenderer::MouseMove(int x, int y)
{
    if (!mouseCaptured)
    {
        return;
    }
    if (lastMouseX == -1)
    {
        lastMouseX = x;
    }
    if (lastMouseY == -1)
    {
        lastMouseY = y;
    }
    auto deltaX = x - lastMouseX;
    auto deltaY = y - lastMouseY;
    cubeRotation.y += (float)deltaX * 2.0 / (float)width;
    cubeRotation.x += (float)deltaY * 2.0 / (float)height;
    lastMouseX = x;
    lastMouseY = y;
}

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

constexpr size_t nbElements = 9;

// clang-format off
float vertexList[] = {
    // front
    0, 0, 0, 0, 0, 0, 0, 0, -1,
    0, 1, 0, 0, 1, 0, 0, 0, -1,
    1, 1, 0, 1, 1, 0, 0, 0, -1,
    1, 0, 0, 1, 0, 0, 0, 0, -1,

    // left
    0, 0, 1, 0, 0, 0, -1, 0, 0,
    0, 1, 1, 0, 1, 0, -1, 0, 0,
    0, 1, 0, 1, 1, 0, -1, 0, 0,
    0, 0, 0, 1, 0, 0, -1, 0, 0,

    // back
    1, 0, 1, 0, 0, 0, 0, 0, 1,
    1, 1, 1, 0, 1, 0, 0, 0, 1,
    0, 1, 1, 1, 1, 0, 0, 0, 1,
    0, 0, 1, 1, 0, 0, 0, 0, 1,

    // right
    1, 0, 0, 0, 0, 0, 1, 0, 0,
    1, 1, 0, 0, 1, 0, 1, 0, 0,
    1, 1, 1, 1, 1, 0, 1, 0, 0,
    1, 0, 1, 1, 0, 0, 1, 0, 0,

    // bottom
    1, 0, 0, 0, 0, 0, 0, -1, 0,
    1, 0, 1, 0, 1, 0, 0, -1, 0,
    0, 0, 1, 1, 1, 0, 0, -1, 0,
    0, 0, 0, 1, 0, 0, 0, -1, 0,

    // top
    0, 1, 0, 0, 0, 0, 0, 1, 0,
    0, 1, 1, 0, 1, 0, 0, 1, 0,
    1, 1, 1, 1, 1, 0, 0, 1, 0,
    1, 1, 0, 1, 0, 0, 0, 1, 0,
    };

// clang-format on
enum VertexElementType
{
    VET_POSITION,
    VET_UV,
    VET_NORMAL
};

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
    glm::vec3 normal0, normal1, normal2;
    glm::vec3 fragPos0, fragPos1, fragPos2;

    glm::vec3 min, max;

    float avg;

    Triangle() = default;
    Triangle(
        const glm::vec3 &a,
        const glm::vec3 &b,
        const glm::vec3 &c,
        const glm::vec3 &uv0,
        const glm::vec3 &uv1,
        const glm::vec3 &uv2,
        const glm::vec3 &normal0,
        const glm::vec3 &normal1,
        const glm::vec3 &normal2,
        const glm::vec3 &fragPos0,
        const glm::vec3 &fragPos1,
        const glm::vec3 &fragPos2) : p0(a),
                                     p1(b),
                                     p2(c),
                                     uv0(uv0),
                                     uv1(uv1),
                                     uv2(uv2),
                                     normal0(normal0),
                                     normal1(normal1),
                                     normal2(normal2),
                                     fragPos0(fragPos0),
                                     fragPos1(fragPos1),
                                     fragPos2(fragPos2),
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
    stats.emplace_back(timeElapsed);
    if (stats.size() > 5)
    {
        stats.pop_front();
    }
    auto avgTime = std::accumulate(std::begin(stats), std::end(stats), 0.f) / 5.0;

    float fov = 50;
    float aspectRatio = (float)width / (float)height;
    float zNear = 0.01;
    float zFar = 1000;

    float canvasWidth = 1;
    float canvasHeight = 1;

    glm::vec4 omniLight{0, 0, -10, 1};
    glm::vec3 lightColor{1, 1, 1};
    float lightIntensity = 1;
    float lightFadeFactor = 18;

    auto viewTransform = glm::inverse(glm::lookAt(glm::vec3{0, 0, -7}, glm::vec3{0, 0, 0}, glm::vec3{0, 1, 0}));
    auto projectionMatrix = glm::perspective(glm::radians(55.f), aspectRatio, zNear, zFar);
    auto scaleMatrix = glm::scale(glm::identity<glm::mat4>(), glm::vec3{1, 1, 1});
    auto translateOriginMatrix = glm::translate(glm::identity<glm::mat4>(), glm::vec3{-0.5, -0.5, -0.5});
    auto translateBackMatrix = glm::identity<glm::mat4>();
    auto translateMatrix = glm::translate(glm::identity<glm::mat4>(), glm::vec3{0, 0, 0});
    auto rotationMatrix = glm::eulerAngleXYZ(cubeRotation.x, cubeRotation.y, cubeRotation.z);
    auto modelTransform = translateMatrix * translateBackMatrix * rotationMatrix * scaleMatrix * translateOriginMatrix;
    auto projVP = projectionMatrix * viewTransform;
    constexpr int nbIndices = count_of(indexList);

    std::vector<Triangle> triangleList;
    triangleList.reserve(nbIndices / 3);

    canvas[bufferIndex]->Clear(0);
    ClearZBuffer();

    for (int i = 0; i < nbIndices; i += 3)
    {
        glm::vec3 v0;
        glm::vec3 v1;
        glm::vec3 v2;

        glm::vec3 uv0;
        glm::vec3 uv1;
        glm::vec3 uv2;

        glm::vec3 normal0;
        glm::vec3 normal1;
        glm::vec3 normal2;

        v0 = glm::vec3{vertexList[indexList[i + 0] * nbElements + 0], vertexList[indexList[i + 0] * nbElements + 1], vertexList[indexList[i + 0] * nbElements + 2]};
        v1 = glm::vec3{vertexList[indexList[i + 1] * nbElements + 0], vertexList[indexList[i + 1] * nbElements + 1], vertexList[indexList[i + 1] * nbElements + 2]};
        v2 = glm::vec3{vertexList[indexList[i + 2] * nbElements + 0], vertexList[indexList[i + 2] * nbElements + 1], vertexList[indexList[i + 2] * nbElements + 2]};

        uv0 = glm::vec3{vertexList[indexList[i + 0] * nbElements + 3], vertexList[indexList[i + 0] * nbElements + 4], vertexList[indexList[i + 0] * nbElements + 5]};
        uv1 = glm::vec3{vertexList[indexList[i + 1] * nbElements + 3], vertexList[indexList[i + 1] * nbElements + 4], vertexList[indexList[i + 1] * nbElements + 5]};
        uv2 = glm::vec3{vertexList[indexList[i + 2] * nbElements + 3], vertexList[indexList[i + 2] * nbElements + 4], vertexList[indexList[i + 2] * nbElements + 5]};

        normal0 = glm::vec3{vertexList[indexList[i + 0] * nbElements + 6], vertexList[indexList[i + 0] * nbElements + 7], vertexList[indexList[i + 0] * nbElements + 8]};
        normal1 = glm::vec3{vertexList[indexList[i + 1] * nbElements + 6], vertexList[indexList[i + 1] * nbElements + 7], vertexList[indexList[i + 1] * nbElements + 8]};
        normal2 = glm::vec3{vertexList[indexList[i + 2] * nbElements + 6], vertexList[indexList[i + 2] * nbElements + 7], vertexList[indexList[i + 2] * nbElements + 8]};

        auto mv0 = modelTransform * glm::vec4{v0, 1};
        auto mv1 = modelTransform * glm::vec4{v1, 1};
        auto mv2 = modelTransform * glm::vec4{v2, 1};

        auto mnormal0 = glm::transpose(glm::inverse(modelTransform)) * glm::vec4{normal0, 1};
        auto mnormal1 = glm::transpose(glm::inverse(modelTransform)) * glm::vec4{normal1, 1};
        auto mnormal2 = glm::transpose(glm::inverse(modelTransform)) * glm::vec4{normal2, 1};

        auto pv0 = projVP * mv0;
        auto pv1 = projVP * mv1;
        auto pv2 = projVP * mv2;

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

        sv0 = glm::vec3{(sv0.x + canvasWidth / 2.f) / canvasWidth, (sv0.y + canvasHeight / 2.f) / canvasHeight, pv0.z};
        sv1 = glm::vec3{(sv1.x + canvasWidth / 2.f) / canvasWidth, (sv1.y + canvasHeight / 2.f) / canvasHeight, pv1.z};
        sv2 = glm::vec3{(sv2.x + canvasWidth / 2.f) / canvasWidth, (sv2.y + canvasHeight / 2.f) / canvasHeight, pv2.z};

        glm::vec3 rv0{sv0.x * width, (1.0 - sv0.y) * height, -sv0.z};
        glm::vec3 rv1{sv1.x * width, (1.0 - sv1.y) * height, -sv1.z};
        glm::vec3 rv2{sv2.x * width, (1.0 - sv2.y) * height, -sv2.z};

        uv0 /= rv0.z;
        uv1 /= rv1.z;
        uv2 /= rv2.z;

        triangleList.emplace_back(rv0, rv1, rv2, uv0, uv1, uv2, mnormal0, mnormal1, mnormal2, mv0, mv1, mv2);

        // canvas[bufferIndex]->LineTo(std::round(rv0.x), std::round(rv0.y), std::round(rv1.x), std::round(rv1.y), 0xFFFFFFFF);
        // canvas[bufferIndex]->LineTo(std::round(rv1.x), std::round(rv1.y), std::round(rv2.x), std::round(rv2.y), 0xFFFFFFFF);
        // canvas[bufferIndex]->LineTo(std::round(rv2.x), std::round(rv2.y), std::round(rv0.x), std::round(rv0.y), 0xFFFFFFFF);

        // texturing
    }

    canvas[bufferIndex]->AddText(0, 0, 12, std::to_string(1.0 / avgTime), 0xFFFFFFFF);

    // for cache friendly
    std::sort(std::begin(triangleList), std::end(triangleList), [](const Triangle &a, const Triangle &b)
              { return a.avg < b.avg; });
    for (int y = 0; y < height; y++)
    {
#pragma omp parallel for
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

                auto fragx = (tri.fragPos0.x * weight.x + tri.fragPos1.x * weight.y + tri.fragPos2.x * weight.z);
                auto fragy = (tri.fragPos0.y * weight.x + tri.fragPos1.y * weight.y + tri.fragPos2.y * weight.z);
                auto fragz = (tri.fragPos0.z * weight.x + tri.fragPos1.z * weight.y + tri.fragPos2.z * weight.z);

                auto normalx = (tri.normal0.x * weight.x + tri.normal1.x * weight.y + tri.normal2.x * weight.z);
                auto normaly = (tri.normal0.y * weight.x + tri.normal1.y * weight.y + tri.normal2.y * weight.z);
                auto normalz = (tri.normal0.z * weight.x + tri.normal1.z * weight.y + tri.normal2.z * weight.z);

                auto depth = 1.0 / (1.0 / tri.p0.z * weight.x + 1.0 / tri.p1.z * weight.y + 1.0 / tri.p2.z * weight.z);

                glm::vec4 fragPos{fragx, fragy, fragz, 1};
                glm::vec4 normal{normalx, normaly, normalz, 1};

                if (depthTestEnabled && zBuffer[y * width + x] < depth)
                {
                    continue;
                }
                if (depthWriteEnabled)
                {
                    zBuffer[y * width + x] = depth;
                }
                auto uvw = tri.uv0 * weight.x + tri.uv1 * weight.y + tri.uv2 * weight.z;
                uvw *= depth;
                uvw.x = 1.0 - uvw.x;
                auto imgX = std::clamp((int)std::round(uvw.x * textureW), 0, textureW - 1);
                auto imgY = std::clamp((int)std::round(uvw.y * textureH), 0, textureH - 1);
                auto lightDistance = glm::distance(omniLight, fragPos);
                auto lightValue = (1.0 - glm::clamp(lightDistance, 0.f, lightFadeFactor) / lightFadeFactor) * lightIntensity;

                auto diffuse = std::clamp(glm::dot(glm::normalize(omniLight - fragPos), glm::normalize(normal)), 0.f, 1.f);
                lightValue *= diffuse;

                buffer[bufferIndex][y * width * 4 + x * 4 + 0] = std::clamp<std::uint8_t>(textureData[imgY * textureW * 4 + imgX * 4 + 2] * lightValue, 0, 255);
                buffer[bufferIndex][y * width * 4 + x * 4 + 1] = std::clamp<std::uint8_t>(textureData[imgY * textureW * 4 + imgX * 4 + 1] * lightValue, 0, 255);
                buffer[bufferIndex][y * width * 4 + x * 4 + 2] = std::clamp<std::uint8_t>(textureData[imgY * textureW * 4 + imgX * 4 + 0] * lightValue, 0, 255);
            }
        }
    }
}
