#include "swrenderer.h"
#include "vec.h"
#include "matrix.h"
#define _USE_MATH_DEFINES
#include <cmath>

SWRenderer::SWRenderer(HDC hdc, int w, int h) : hdc(hdc), width(w), height(h)
{
    memDc = ::CreateCompatibleDC(hdc);
}

void SWRenderer::CreateBuffer(int pixelFormat)
{
    constexpr auto nBuffer = sizeof(buffer) / sizeof(buffer[0]);
    constexpr auto align = (32 / 8);
    bufferLinesize = (width / align) + (align - (width % align));

    BITMAPINFO bm = {sizeof(BITMAPINFOHEADER),
                     width,
                     height, 1, 32,
                     BI_RGB, width * height * 4, 0, 0, 0, 0};

    for (int i = 0; i < nBuffer; i++)
    {
        bitmaps[i] = CreateDIBSection(memDc, &bm, DIB_RGB_COLORS, (void **)&buffer[i], 0, 0);
        canvas[i] = std::make_unique<Canvas>(memDc, bitmaps[i], width, height);
    }
}

void SWRenderer::UpdateBuffer(std::uint8_t *data, int srcWidth, int srcHeight, int linesize)
{
    int idx = !bufferIndex;
    auto h = std::min(height, srcHeight);
    auto w = std::min(width, srcWidth);
    for (auto i = 0; i < h; i++)
    {
        memcpy(buffer[idx] + (i * bufferLinesize), data + (i * linesize), w);
    }
    InterlockedExchange(&bufferIndex, idx);
}

void SWRenderer::SwapBuffer()
{
    InterlockedExchange(&bufferIndex, !bufferIndex);
}

HBITMAP SWRenderer::GetBitmap() const
{
    return bitmaps[bufferIndex];
}

float vertexList[] = {
    // front
    0,
    0,
    0,
    0,
    1,
    0,
    1,
    1,
    0,
    1,
    0,
    0,

    // left
    0,
    0,
    1,
    0,
    1,
    1,
    0,
    1,
    0,
    0,
    0,
    0,

    // back
    1,
    0,
    1,

    1,
    1,
    1,

    0,
    1,
    1,

    0,
    0,
    1,

    // right
    1,
    0,
    0,
    1,
    1,
    0,
    1,
    1,
    1,
    1,
    0,
    1,

    // bottom
    1,
    0,
    0,

    1,
    0,
    1,

    0,
    0,
    1,

    0,
    0,
    0,

    // top
    0,
    1,
    0,

    0,
    1,
    1,

    1,
    1,
    1,

    1,
    1,
    0,
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

float fpi()
{
    return atan(1.f) * 4;
}

Mat4x4f perspective(
    const float &angleOfView,
    const float &imageAspectRatio,
    const float &n, const float &f)
{
    float r, l, b, t;
    float scale = tan(angleOfView * 0.5 * fpi() / 180.f) * n;
    r = imageAspectRatio * scale, l = -r;
    t = scale, b = -t;

    Mat4x4f M;
    M.vals[0][0] = 2 * n / (r - l);
    M.vals[0][1] = 0;
    M.vals[0][2] = 0;
    M.vals[0][3] = 0;

    M.vals[1][0] = 0;
    M.vals[1][1] = 2 * n / (t - b);
    M.vals[1][2] = 0;
    M.vals[1][3] = 0;

    M.vals[2][0] = (r + l) / (r - l);
    M.vals[2][1] = (t + b) / (t - b);
    M.vals[2][2] = -(f + n) / (f - n);
    M.vals[2][3] = -1;

    M.vals[3][0] = 0;
    M.vals[3][1] = 0;
    M.vals[3][2] = -2 * f * n / (f - n);
    M.vals[3][3] = 0;

    return M;
}

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
    //y += timeElapsed * 0.5;
    z += timeElapsed * 0.3;

    Mat4x4f projectionMatrix = perspective(fov, aspectRatio, zNear, zFar);

    Mat4x4f screenMatrix{
        {1.f * width, 0, 0, 0},
        {0, 1.f * height, 0, 0},
        {0, 0, 1, 0},
        {0, 0, 0, 1}};

    Mat4x4f R_x{
        {1, 0, 0, 0},
        {0, cos(x), sin(x), 0},
        {0, -sin(x), cos(x), 0},
        {0, 0, 0, 1}};

    // Calculate rotation about y axis
    Mat4x4f R_y{
        {cos(y), 0, -sin(y), 0},
        {0, 1, 0, 0},
        {sin(y), 0, cos(y), 0},
        {0, 0, 0, 1}};

    // Calculate rotation about z axis
    Mat4x4f R_z{
        {cos(z), sin(z), 0, 0},
        {-sin(z), cos(z), 0, 0},
        {0, 0, 1, 0},
        {0, 0, 0, 1}};

    canvas[bufferIndex]->Clear(0);
    constexpr int nbIndices = sizeof(indexList) / sizeof(indexList[0]);
    for (int i = 0; i < nbIndices; i += 3)
    {
        Vector3f v0{vertexList[indexList[i] * 3], vertexList[indexList[i] * 3 + 1], vertexList[indexList[i] * 3 + 2]};
        Vector3f v1{vertexList[indexList[i + 1] * 3], vertexList[indexList[i + 1] * 3 + 1], vertexList[indexList[i + 1] * 3 + 2]};
        Vector3f v2{vertexList[indexList[i + 2] * 3], vertexList[indexList[i + 2] * 3 + 1], vertexList[indexList[i + 2] * 3 + 2]};

        Mat4x4f scaleMatrix{
            {1, 0, 0, 0},
            {0, 1, 0, 0},
            {0, 0, 1, 0},
            {0, 0, 0, 1},
        };

        Mat4x4f translateOriginMatrix{
            {1, 0, 0, 0},
            {0, 1, 0, 0},
            {0, 0, 1, 0},
            {-0.5, -0.5, -0.5, 1}};

        Mat4x4f translateBackMatrix{
            {1, 0, 0, 0},
            {0, 1, 0, 0},
            {0, 0, 1, 0},
            {0, 0, 0, 1}};

        Mat4x4f translateMatrix{
            {1, 0, 0, 0},
            {0, 1, 0, 0},
            {0, 0, 1, 0},
            {0, 0, 7, 1}};

        auto sv0 = v0 * translateOriginMatrix * scaleMatrix * R_x * R_y * R_z * translateBackMatrix * translateMatrix * projectionMatrix;
        auto sv1 = v1 * translateOriginMatrix * scaleMatrix * R_x * R_y * R_z * translateBackMatrix * translateMatrix * projectionMatrix;
        auto sv2 = v2 * translateOriginMatrix * scaleMatrix * R_x * R_y * R_z * translateBackMatrix * translateMatrix * projectionMatrix;

        sv0 = sv0 / -sv0.z;
        sv1 = sv1 / -sv1.z;
        sv2 = sv2 / -sv2.z;

        // backface culling
        auto t0 = sv1 - sv0;
        auto t1 = sv2 - sv1;

        if (t0.cross(t1).z > 0)
        {
            continue;
        }

        sv0 = Vector3f{(sv0.x + canvasWidth / 2.f) / canvasWidth, (sv0.y + canvasHeight / 2.f) / canvasHeight};
        sv1 = Vector3f{(sv1.x + canvasWidth / 2.f) / canvasWidth, (sv1.y + canvasHeight / 2.f) / canvasHeight};
        sv2 = Vector3f{(sv2.x + canvasWidth / 2.f) / canvasWidth, (sv2.y + canvasHeight / 2.f) / canvasHeight};

        // frustum culling
        if ((sv0.x < 0 || sv0.y < 0 || sv0.x > 1 || sv0.y > 1) &&
            (sv1.x < 0 || sv1.y < 0 || sv1.x > 1 || sv1.y > 1) &&
            (sv2.x < 0 || sv2.y < 0 || sv2.x > 1 || sv2.y > 1))
        {
            continue;
        }

        sv0 = Vector3f{sv0.x * width, sv0.y * height};
        sv1 = Vector3f{sv1.x * width, sv1.y * height};
        sv2 = Vector3f{sv2.x * width, sv2.y * height};

        canvas[bufferIndex]->LineTo(std::round(sv0.x), std::round(sv0.y), std::round(sv1.x), std::round(sv1.y), 0xFFFFFFFF);
        canvas[bufferIndex]->LineTo(std::round(sv1.x), std::round(sv1.y), std::round(sv2.x), std::round(sv2.y), 0xFFFFFFFF);
        canvas[bufferIndex]->LineTo(std::round(sv2.x), std::round(sv2.y), std::round(sv0.x), std::round(sv0.y), 0xFFFFFFFF);
    }
}
