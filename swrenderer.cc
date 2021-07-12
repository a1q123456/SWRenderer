#include "swrenderer.h"
#include "vec.h"
#include "matrix.h"
#define _USE_MATH_DEFINES
#include <math.h>

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
    0,
    0,
    0,
    0,
    0,
    1,
    1,
    0,
    1,
    1,
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

void SWRenderer::Render(float timeElapsed)
{
    float n = 0.01;
    float f = 1000;
    float fov = 75;
    float aspectRatio = width / height;
    float zNear = 0.01;
    float zFar = 1000;
    float zRange = zNear - zFar;
    float tanHalfFOV = tanf(fov / 2.0 / 180.f * M_PI);

    static float x = 0;
    static float y = 0;
    static float z = 0;

    x += timeElapsed * 1;
    y += timeElapsed * 1;
    z += timeElapsed * 1;

    Mat4x4f projectionMatrix{
        {1.0f / (tanHalfFOV * aspectRatio), 0,                 0,                             0},
        {0,                                 1.0f / tanHalfFOV, 0,                             0},
        {0,                                 0,                 (-zNear - zFar) / zRange,      -1.f},
        {0,                                 0,                 2.0f * zFar * zNear / zRange,  0}};

    Mat4x4f screenMatrix{
        {1.f * width, 0, 0, 0},
        {0, 1.f * height, 0, -1.f * height},
        {0, 0, 1, 0},
        {0, 0, 0, 1}};

    Mat4x4f R_x{
        {1, 0, 0, 0},
        {0, cos(x), -sin(x), 0},
        {0, sin(x), cos(x), 0},
        {0, 0, 0, 1}};

    // Calculate rotation about y axis
    Mat4x4f R_y{
        {cos(y), 0, sin(y), 0},
        {0, 1, 0, 0},
        {-sin(y), 0, cos(y), 0},
        {0, 0, 0, 1}};

    // Calculate rotation about z axis
    Mat4x4f R_z{
        {cos(z), -sin(z), 0, 0},
        {sin(z), cos(z), 0, 0},
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
            {0.5, 0, 0, 0},
            {0, 0.5, 0, 0},
            {0, 0, 0.5, 0},
            {0, 0, 0, 1},
        };

        Mat4x4f translateMatrix{
            {1, 0, 0, 0},
            {0, 1, 0, 0},
            {0, 0, 1, 0},
            {-0.5, -0.5, -0.5, 1}};

        Mat4x4f translate2Matrix{
            {1, 0, 0, 0},
            {0, 1, 0, 0},
            {0, 0, 1, 0},
            {0.5, 0.5, 6, 1}};

        auto sv0 = v0 * translateMatrix * scaleMatrix * R_x * R_y * R_z * translate2Matrix * projectionMatrix * screenMatrix;
        auto sv1 = v1 * translateMatrix * scaleMatrix * R_x * R_y * R_z * translate2Matrix * projectionMatrix * screenMatrix;
        auto sv2 = v2 * translateMatrix * scaleMatrix * R_x * R_y * R_z * translate2Matrix * projectionMatrix * screenMatrix;

        canvas[bufferIndex]->LineTo(sv0.x, sv0.y, sv1.x, sv1.y, 0xFFFFFFFF);
        canvas[bufferIndex]->LineTo(sv1.x, sv1.y, sv2.x, sv2.y, 0xFFFFFFFF);
        canvas[bufferIndex]->LineTo(sv2.x, sv2.y, sv0.x, sv0.y, 0xFFFFFFFF);
    }
}
