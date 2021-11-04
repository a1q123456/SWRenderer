#include "scene_controller.h"
#include "icanvas.h"

// clang-format off
constexpr float vertexList[] = {
    // back
    1, 0, 0, 0, 1, 0, 0, 0, -1,
    1, 1, 0, 0, 0, 0, 0, 0, -1,
    0, 1, 0, 1, 0, 0, 0, 0, -1,
    0, 0, 0, 1, 1, 0, 0, 0, -1,

    // left
    0, 0, 0, 0, 1, 0, -1, 0, 0,
    0, 1, 0, 0, 0, 0, -1, 0, 0,
    0, 1, 1, 1, 0, 0, -1, 0, 0,
    0, 0, 1, 1, 1, 0, -1, 0, 0,

    // front
    0, 0, 1, 0, 1, 0, 0, 0, 1,
    0, 1, 1, 0, 0, 0, 0, 0, 1,
    1, 1, 1, 1, 0, 0, 0, 0, 1,
    1, 0, 1, 1, 1, 0, 0, 0, 1,

    // right
    1, 0, 1, 0, 1, 0, 1, 0, 0,
    1, 1, 1, 0, 0, 0, 1, 0, 0,
    1, 1, 0, 1, 0, 0, 1, 0, 0,
    1, 0, 0, 1, 1, 0, 1, 0, 0,

    // bottom
    0, 0, 0, 0, 1, 0, 0, -1, 0,
    0, 0, 1, 0, 0, 0, 0, -1, 0,
    1, 0, 1, 1, 0, 0, 0, -1, 0,
    1, 0, 0, 1, 1, 0, 0, -1, 0,

    // top
    0, 1, 1, 0, 1, 0, 0, 1, 0,
    0, 1, 0, 0, 0, 0, 0, 1, 0,
    1, 1, 0, 1, 0, 0, 0, 1, 0,
    1, 1, 1, 1, 1, 0, 0, 1, 0,
    };

// clang-format on

constexpr int indexList[] = {
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

template<CanvasDrawable Canvas>
template<CanvasDrawable T>
SceneController<Canvas>::SceneController(T&& canvas) : width(canvas.Width()), height(canvas.Height()), renderer(std::move(canvas))
{
    float fov = 50;
    float aspectRatio = (float)width / (float)height;
    float zNear = 0.01;
    float zFar = 1000;

    projectionMatrix = glm::perspective(glm::radians(55.f), aspectRatio, zNear, zFar);
    renderer.ProjectionMatrix(projectionMatrix);

    textureData.reset(stbi_load("sample_640x426.jpeg", &textureW, &textureH, &textureChannels, STBI_rgb_alpha), stbi_image_free);

    modelData.SetIndexList({std::cbegin(indexList), std::cend(indexList)});
    modelData.SetVertexList({std::cbegin(vertexList), std::cend(vertexList)});

    modelData.SetVertexDescriptor({
        {VertexAttributes::Position, VertexAttributeTypes::Vec3},
        {VertexAttributes::TextureCoordinate, VertexAttributeTypes::Vec3},
        {VertexAttributes::Normal, VertexAttributeTypes::Vec3},
    });

    pixelProgram.UseLights({&ambientLight, &pointLight});
    pixelProgram.SetDiffuseMap(textureData.get(), textureW, textureH);
    programCtx = renderer.LinkProgram(vertexProgram, pixelProgram);
}

template<CanvasDrawable Canvas>
void SceneController<Canvas>::Render(float timeElapsed)
{
    glm::vec3 cameraPos{0, 0, cameraDistance};
    cameraPos = glm::eulerAngleYXZ(cubeRotation.x, cubeRotation.y, cubeRotation.z) * glm::vec4{cameraPos, 1};

    auto viewTransform = (glm::lookAt(cameraPos, glm::vec3{0, 0, 0}, glm::vec3{0, 1, 0}));

    auto scaleMatrix = glm::scale(glm::identity<glm::mat4>(), glm::vec3{1, 1, 1});
    auto translateOriginMatrix = glm::translate(glm::identity<glm::mat4>(), glm::vec3{-0.5, -0.5, -0.5});
    auto translateBackMatrix = glm::identity<glm::mat4>();
    auto translateMatrix = glm::translate(glm::identity<glm::mat4>(), glm::vec3{0, 0, 0});
    auto rotationMatrix = glm::eulerAngleXYZ(0.f, 0.f, 0.f);
    auto modelTransform = translateMatrix * translateBackMatrix * rotationMatrix * scaleMatrix * translateOriginMatrix;

    renderer.SetMesh(modelData);
    renderer.SetProgram(programCtx);
    renderer.ClearColorBuffer(argb(0xFF, 0xFF, 0xFF, 0xFF));
    // renderer.ClearColorBuffer(argb(0x00, 0x00, 0x00, 0x00));
    renderer.ClearZBuffer();

    auto projVP = projectionMatrix * viewTransform;
    vertexProgram.SetModelMatrix(modelTransform);
    vertexProgram.SetViewProjectMatrix(projVP);
    pixelProgram.SetViewPosition(cameraPos);
    renderer.Canvas().SwapBuffer();
    renderer.Draw(timeElapsed);

    // for (int i = 0; i < 300; i++)
    // {
    //     memcpy(renderer.GetColorBuffer() + (i * width * 4), textureData.get() + (i * textureW * 4), 300 * 4);
    // }
}

template<CanvasDrawable Canvas>
void SceneController<Canvas>::SetHWND(NativeWindowHandle hwnd)
{
    this->hwnd = hwnd;
}

template<CanvasDrawable Canvas>
void SceneController<Canvas>::MouseDown()
{
    if (hwnd != 0)
    {
        // SetCapture(hwnd);
    }
    mouseCaptured = true;
}

template<CanvasDrawable Canvas>
void SceneController<Canvas>::MouseUp()
{
    if (hwnd != 0)
    {
        // ReleaseCapture();
    }
    mouseCaptured = false;
    lastMouseX = -1;
    lastMouseY = -1;
}

template<CanvasDrawable Canvas>
void SceneController<Canvas>::MouseWheel(int val)
{
    cameraDistance += val / 100.f;
}

template<CanvasDrawable Canvas>
void SceneController<Canvas>::MouseMove(int x, int y)
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
    cubeRotation.x -= (float)deltaX * 2.0 / (float)width;
    cubeRotation.y += (float)deltaY * 2.0 / (float)height;
    cubeRotation.y = std::clamp(cubeRotation.y, -89.99f, 89.99f);
    lastMouseX = x;
    lastMouseY = y;
}
