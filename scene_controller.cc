#include "scene_controller.h"
#include "utils.h"

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

SceneController::SceneController(CanvasType&& canvas) : width(canvas.Width()), height(canvas.Height()), renderer(std::move(canvas)), textureData(nullptr, stbi_image_free)
{
    float fov = 55.f;
    float aspectRatio = (float)width / (float)height;
    float zNear = 0.01;
    float zFar = 1000;

    renderer.SetMultiSampleLevel(2);

    projectionMatrix = glm::perspective(glm::radians(fov), aspectRatio, zNear, zFar);
    renderer.ProjectionMatrix(projectionMatrix);

    textureData.reset(stbi_load("sample_640x426.jpeg", &textureW, &textureH, &textureChannels, STBI_rgb_alpha));
    if (!textureData)
    {
        throw std::runtime_error(stbi_failure_reason());
    }

    modelData = CudaNewManaged<RendererType::ModelDataType>();

    modelData->SetIndexList({std::cbegin(indexList), std::cend(indexList)});
    modelData->SetVertexList({std::cbegin(vertexList), std::cend(vertexList)});

    modelData->SetVertexDescriptor({
        {VertexAttributes::Position, VertexAttributeTypes::Vec3},
        {VertexAttributes::TextureCoordinate, VertexAttributeTypes::Vec3},
        {VertexAttributes::Normal, VertexAttributeTypes::Vec3},
    });

    pixelProgram = CudaNewManaged<PBRMaterial>();

    pixelProgram->SetDiffuseMap(CudaUploadData<std::uint8_t>(textureData.get(), textureW * textureH * 4), textureW, textureH);
    programCtx = renderer.LinkProgram(pixelProgram.get());
}

void SceneController::Render(float timeElapsed)
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

    renderer.AddMesh(modelData.get(), programCtx);
    renderer.ClearColorBuffer(argb(0xFF, 0xFF, 0xFF, 0xFF));
    renderer.ClearZBuffer();

    renderer.Canvas().SwapBuffer();
    renderer.Draw(timeElapsed);
    renderer.Canvas().AddText(0, 24, 12, std::format("x: {}, y: {}", mouseX, mouseY), 0xFFFFFFFF);
}


void SceneController::SetHWND(NativeWindowHandle hwnd)
{
    this->hwnd = hwnd;
}


void SceneController::MouseDown()
{
    if (hwnd != 0)
    {
        // SetCapture(hwnd);
    }
    mouseCaptured = true;
}


void SceneController::MouseUp()
{
    if (hwnd != 0)
    {
        // ReleaseCapture();
    }
    mouseCaptured = false;
    lastMouseX = -1;
    lastMouseY = -1;
}


void SceneController::MouseWheel(int val)
{
    cameraDistance += val / 100.f;
}


void SceneController::MouseMove(int x, int y)
{
    mouseX = x;
    mouseY = y;
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
    cubeRotation.y -= (float)deltaY * 2.0 / (float)height;
    cubeRotation.y = std::clamp(cubeRotation.y, -89.99f, 89.99f);
    lastMouseX = x;
    lastMouseY = y;
}
