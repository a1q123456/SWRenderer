#pragma once
#include "swrenderer.h"
#include "shading/material/simple_vertex_program.h"
#include "shading/material/blinn_material.h"
#include "model/model_object.h"

class SceneController
{
    int width = 500;
    int height = 500;

    std::shared_ptr<std::uint8_t> textureData;
    int textureW = 0;
    int textureH = 0;
    int textureChannels = 0;
    PointLight pointLight;
    AmbientLight ambientLight;
    float cameraDistance = 3.f;
    glm::vec3 cubeRotation = glm::vec3{0, 0, 0};
    HWND hwnd = 0;
    bool mouseCaptured = false;
    int lastMouseX = -1;
    int lastMouseY = -1;

    SWRenderer renderer;
    std::vector<ModelObject> sceneObjects;

public:
    SceneController()
    {
        float fov = 50;
        float aspectRatio = (float)width / (float)height;
        float zNear = 0.01;
        float zFar = 1000;

        auto projectionMatrix = glm::perspective(glm::radians(55.f), aspectRatio, zNear, zFar);
        renderer.ProjectionMatrix(projectionMatrix);
    }

    void Render(float timeElapsed)
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

        renderer.ClearColorBuffer(0);
        renderer.ClearZBuffer();

        vertexProgram.SetModelMatrix(modelTransform);
        vertexProgram.SetViewProjectMatrix(projVP);
        pixelProgram.SetViewPosition(cameraPos);

    }
};
