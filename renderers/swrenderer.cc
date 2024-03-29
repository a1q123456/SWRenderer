#include "swrenderer.h"
#include "model/data_pack.h"
#include "renderers/triangle.h"
#include "utils.h"
#include <glm/mat4x4.hpp>
#include <glm/vec4.hpp>
#include <glm/vec3.hpp>
#include <glm/common.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/euler_angles.hpp>
#include <glm/gtx/rotate_vector.hpp>
#include <numeric>

#define LOAD_ATTR_IDX(cattr, outidx, ctype)                                                                            \
    if (d.attr == cattr)                                                                                               \
    {                                                                                                                  \
        outidx = i;                                                                                                    \
        ctype = d.type;                                                                                                \
    }

SWRenderer::SWRenderer(CanvasType&& canvas)
    : canvas(std::move(canvas)), width(this->canvas.Width()), height(this->canvas.Height())
{
}

void SWRenderer::ClearColorBuffer(std::uint32_t color)
{
    canvas.Clear(color);
}

void SWRenderer::ClearZBuffer()
{
    ClearBuffer(zBuffer, 1, std::numeric_limits<float>::infinity());
}

void SWRenderer::CreateBuffer(EPixelFormat pixelFormat)
{
    zBuffer.reset(new float[width * height]);
    colorBuffer.reset(new float[width * height * 4]);
    memset(colorBuffer.get(), 0, width * height * 4);
}

SWRenderer::ProgramContextType SWRenderer::LinkProgram(VertexProgram *vp, PixelProgram *pp) noexcept
{
    SWRenderer::ProgramContextType ctx;
    ctx.vertexProgram = vp;
    ctx.pixelProgram = pp;

    ctx.vsInputDesc = ctx.vertexProgram->GetInput();
    ctx.vsOutputDesc = ctx.vertexProgram->GetOutput();
    ctx.psInputDesc = ctx.pixelProgram->GetInputDefinition();

    ctx.vertexEntry = ctx.vertexProgram->GetEntry();
    ctx.pixelEntry = ctx.pixelProgram->GetEntry();

    ctx.inputVertexAttributes = GetVertexAttributeMask(ctx.vsInputDesc);

    auto vsOutputMask = GetVertexAttributeMask(ctx.vsOutputDesc);

    assert((vsOutputMask & ((std::uint32_t)(VertexAttributes::Position))));

    ctx.vsOutputsUv = (vsOutputMask & ((std::uint32_t)(VertexAttributes::TextureCoordinate))) != 0;
    ctx.vsOutputsColor = (vsOutputMask & ((std::uint32_t)(VertexAttributes::Color))) != 0;

    VertexAttributeTypes _vsType;
    int _vsIdx = 0;

    for (int i = 0; i < ctx.vsOutputDesc.size(); i++)
    {
        auto&& d = ctx.vsOutputDesc[i];
        LOAD_ATTR_IDX(VertexAttributes::Position, ctx.vsOutputPosIdx, ctx.vsOutputPosType)
        LOAD_ATTR_IDX(VertexAttributes::TextureCoordinate, ctx.vsOutputUvIdx, ctx.vsOutputUvType)
        LOAD_ATTR_IDX(VertexAttributes::Color, ctx.vsOutputColorIdx, ctx.vsOutputColorType)
    }

    for (int src = 0; src < ctx.psInputDesc.size(); src++)
    {
        auto&& d = ctx.psInputDesc[src];
        std::vector<VertexDataDescriptor>::const_iterator iter;
        if (d.attr == VertexAttributes::Custom)
        {
            iter = std::find_if(std::cbegin(ctx.vsOutputDesc), std::cend(ctx.vsOutputDesc),
                                [d](auto&& vd) { return vd.name && std::strcmp(d.name, vd.name) == 0; });
        }
        else
        {
            iter = std::find_if(std::cbegin(ctx.vsOutputDesc), std::cend(ctx.vsOutputDesc),
                                [d](auto&& vd) { return d.attr == vd.attr; });
        }
        assert(iter != std::cend(ctx.vsOutputDesc));

        ctx.psVsIndexMap[src] = iter - std::cbegin(ctx.vsOutputDesc);
    }
    return ctx;
}

void SWRenderer::SetProgram(SWRendererProgramContext& programCtx)
{
    this->programCtx = &programCtx;
}

void SWRenderer::SetMesh(ModelData* mesh)
{
    modelData = mesh;
}

void SWRenderer::SetViewMatrix(const glm::mat4& view)
{
    viewTransform = view;
}

void SWRenderer::ProjectionMatrix(const glm::mat4& proj)
{
    projectionMatrix = proj;
}

std::size_t SWRenderer::GetNumberOfSubsamples() const noexcept
{
    return std::max(1, (1 << multisampleLevel) * (1 << multisampleLevel) / 2);
}


void SWRenderer::GenerateSubsamples(glm::vec3 pt, std::vector<glm::vec3>& subsamples)
{
    subsamples.clear();
    if (multisampleLevel == 0)
    {
        subsamples.emplace_back(pt);
        return;
    }

    int level = 1 << multisampleLevel;
    int level2 = level * level;

    auto step = 1.0 / level;
    auto start = step / 2;

    for (int v = 0; v < level2; v += 2)
    {
        int x = v % level;
        int y = v / level;

        if (y % 2 == 1)
        {
            x++;
        }

        glm::vec3 vec{x * step + start - 0.5, y * step + start - 0.5, 0.0};
        vec += pt;
        subsamples.emplace_back(vec);
    }
}


void SWRenderer::ClearBuffer(std::unique_ptr<float[]>& buffer, std::size_t nElement, float value)
{
    for (int i = 0; i < width * nElement * height; i++)
    {
        buffer[i] = value;
    }
}

void SWRenderer::Draw(float timeElapsed)
{
    stats.emplace_back(timeElapsed);
    if (stats.size() > 5)
    {
        stats.pop_front();
    }
    auto avgTime = std::accumulate(std::begin(stats), std::end(stats), 0.f) / stats.size();

    float canvasWidth = 1;
    float canvasHeight = 1;

    int nbIndices = modelData->GetNumberIndices();

    std::vector<Triangle> triangleList;
    triangleList.resize(nbIndices / 3);

#ifndef _DEBUG
#pragma omp parallel for
#endif
    for (int i = 0; i < nbIndices; i += 3)
    {
        glm::vec4 v[3];
        glm::vec4 uv[3];
        glm::vec4 color[3];
        ProgramDataPack vsInput[3];
        ProgramDataPack vsOutput[3];

        for (int j = 0; j < 3; j++)
        {
            vsInput[j].SetDataDescriptor(programCtx->vsInputDesc);
            for (auto&& d : programCtx->vsInputDesc)
            {
                for (int k = 0; k < (int)d.type; k++)
                {
                    vsInput[j].SetData(0, k, d.attr, modelData->GetVertexData(i + j, k, d.attr));
                }
            }
            vsOutput[j] = programCtx->vertexEntry(programCtx->vertexProgram, vsInput[j]);
            for (int i = 0; i < (int)programCtx->vsOutputPosType; i++)
            {
                v[j][i] = vsOutput[j].GetData(0, i, programCtx->vsOutputPosIdx);
            }
            if (programCtx->vsOutputsUv)
            {
                for (int i = 0; i < (int)programCtx->vsOutputUvType; i++)
                {
                    uv[j][i] = vsOutput[j].GetData(0, i, programCtx->vsOutputUvIdx);
                }
            }
            if (programCtx->vsOutputsColor)
            {
                for (int i = 0; i < (int)programCtx->vsOutputColorType; i++)
                {
                    color[j][i] = vsOutput[j].GetData(0, i, programCtx->vsOutputColorIdx);
                }
            }
            v[j].w = 1;
            uv[j].w = 1;
        }
        glm::vec3 sv[3];
        glm::vec3 rv[3];
        for (int j = 0; j < 3; j++)
        {
            sv[j] = glm::vec3{v[j] / v[j].z};
        }

        // backface culling
        auto t0 = sv[1] - sv[0];
        auto t1 = sv[2] - sv[1];
        if (backFaceCulling && glm::cross(t0, t1).z > 0)
        {
            continue;
        }

        for (int j = 0; j < 3; j++)
        {
            sv[j] = glm::vec3{(sv[j].x + canvasWidth / 2.f) / canvasWidth,
                              (sv[j].y + canvasHeight / 2.f) / canvasHeight, v[j].z};
            rv[j] = glm::vec3{sv[j].x * width, (1.0 - sv[j].y) * height, sv[j].z};
        }

        if (programCtx->vsOutputsUv)
        {
            for (int j = 0; j < 3; j++)
            {
                uv[j] /= rv[j].z;
                for (int i = 0; i < (int)programCtx->vsOutputUvType; i++)
                {
                    vsOutput[j].SetData(0, i, programCtx->vsOutputUvIdx, uv[j][i]);
                }
            }
        }

        if (programCtx->vsOutputsColor)
        {
            for (int j = 0; j < 3; j++)
            {
                color[j] /= rv[j].z;
                for (int i = 0; i < (int)programCtx->vsOutputColorType; i++)
                {
                    vsOutput[j].SetData(0, i, programCtx->vsOutputColorIdx, color[j][i]);
                }
            }
        }

        triangleList[i / 3] = Triangle{rv[0], rv[1], rv[2], vsOutput};
    }

    triangleList.erase(
        std::remove_if(std::begin(triangleList), std::end(triangleList), [](const Triangle& t) { return !t.isValid; }),
        std::end(triangleList));

    // for cache friendly
    std::sort(std::begin(triangleList), std::end(triangleList),
              [](const Triangle& a, const Triangle& b) { return a.avg < b.avg; });

    std::vector<glm::vec3> pixelSubsamples{};
    std::vector<int> colorMasks{};
    std::vector<int> samplesInTriangle;
    std::vector<std::tuple<glm::vec4, float>> colors{
        GetNumberOfSubsamples(), std::make_tuple(glm::vec4{}, std::numeric_limits<float>::infinity())};
    colorMasks.resize(GetNumberOfSubsamples(), 0);
    pixelSubsamples.reserve(GetNumberOfSubsamples());
    samplesInTriangle.reserve(GetNumberOfSubsamples());
    float sampleCoverage = 1.f / static_cast<float>(colors.size());

    for (int y = 0; y < height; y++)
    {
#ifndef _DEBUG
#pragma omp parallel for firstprivate(pixelSubsamples, colorMasks, colors, samplesInTriangle)
#endif
        for (int x = 0; x < width; x++)
        {
            glm::vec3 pt{x, y, 0};

            colorMasks.assign(colorMasks.size(), 0);
            colors.assign(colors.size(), std::make_tuple(glm::vec4{}, std::numeric_limits<float>::infinity()));
            GenerateSubsamples(pt, pixelSubsamples);

            for (auto&& tri : triangleList)
            {
                if (!tri.InRange(pt))
                {
                    continue;
                }

                samplesInTriangle.clear();
                for (int i = 0; i < pixelSubsamples.size(); i++)
                {
                    auto& subsample = pixelSubsamples[i];
                    if (tri.PointInTriangle(subsample))
                    {
                        samplesInTriangle.emplace_back(i);
                    }
                }

                if (samplesInTriangle.empty())
                {
                    continue;
                }

                bool hasSubsample = samplesInTriangle.size() != pixelSubsamples.size();
                if (!hasSubsample)
                {
                    samplesInTriangle.clear();
                    samplesInTriangle.push_back(0);
                }

                glm::vec3 weight = tri.Barycentric(pt);
                float depth = 1.0 / (1.0 / tri.p0.z * weight.x + 1.0 / tri.p1.z * weight.y + 1.0 / tri.p2.z * weight.z);

                ProgramDataPack psInput;
                psInput.SetDataDescriptor(programCtx->psInputDesc);

                for (int id = 0; id < programCtx->psInputDesc.size(); id++)
                {
                    auto&& d = programCtx->psInputDesc[id];
                    glm::vec4 tmpVal[3];
                    for (int j = 0; j < 3; j++)
                    {
                        for (int i = 0; i < (int)d.type; i++)
                        {
                            tmpVal[j][i] = tri.vsOutput[j].GetData(0, i, programCtx->psVsIndexMap.at(id));
                        }
                    }

                    glm::vec4 val{(tmpVal[0].x * weight.x + tmpVal[1].x * weight.y + tmpVal[2].x * weight.z),
                                  (tmpVal[0].y * weight.x + tmpVal[1].y * weight.y + tmpVal[2].y * weight.z),
                                  (tmpVal[0].z * weight.x + tmpVal[1].z * weight.y + tmpVal[2].z * weight.z),
                                  (tmpVal[0].w * weight.x + tmpVal[1].w * weight.y + tmpVal[2].w * weight.z)};

                    if (d.attr == VertexAttributes::TextureCoordinate)
                    {
                        val *= depth;
                    }
                    for (int i = 0; i < (int)d.type; i++)
                    {
                        psInput.SetData(0, i, id, val[i]);
                    }
                }

                auto sampleColor = programCtx->pixelEntry(programCtx->pixelProgram, psInput);
                bool hasRendered = false;

                for (auto sampleIndex : samplesInTriangle)
                {
                    auto& [colorUnderLayer, depthUnderLayer] = colors[sampleIndex];

                    if (depthTestEnabled && depthUnderLayer < depth)
                    {
                        continue;
                    }
                    hasRendered = true;
                    colors[sampleIndex] = std::make_tuple(sampleColor, depth);
                    colorMasks[sampleIndex] = 1;
                }

                if (!hasSubsample && hasRendered)
                {
                    for (int i = 1; i < colors.size(); i++)
                    {
                        colors[i] = colors[0];
                        colorMasks[i] = 1;
                    }
                }
            }

            float finalDepth = 0;
            glm::vec4 finalColor{};
            float sampleWeight =
                1.f / static_cast<float>(std::accumulate(std::begin(colorMasks), std::end(colorMasks), 0));

            for (int i = 0; i < colors.size(); i++)
            {
                auto& color = colors[i];
                auto& colorMask = colorMasks[i];
                if (colorMask == 0)
                {
                    continue;
                }
                auto& [sampleColor, depth] = color;

                finalColor += (sampleColor * glm::vec4{sampleWeight, sampleWeight, sampleWeight, sampleCoverage});
                finalDepth += (depth * sampleWeight);
            }

            if (depthWriteEnabled)
            {
                zBuffer[y * width + x] = finalDepth;
            }
            colorBuffer[y * width * 4 + x * 4 + 0] = finalColor.b;
            colorBuffer[y * width * 4 + x * 4 + 1] = finalColor.g;
            colorBuffer[y * width * 4 + x * 4 + 2] = finalColor.r;
            colorBuffer[y * width * 4 + x * 4 + 3] = finalColor.a;
        }
    }

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            auto alpha = colorBuffer[y * width * 4 + x * 4 + 3];
            canvas.Buffer()[(height - y - 1) * width * 4 + x * 4 + 0] =
                (std::uint8_t)std::clamp(colorBuffer[y * width * 4 + x * 4 + 0] * 255.f * alpha, 0.f, 255.f);
            canvas.Buffer()[(height - y - 1) * width * 4 + x * 4 + 1] =
                (std::uint8_t)std::clamp(colorBuffer[y * width * 4 + x * 4 + 1] * 255.f * alpha, 0.f, 255.f);
            canvas.Buffer()[(height - y - 1) * width * 4 + x * 4 + 2] =
                (std::uint8_t)std::clamp(colorBuffer[y * width * 4 + x * 4 + 2] * 255.f * alpha, 0.f, 255.f);
            canvas.Buffer()[(height - y - 1) * width * 4 + x * 4 + 3] =
                (std::uint8_t)std::clamp(alpha * 255.f, 0.f, 255.f);
        }
    }
    canvas.AddText(0, 0, 12, std::to_string(1.0 / avgTime), 0xFFFFFFFF);
}
