#include "utils.h"
#include "data_pack.h"

#define LOAD_ATTR_IDX(cattr, outidx, ctype) \
    if (d.attr == cattr)                    \
    {                                       \
        outidx = i;                         \
        ctype = d.type;                     \
    }

template <CanvasDrawable Canvas>
template <CanvasDrawable T>
SWRenderer<Canvas>::SWRenderer(T &&canvas) : canvas(std::forward<T>(canvas)),
                                             width(this->canvas.Width()),
                                             height(this->canvas.Height())
{
}

template <CanvasDrawable Canvas>
void SWRenderer<Canvas>::ClearColorBuffer(std::uint32_t color)
{
    canvas.Clear(color);
}

template <CanvasDrawable Canvas>
void SWRenderer<Canvas>::ClearZBuffer()
{
    ClearBuffer(zBuffer, 1, std::numeric_limits<float>::infinity());
}

template <CanvasDrawable Canvas>
void SWRenderer<Canvas>::CreateBuffer(EPixelFormat pixelFormat)
{
    zBuffer.reset(new float[width * height]);
    colorBuffer.reset(new float[width * height * 4]);
}

struct Triangle
{
    glm::vec3 p0, p1, p2;
    glm::vec3 min, max;

    ProgramDataPack vsOutput[3];

    bool isValid = false;

    float avg = 0.0;

    Triangle() = default;
    Triangle(
        const glm::vec3 &a,
        const glm::vec3 &b,
        const glm::vec3 &c,
        ProgramDataPack vsOutput[3]) : p0(a),
                                       p1(b),
                                       p2(c),
                                       isValid(true),
                                       min(glm::vec3{
                                           std::min(std::min(a.x, b.x), c.x),
                                           std::min(std::min(a.y, b.y), c.y),
                                           std::min(std::min(a.z, b.z), c.z)}),
                                       max(glm::vec3{
                                           std::max(std::max(a.x, b.x), c.x),
                                           std::max(std::max(a.y, b.y), c.y),
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

    bool InRange(const glm::vec3 &pt) const noexcept
    {
        return pt.x >= min.x && pt.y >= min.y && pt.x <= max.x && pt.y <= max.y;
    }

    glm::vec3 Barycentric(const glm::vec3 &pt)
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

    bool PixelInTriangle(const glm::vec3 &pt)
    {
        glm::vec3 pts[4];
        pts[0] = pt + glm::vec3{0.5, 0.5, 0};
        pts[1] = pt + glm::vec3{-0.5, 0.5, 0};
        pts[2] = pt + glm::vec3{-0.5, -0.5, 0};
        pts[3] = pt + glm::vec3{0.5, -0.5, 0};

        auto weights = pts | std::views::transform([&](auto &&pt)
                                                   { return Barycentric(pt); });
        return std::ranges::all_of(std::ranges::begin(weights), std::ranges::end(weights), [](auto &&weight)
                                   { return weight.x >= 0 && weight.y >= 0 && weight.z >= 0; });
    }

    bool PointInTriangle(const glm::vec3 &pt)
    {
        auto weight = Barycentric(pt);
        return weight.x >= 0 && weight.y >= 0 && weight.z >= 0;
    }
};

template <CanvasDrawable Canvas>
ProgramContext SWRenderer<Canvas>::LinkProgram(VertexProgram &vp, PixelProgram &pp) noexcept
{
    ProgramContext ctx;
    ctx.vertexProgram = &vp;
    ctx.pixelProgram = &pp;

    ctx.vsInputDesc = ctx.vertexProgram->GetInput();
    ctx.vsOutputDesc = ctx.vertexProgram->GetOutput();
    ctx.psInputDesc = ctx.pixelProgram->GetInput();

    ctx.inputVertexAttributes = GetVertexAttributeMask(ctx.vsInputDesc);
    ctx.vertexEntry = ctx.vertexProgram->GetEntry();
    ctx.pixelEntry = ctx.pixelProgram->GetEntry();

    auto vsOutputMask = GetVertexAttributeMask(ctx.vsOutputDesc);

    assert((vsOutputMask & ((std::uint32_t)(VertexAttributes::Position))));

    ctx.vsOutputsUv = (vsOutputMask & ((std::uint32_t)(VertexAttributes::TextureCoordinate))) != 0;
    ctx.vsOutputsColor = (vsOutputMask & ((std::uint32_t)(VertexAttributes::Color))) != 0;

    VertexAttributeTypes _vsType;
    int _vsIdx = 0;

    for (int i = 0; i < ctx.vsOutputDesc.size(); i++)
    {
        auto &&d = ctx.vsOutputDesc[i];
        LOAD_ATTR_IDX(VertexAttributes::Position, ctx.vsOutputPosIdx, ctx.vsOutputPosType)
        LOAD_ATTR_IDX(VertexAttributes::TextureCoordinate, ctx.vsOutputUvIdx, ctx.vsOutputUvType)
        LOAD_ATTR_IDX(VertexAttributes::Color, ctx.vsOutputColorIdx, ctx.vsOutputColorType)
    }

    for (int src = 0; src < ctx.psInputDesc.size(); src++)
    {
        auto &&d = ctx.psInputDesc[src];
        std::vector<VertexDataDescriptor>::const_iterator iter;
        if (d.attr == VertexAttributes::Custom)
        {
            iter = std::find_if(std::cbegin(ctx.vsOutputDesc), std::cend(ctx.vsOutputDesc), [d](auto &&vd)
                                { return vd.name && std::strcmp(d.name, vd.name) == 0; });
        }
        else
        {
            iter = std::find_if(std::cbegin(ctx.vsOutputDesc), std::cend(ctx.vsOutputDesc), [d](auto &&vd)
                                { return d.attr == vd.attr; });
        }
        assert(iter != std::cend(ctx.vsOutputDesc));

        ctx.psVsIndexMap[src] = iter - std::cbegin(ctx.vsOutputDesc);
    }
    return ctx;
}

template <CanvasDrawable Canvas>
void SWRenderer<Canvas>::SetProgram(ProgramContext &programCtx)
{
    this->programCtx = &programCtx;
}

template <CanvasDrawable Canvas>
void SWRenderer<Canvas>::SetMesh(ModelData &mesh)
{
    modelData = &mesh;
}

template <CanvasDrawable Canvas>
void SWRenderer<Canvas>::SetViewMatrix(const glm::mat4 &view)
{
    viewTransform = view;
}

template <CanvasDrawable Canvas>
void SWRenderer<Canvas>::ProjectionMatrix(const glm::mat4 &proj)
{
    projectionMatrix = proj;
}

template <CanvasDrawable Canvas>
std::vector<glm::vec3> SWRenderer<Canvas>::GenerateSubsamples(glm::vec3 pt)
{
    std::vector<glm::vec3> ret;
    auto step = 1.0 / multisampleLevel;
    auto start = step / 2;
    bool even = false;
    for (double x = start; x < 1.0; x += step)
    {
        even = !even;
        for (double y = start; y < 1.0; y += step)
        {
            even = !even;
            if (even)
            {
                continue;
            }
            glm::vec3 vec{x - 0.5, y - 0.5, 0.0};
            // vec *= (glm::sqrt(5.0) / 2.0);
            // vec = glm::rotateZ(vec, glm::atan(0.5f));
            vec += pt;
            ret.emplace_back(vec);
        }
    }

    return ret;
}

template <CanvasDrawable Canvas>
void SWRenderer<Canvas>::ClearBuffer(std::unique_ptr<float[]> &buffer, std::size_t nElement, float value)
{
    for (int i = 0; i < width * nElement * height; i++)
    {
        buffer[i] = value;
    }
}

template <CanvasDrawable Canvas>
void SWRenderer<Canvas>::Draw(float timeElapsed)
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
            for (auto &&d : programCtx->vsInputDesc)
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
            for (int i = 0; i < (int)programCtx->vsOutputUvType; i++)
            {
                if (programCtx->vsOutputsUv)
                {
                    uv[j][i] = vsOutput[j].GetData(0, i, programCtx->vsOutputUvIdx);
                }
            }
            for (int i = 0; i < (int)programCtx->vsOutputColorType; i++)
            {
                if (programCtx->vsOutputsColor)
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
            sv[j] = glm::vec3{(sv[j].x + canvasWidth / 2.f) / canvasWidth, (sv[j].y + canvasHeight / 2.f) / canvasHeight, v[j].z};
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
                for (int i = 0; i < (int)programCtx->vsOutputUvType; i++)
                {
                    vsOutput[j].SetData(0, i, programCtx->vsOutputColorIdx, color[j][i]);
                }
            }
        }

        triangleList[i / 3] = Triangle{rv[0], rv[1], rv[2], vsOutput};
#ifdef _DEBUG
        // canvas[bufferIndex]->LineTo(std::round(rv[0].x), std::round(rv[0].y), std::round(rv[1].x), std::round(rv[1].y), 0xFFFFFFFF);
        // canvas[bufferIndex]->LineTo(std::round(rv[1].x), std::round(rv[1].y), std::round(rv[2].x), std::round(rv[2].y), 0xFFFFFFFF);
        // canvas[bufferIndex]->LineTo(std::round(rv[2].x), std::round(rv[2].y), std::round(rv[0].x), std::round(rv[0].y), 0xFFFFFFFF);
#endif
    }
    triangleList.erase(std::remove_if(std::begin(triangleList), std::end(triangleList), [](const Triangle &t)
                                      { return !t.isValid; }),
                       std::end(triangleList));
    // for cache friendly
    std::sort(std::begin(triangleList), std::end(triangleList), [](const Triangle &a, const Triangle &b)
              { return a.avg < b.avg; });

    for (int y = 0; y < height; y++)
    {
#ifndef _DEBUG
#pragma omp parallel for
#endif
        for (int x = 0; x < width; x++)
        {
            glm::vec3 pt{x, y, 0};
            auto pixelSubsamples = GenerateSubsamples(pt);

            std::vector<std::tuple<glm::vec4, float>> colors{
                pixelSubsamples.size(),
                std::make_tuple(glm::vec4{}, std::numeric_limits<float>::infinity())};
            std::vector<int> colorMasks{};
            colorMasks.resize(pixelSubsamples.size(), 0);
            
            for (auto &&tri : triangleList)
            {
                if (!tri.InRange(pt))
                {
                    continue;
                }

                auto hasSubsample = !tri.PixelInTriangle(pt);
                auto subsamples = !hasSubsample ? std::vector<glm::vec3>{pt} : pixelSubsamples;

                std::vector<int> samplesInTriangle;
                for (int i = 0; i < subsamples.size(); i++)
                {
                    auto &subsample = subsamples[i];
                    if (!tri.PointInTriangle(subsample))
                    {
                        continue;
                    }
                    samplesInTriangle.emplace_back(i);
                }

                if (samplesInTriangle.empty())
                {
                    continue;
                }

                glm::vec3 weight = tri.Barycentric(pt);
                float depth = 1.0 / (1.0 / tri.p0.z * weight.x + 1.0 / tri.p1.z * weight.y + 1.0 / tri.p2.z * weight.z);

                ProgramDataPack psInput;
                psInput.SetDataDescriptor(programCtx->psInputDesc);

                for (int id = 0; id < programCtx->psInputDesc.size(); id++)
                {
                    auto &&d = programCtx->psInputDesc[id];
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
                    auto &[colorUnderLayer, depthUnderLayer] = colors[sampleIndex];

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
            float sampleWeight = 1.f / static_cast<float>(std::accumulate(std::begin(colorMasks), std::end(colorMasks), 0));
            float sampleCoverage = 1.f / static_cast<float>(colors.size());

            for (int i = 0; i < colors.size(); i++)
            {
                auto &color = colors[i];
                auto &colorMask = colorMasks[i];
                if (colorMask == 0)
                {
                    continue;
                }
                auto &[sampleColor, depth] = color;

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
            canvas.Buffer()[(height - y - 1) * width * 4 + x * 4 + 0] = (std::uint8_t)std::clamp(colorBuffer.get()[y * width * 4 + x * 4 + 0] * 255.f, 0.f, 255.f);
            canvas.Buffer()[(height - y - 1) * width * 4 + x * 4 + 1] = (std::uint8_t)std::clamp(colorBuffer.get()[y * width * 4 + x * 4 + 1] * 255.f, 0.f, 255.f);
            canvas.Buffer()[(height - y - 1) * width * 4 + x * 4 + 2] = (std::uint8_t)std::clamp(colorBuffer.get()[y * width * 4 + x * 4 + 2] * 255.f, 0.f, 255.f);
            canvas.Buffer()[(height - y - 1) * width * 4 + x * 4 + 3] = (std::uint8_t)std::clamp(colorBuffer.get()[y * width * 4 + x * 4 + 3] * 255.f, 0.f, 255.f);
        }
    }

    canvas.AddText(0, 0, 12, std::to_string(1.0 / avgTime), 0xFFFFFFFF);
}
