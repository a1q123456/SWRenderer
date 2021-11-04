#include "raytracing_renderer.h"
#include "utils.h"

void RayTracingRenderer::Render()
{
    auto width = 500;
    auto height = 500;

    // for (int y = 0; y < height; y++)
    // {
    //     for (int x = 0; x < width; x++)
    //     {
    //         Ray ray
    //         {
    //             glm::vec3{0, 0, 0},
    //             glm::normalize(
    //                 glm::vec3{x, y, viewMatrix[0][0]} - glm::vec3{width / 2, height / 2, 0.0}
    //             )
    //         };
    //         for (auto& triangle : trangleList)
    //         {
    //             glm::vec3 pos;
    //             if (ray.tryIntersect(triangle, pos))
    //             {
    //                 auto barycentric = triangle.barycentric(pos);

    //                 for (auto& light : lightList)
    //                 {
    //                     std::vector<LightSample> sampleList;
    //                     if (light.hasArea)
    //                     {
    //                         sampleList.emplace_back(light.CreateSamplePoints(pos));
    //                     }
    //                     else
    //                     {
    //                         sampleList.emplace_back(light.sample);
    //                     }
    //                     std::vector<LightSample> visibleSamples;
    //                     auto sampleIntensityView = sampleList | std::views::filter([&](auto&& sample) -> bool
    //                     {
    //                         return trangleList | std::views::all([&](auto&& tri)
    //                         {
    //                             return !tri.hasIntersection(Segment{sample.pos, pos});
    //                         });
    //                     });
    //                     std::ranges::copy(sampleIntensityView, std::back_inserter(visibleSamples));
    //                     auto intensity = std::accumulate(std::begin(visibleSamples), std::end(visibleSamples), 0.0, [](double a, auto&& sample)
    //                     {
    //                         return a + sample.intensity;
    //                     });
                        

    //                 }
    //             }
    //         }
    //     }
    // }
}