void AnisotropicSampler::AttachResource(std::span<float> buffer, ResourceDescriptor rd)
{
    this->arrayBuffer.clear();
    arrayBuffer.emplace_back(buffer);
    this->rd = rd;
}

void AnisotropicSampler::AttachResource(const std::vector<std::span<float>> &arrayBuffer, ResourceDescriptor rd)
{
    this->arrayBuffer = arrayBuffer;
    this->rd = rd;
}

glm::vec4 AnisotropicSampler::Sample(const glm::vec2 &coord)
{
    return {};
}
