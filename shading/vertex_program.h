#pragma once
#include "model/vertex.h"
#include "data_pack.h"

class VertexProgram;
using TVertexFunction = ProgramDataPack(VertexProgram* self, const ProgramDataPack& args);
using VertexFunction = TVertexFunction*;

class VertexProgram
{
public:
    virtual const std::vector<VertexDataDescriptor>& GetInput() const noexcept = 0;
    virtual const std::vector<VertexDataDescriptor>& GetOutput() const noexcept = 0;
    virtual VertexFunction GetEntry() const noexcept = 0;
};
