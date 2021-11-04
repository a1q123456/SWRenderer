#pragma once
#include "model/model_data.h"

class ObjLoader
{
    struct ParserContext
    {
        struct Object
        {
            struct Group
            {
                std::vector<glm::vec3> vertices;
                std::vector<glm::vec3> normals;
                std::vector<glm::vec3> uvs;
                std::vector<int> indices;
                std::string name;
                std::string materialName;
            };
            std::vector<Group> groups;
            int currentIdx = -1;
            std::string name;

            Group &currentGroup() noexcept { return groups[currentIdx]; }
        };
        std::vector<Object> objs;
        int currentIdx = -1;

        Object &currentObj() noexcept { return objs[currentIdx]; }
    };

    template<size_t N, typename TFloat, glm::qualifier Q>
    bool TryParseVec(std::string_view sv, glm::vec<N, TFloat, Q> &v)
    {
        for (int i = 0; i < N; i++)
        {
            auto pos = sv.find(' ');
            std::string sval{sv.substr(0, pos)};
            sv = sv.substr(pos + 1);
            char *endPtr = nullptr;
            auto nval = std::strtol(sval.c_str(), &endPtr, 10);
            if (*endPtr != '\0')
            {
                return false;
            }
            v[i] = nval;
        }
        return true;
    }

public:
    bool TryParseMtllib(std::string_view sv, ParserContext &ctx)
    {
        ctx.currentObj().currentGroup().materialName = sv;
        return true;
    }

    bool TryParseO(std::string_view sv, ParserContext &ctx)
    {
        ctx.currentIdx++;
        ctx.objs.emplace_back();
        ctx.currentObj().name = sv;
        return true;
    }

    bool TryParseV(std::string_view sv, ParserContext &ctx)
    {
        glm::vec3 v;
        if (!TryParseVec(sv, v))
        {
            return false;
        }
        ctx.currentObj().currentGroup().vertices.emplace_back(v);
        return true;
    }

    bool TryParseVt(std::string_view sv, ParserContext &ctx)
    {
        glm::vec2 v;
        if (!TryParseVec(sv, v))
        {
            return false;
        }
        ctx.currentObj().currentGroup().vertices.emplace_back(v);
        return true;
    }

    bool TryParseVn(std::string_view sv, ParserContext &ctx)
    {
        glm::vec3 v;
        if (!TryParseVec(sv, v))
        {
            return false;
        }
        ctx.currentObj().currentGroup().normals.emplace_back(v);
        return true;
    }

    void LoadModel(std::istream &is)
    {
        std::string line;
        while (std::getline(is, line))
        {
            if (line.empty())
            {
                continue;
            }
            std::string_view sv{line};
            auto pos = sv.find(' ');
            auto kind = sv.substr(0, pos);
        }
    }
};
