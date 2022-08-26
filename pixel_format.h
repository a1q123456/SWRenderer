#pragma once
#include "resource.h"

enum class EPixelFormat
{
    R_I8,
    RGB_I8,
    RGBA_I8,
    ARGB_I8,
    BGRA_I8,
    ABGR_I8,

    R_I16,
    RGB_I16,
    RGBA_I16,
    ARGB_I16,
    BGRA_I16,
    ABGR_I16,
        
    R_I32,
    RGB_I32,
    RGBA_I32,
    ARGB_I32,
    BGRA_I32,
    ABGR_I32,

    R_I64,
    RGB_I64,
    RGBA_I64,
    ARGB_I64,
    BGRA_I64,
    ABGR_I64,

    R_U8,
    RGB_U8,
    RGBA_U8,
    ARGB_U8,
    BGRA_U8,
    ABGR_U8,

    R_U16,
    RGB_U16,
    RGBA_U16,
    ARGB_U16,
    BGRA_U16,
    ABGR_U16,
        
    R_U32,
    RGB_U32,
    RGBA_U32,
    ARGB_U32,
    BGRA_U32,
    ABGR_U32,
        
    R_U64,
    RGB_U64,    
    RGBA_U64,
    ARGB_U64,
    BGRA_U64,
    ABGR_U64,

    R_FLOAT,
    RGB_FLOAT,
    RGBA_FLOAT,
    ARGB_FLOAT,
    BGRA_FLOAT,
    ABGR_FLOAT,

    R_DOUBLE,
    RGB_DOUBLE,
    RGBA_DOUBLE,
    ARGB_DOUBLE,
    BGRA_DOUBLE,
    ABGR_DOUBLE,

    YUV420P_I8,
    YUV420P_U8,

    NV12_U8,

    MAX_PIX_FMT
};

constexpr auto TEXTURE_MAX_CHANNELS = 4;

struct ComponentDesc
{
    int plane;
    int step;
    int offset;
    int shift;
    int depth;
};

struct PixelDesc
{
    int nbChannels;
    EResourceDataType dataType;
    bool isPlannar;
    bool isYUV;
    bool isRGB;
    int log2ChromaW;
    int log2ChromaH;
    ComponentDesc components[TEXTURE_MAX_CHANNELS];
};

PixelDesc* getPixelDesc(EPixelFormat pixFmt);
