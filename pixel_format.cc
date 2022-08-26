#include "pixel_format.h"
#include "resource.h"

namespace
{
static PixelDesc pixDesc[static_cast<int>(EPixelFormat::MAX_PIX_FMT)]{};

struct PixDescInitialiser
{
    PixDescInitialiser()
    {
        pixDesc[static_cast<int>(EPixelFormat::R_I8)] = {
            .nbChannels = 1,
            .dataType = EResourceDataType::Int8,
            .isPlannar = false,
            .isYUV = false,
            .isRGB = true,
            .log2ChromaW = 0,
            .log2ChromaH = 0,
            .components =
                {
                    {
                        .plane = 0,
                        .step = 1,
                        .offset = 0,
                        .shift = 0,
                        .depth = 8,
                    }
                },
        };
        pixDesc[static_cast<int>(EPixelFormat::RGB_I8)] = {
            .nbChannels = 3,
            .dataType = EResourceDataType::Int8,
            .isPlannar = false,
            .isYUV = false,
            .isRGB = true,
            .log2ChromaW = 0,
            .log2ChromaH = 0,
            .components =
                {
                    {
                        .plane = 0,
                        .step = 3,
                        .offset = 0,
                        .shift = 0,
                        .depth = 8,
                    },
                    {
                        .plane = 0,
                        .step = 3,
                        .offset = 1,
                        .shift = 0,
                        .depth = 8,
                    },
                    {
                        .plane = 0,
                        .step = 3,
                        .offset = 2,
                        .shift = 0,
                        .depth = 8,
                    },
                },
        };
        pixDesc[static_cast<int>(EPixelFormat::RGBA_I8)] = {
            .nbChannels = 4,
            .dataType = EResourceDataType::Int8,
            .isPlannar = false,
            .isYUV = false,
            .isRGB = true,
            .log2ChromaW = 0,
            .log2ChromaH = 0,
            .components =
                {
                    {
                        .plane = 0,
                        .step = 4,
                        .offset = 0,
                        .shift = 0,
                        .depth = 8,
                    },
                    {
                        .plane = 0,
                        .step = 4,
                        .offset = 1,
                        .shift = 0,
                        .depth = 8,
                    },
                    {
                        .plane = 0,
                        .step = 4,
                        .offset = 2,
                        .shift = 0,
                        .depth = 8,
                    },
                    {
                        .plane = 0,
                        .step = 4,
                        .offset = 3,
                        .shift = 0,
                        .depth = 8,
                    },
                },
        };
        pixDesc[static_cast<int>(EPixelFormat::ARGB_I8)] = pixDesc[static_cast<int>(EPixelFormat::RGBA_I8)];
        pixDesc[static_cast<int>(EPixelFormat::BGRA_I8)] = pixDesc[static_cast<int>(EPixelFormat::RGBA_I8)];
        pixDesc[static_cast<int>(EPixelFormat::ABGR_I8)] = pixDesc[static_cast<int>(EPixelFormat::RGBA_I8)];
        pixDesc[static_cast<int>(EPixelFormat::R_I16)] = {
            .nbChannels = 1,
            .dataType = EResourceDataType::Int16,
            .isPlannar = false,
            .isYUV = false,
            .isRGB = true,
            .log2ChromaW = 0,
            .log2ChromaH = 0,
            .components =
                {
                    {
                        .plane = 0,
                        .step = 1,
                        .offset = 0,
                        .shift = 0,
                        .depth = 16,
                    }
                },
        };
        pixDesc[static_cast<int>(EPixelFormat::RGB_I16)] = {
            .nbChannels = 3,
            .dataType = EResourceDataType::Int16,
            .isPlannar = false,
            .isYUV = false,
            .isRGB = true,
            .log2ChromaW = 0,
            .log2ChromaH = 0,
            .components =
                {
                    {
                        .plane = 0,
                        .step = 3,
                        .offset = 0,
                        .shift = 0,
                        .depth = 16,
                    },
                    {
                        .plane = 0,
                        .step = 3,
                        .offset = 1,
                        .shift = 0,
                        .depth = 16,
                    },
                    {
                        .plane = 0,
                        .step = 3,
                        .offset = 2,
                        .shift = 0,
                        .depth = 16,
                    },
                },
        };
        pixDesc[static_cast<int>(EPixelFormat::RGBA_I16)] = {
            .nbChannels = 4,
            .dataType = EResourceDataType::Int16,
            .isPlannar = false,
            .isYUV = false,
            .isRGB = true,
            .log2ChromaW = 0,
            .log2ChromaH = 0,
            .components =
                {
                    {
                        .plane = 0,
                        .step = 4,
                        .offset = 0,
                        .shift = 0,
                        .depth = 16,
                    },
                    {
                        .plane = 0,
                        .step = 4,
                        .offset = 1,
                        .shift = 0,
                        .depth = 16,
                    },
                    {
                        .plane = 0,
                        .step = 4,
                        .offset = 2,
                        .shift = 0,
                        .depth = 16,
                    },
                    {
                        .plane = 0,
                        .step = 4,
                        .offset = 3,
                        .shift = 0,
                        .depth = 16,
                    },
                },
        };
        pixDesc[static_cast<int>(EPixelFormat::ARGB_I16)] = pixDesc[static_cast<int>(EPixelFormat::RGBA_I16)];
        pixDesc[static_cast<int>(EPixelFormat::BGRA_I16)] = pixDesc[static_cast<int>(EPixelFormat::RGBA_I16)];
        pixDesc[static_cast<int>(EPixelFormat::ABGR_I16)] = pixDesc[static_cast<int>(EPixelFormat::RGBA_I16)];
        pixDesc[static_cast<int>(EPixelFormat::R_I32)] = {
            .nbChannels = 1,
            .dataType = EResourceDataType::Int32,
            .isPlannar = false,
            .isYUV = false,
            .isRGB = true,
            .log2ChromaW = 0,
            .log2ChromaH = 0,
            .components =
                {
                    {
                        .plane = 0,
                        .step = 1,
                        .offset = 0,
                        .shift = 0,
                        .depth = 32,
                    }
                },
        };
        pixDesc[static_cast<int>(EPixelFormat::RGB_I32)] = {
            .nbChannels = 3,
            .dataType = EResourceDataType::Int32,
            .isPlannar = false,
            .isYUV = false,
            .isRGB = true,
            .log2ChromaW = 0,
            .log2ChromaH = 0,
            .components =
                {
                    {
                        .plane = 0,
                        .step = 3,
                        .offset = 0,
                        .shift = 0,
                        .depth = 32,
                    },
                    {
                        .plane = 0,
                        .step = 3,
                        .offset = 1,
                        .shift = 0,
                        .depth = 32,
                    },
                    {
                        .plane = 0,
                        .step = 3,
                        .offset = 2,
                        .shift = 0,
                        .depth = 32,
                    },
                },
        };
        pixDesc[static_cast<int>(EPixelFormat::RGBA_I32)] = {
            .nbChannels = 4,
            .dataType = EResourceDataType::Int32,
            .isPlannar = false,
            .isYUV = false,
            .isRGB = true,
            .log2ChromaW = 0,
            .log2ChromaH = 0,
            .components =
                {
                    {
                        .plane = 0,
                        .step = 4,
                        .offset = 0,
                        .shift = 0,
                        .depth = 32,
                    },
                    {
                        .plane = 0,
                        .step = 4,
                        .offset = 1,
                        .shift = 0,
                        .depth = 32,
                    },
                    {
                        .plane = 0,
                        .step = 4,
                        .offset = 2,
                        .shift = 0,
                        .depth = 32,
                    },
                    {
                        .plane = 0,
                        .step = 4,
                        .offset = 3,
                        .shift = 0,
                        .depth = 32,
                    },
                },
        };
        pixDesc[static_cast<int>(EPixelFormat::ARGB_I32)] = pixDesc[static_cast<int>(EPixelFormat::RGBA_I32)];
        pixDesc[static_cast<int>(EPixelFormat::BGRA_I32)] = pixDesc[static_cast<int>(EPixelFormat::RGBA_I32)];
        pixDesc[static_cast<int>(EPixelFormat::ABGR_I32)] = pixDesc[static_cast<int>(EPixelFormat::RGBA_I32)];
        pixDesc[static_cast<int>(EPixelFormat::R_I64)] = {
            .nbChannels = 1,
            .dataType = EResourceDataType::Int64,
            .isPlannar = false,
            .isYUV = false,
            .isRGB = true,
            .log2ChromaW = 0,
            .log2ChromaH = 0,
            .components =
                {
                    {
                        .plane = 0,
                        .step = 1,
                        .offset = 0,
                        .shift = 0,
                        .depth = 64,
                    }
                },
        };
        pixDesc[static_cast<int>(EPixelFormat::RGB_I64)] = {
            .nbChannels = 3,
            .dataType = EResourceDataType::Int64,
            .isPlannar = false,
            .isYUV = false,
            .isRGB = true,
            .log2ChromaW = 0,
            .log2ChromaH = 0,
            .components =
                {
                    {
                        .plane = 0,
                        .step = 3,
                        .offset = 0,
                        .shift = 0,
                        .depth = 64,
                    },
                    {
                        .plane = 0,
                        .step = 3,
                        .offset = 1,
                        .shift = 0,
                        .depth = 64,
                    },
                    {
                        .plane = 0,
                        .step = 3,
                        .offset = 2,
                        .shift = 0,
                        .depth = 64,
                    },
                },
        };
        pixDesc[static_cast<int>(EPixelFormat::RGBA_I64)] = {
            .nbChannels = 4,
            .dataType = EResourceDataType::Int64,
            .isPlannar = false,
            .isYUV = false,
            .isRGB = true,
            .log2ChromaW = 0,
            .log2ChromaH = 0,
            .components =
                {
                    {
                        .plane = 0,
                        .step = 4,
                        .offset = 0,
                        .shift = 0,
                        .depth = 64,
                    },
                    {
                        .plane = 0,
                        .step = 4,
                        .offset = 1,
                        .shift = 0,
                        .depth = 64,
                    },
                    {
                        .plane = 0,
                        .step = 4,
                        .offset = 2,
                        .shift = 0,
                        .depth = 64,
                    },
                    {
                        .plane = 0,
                        .step = 4,
                        .offset = 3,
                        .shift = 0,
                        .depth = 64,
                    },
                },
        };
        pixDesc[static_cast<int>(EPixelFormat::ARGB_I64)] = pixDesc[static_cast<int>(EPixelFormat::RGBA_I64)];
        pixDesc[static_cast<int>(EPixelFormat::BGRA_I64)] = pixDesc[static_cast<int>(EPixelFormat::RGBA_I64)];
        pixDesc[static_cast<int>(EPixelFormat::ABGR_I64)] = pixDesc[static_cast<int>(EPixelFormat::RGBA_I64)];
        pixDesc[static_cast<int>(EPixelFormat::R_U8)] = {
            .nbChannels = 1,
            .dataType = EResourceDataType::UInt8,
            .isPlannar = false,
            .isYUV = false,
            .isRGB = true,
            .log2ChromaW = 0,
            .log2ChromaH = 0,
            .components =
                {
                    {
                        .plane = 0,
                        .step = 1,
                        .offset = 0,
                        .shift = 0,
                        .depth = 8,
                    }
                },
        };
        pixDesc[static_cast<int>(EPixelFormat::RGB_U8)] = {
            .nbChannels = 3,
            .dataType = EResourceDataType::UInt8,
            .isPlannar = false,
            .isYUV = false,
            .isRGB = true,
            .log2ChromaW = 0,
            .log2ChromaH = 0,
            .components =
                {
                    {
                        .plane = 0,
                        .step = 3,
                        .offset = 0,
                        .shift = 0,
                        .depth = 8,
                    },
                    {
                        .plane = 0,
                        .step = 3,
                        .offset = 1,
                        .shift = 0,
                        .depth = 8,
                    },
                    {
                        .plane = 0,
                        .step = 3,
                        .offset = 2,
                        .shift = 0,
                        .depth = 8,
                    },
                },
        };
        pixDesc[static_cast<int>(EPixelFormat::RGBA_U8)] = {
            .nbChannels = 4,
            .dataType = EResourceDataType::UInt8,
            .isPlannar = false,
            .isYUV = false,
            .isRGB = true,
            .log2ChromaW = 0,
            .log2ChromaH = 0,
            .components =
                {
                    {
                        .plane = 0,
                        .step = 4,
                        .offset = 0,
                        .shift = 0,
                        .depth = 8,
                    },
                    {
                        .plane = 0,
                        .step = 4,
                        .offset = 1,
                        .shift = 0,
                        .depth = 8,
                    },
                    {
                        .plane = 0,
                        .step = 4,
                        .offset = 2,
                        .shift = 0,
                        .depth = 8,
                    },
                    {
                        .plane = 0,
                        .step = 4,
                        .offset = 3,
                        .shift = 0,
                        .depth = 8,
                    },
                },
        };
        pixDesc[static_cast<int>(EPixelFormat::ARGB_U8)] = pixDesc[static_cast<int>(EPixelFormat::RGBA_U8)];
        pixDesc[static_cast<int>(EPixelFormat::BGRA_U8)] = pixDesc[static_cast<int>(EPixelFormat::RGBA_U8)];
        pixDesc[static_cast<int>(EPixelFormat::ABGR_U8)] = pixDesc[static_cast<int>(EPixelFormat::RGBA_U8)];
        pixDesc[static_cast<int>(EPixelFormat::R_U16)] = {
            .nbChannels = 1,
            .dataType = EResourceDataType::UInt16,
            .isPlannar = false,
            .isYUV = false,
            .isRGB = true,
            .log2ChromaW = 0,
            .log2ChromaH = 0,
            .components =
                {
                    {
                        .plane = 0,
                        .step = 1,
                        .offset = 0,
                        .shift = 0,
                        .depth = 16,
                    }
                },
        };
        pixDesc[static_cast<int>(EPixelFormat::RGB_U16)] = {
            .nbChannels = 3,
            .dataType = EResourceDataType::UInt16,
            .isPlannar = false,
            .isYUV = false,
            .isRGB = true,
            .log2ChromaW = 0,
            .log2ChromaH = 0,
            .components =
                {
                    {
                        .plane = 0,
                        .step = 3,
                        .offset = 0,
                        .shift = 0,
                        .depth = 16,
                    },
                    {
                        .plane = 0,
                        .step = 3,
                        .offset = 1,
                        .shift = 0,
                        .depth = 16,
                    },
                    {
                        .plane = 0,
                        .step = 3,
                        .offset = 2,
                        .shift = 0,
                        .depth = 16,
                    },
                },
        };
        pixDesc[static_cast<int>(EPixelFormat::RGBA_U16)] = {
            .nbChannels = 4,
            .dataType = EResourceDataType::UInt16,
            .isPlannar = false,
            .isYUV = false,
            .isRGB = true,
            .log2ChromaW = 0,
            .log2ChromaH = 0,
            .components =
                {
                    {
                        .plane = 0,
                        .step = 4,
                        .offset = 0,
                        .shift = 0,
                        .depth = 16,
                    },
                    {
                        .plane = 0,
                        .step = 4,
                        .offset = 1,
                        .shift = 0,
                        .depth = 16,
                    },
                    {
                        .plane = 0,
                        .step = 4,
                        .offset = 2,
                        .shift = 0,
                        .depth = 16,
                    },
                    {
                        .plane = 0,
                        .step = 4,
                        .offset = 3,
                        .shift = 0,
                        .depth = 16,
                    },
                },
        };
        pixDesc[static_cast<int>(EPixelFormat::ARGB_U16)] = pixDesc[static_cast<int>(EPixelFormat::RGBA_U16)];
        pixDesc[static_cast<int>(EPixelFormat::BGRA_U16)] = pixDesc[static_cast<int>(EPixelFormat::RGBA_U16)];
        pixDesc[static_cast<int>(EPixelFormat::ABGR_U16)] = pixDesc[static_cast<int>(EPixelFormat::RGBA_U16)];
        pixDesc[static_cast<int>(EPixelFormat::R_U32)] = {
            .nbChannels = 1,
            .dataType = EResourceDataType::UInt32,
            .isPlannar = false,
            .isYUV = false,
            .isRGB = true,
            .log2ChromaW = 0,
            .log2ChromaH = 0,
            .components =
                {
                    {
                        .plane = 0,
                        .step = 1,
                        .offset = 0,
                        .shift = 0,
                        .depth = 32,
                    }
                },
        };
        pixDesc[static_cast<int>(EPixelFormat::RGB_U32)] = {
            .nbChannels = 3,
            .dataType = EResourceDataType::UInt32,
            .isPlannar = false,
            .isYUV = false,
            .isRGB = true,
            .log2ChromaW = 0,
            .log2ChromaH = 0,
            .components =
                {
                    {
                        .plane = 0,
                        .step = 3,
                        .offset = 0,
                        .shift = 0,
                        .depth = 32,
                    },
                    {
                        .plane = 0,
                        .step = 3,
                        .offset = 1,
                        .shift = 0,
                        .depth = 32,
                    },
                    {
                        .plane = 0,
                        .step = 3,
                        .offset = 2,
                        .shift = 0,
                        .depth = 32,
                    },
                },
        };
        pixDesc[static_cast<int>(EPixelFormat::RGBA_U32)] = {
            .nbChannels = 4,
            .dataType = EResourceDataType::UInt32,
            .isPlannar = false,
            .isYUV = false,
            .isRGB = true,
            .log2ChromaW = 0,
            .log2ChromaH = 0,
            .components =
                {
                    {
                        .plane = 0,
                        .step = 4,
                        .offset = 0,
                        .shift = 0,
                        .depth = 32,
                    },
                    {
                        .plane = 0,
                        .step = 4,
                        .offset = 1,
                        .shift = 0,
                        .depth = 32,
                    },
                    {
                        .plane = 0,
                        .step = 4,
                        .offset = 2,
                        .shift = 0,
                        .depth = 32,
                    },
                    {
                        .plane = 0,
                        .step = 4,
                        .offset = 3,
                        .shift = 0,
                        .depth = 32,
                    },
                },
        };
        pixDesc[static_cast<int>(EPixelFormat::ARGB_U32)] = pixDesc[static_cast<int>(EPixelFormat::RGBA_U32)];
        pixDesc[static_cast<int>(EPixelFormat::BGRA_U32)] = pixDesc[static_cast<int>(EPixelFormat::RGBA_U32)];
        pixDesc[static_cast<int>(EPixelFormat::ABGR_U32)] = pixDesc[static_cast<int>(EPixelFormat::RGBA_U32)];
        pixDesc[static_cast<int>(EPixelFormat::R_U64)] = {
            .nbChannels = 1,
            .dataType = EResourceDataType::UInt64,
            .isPlannar = false,
            .isYUV = false,
            .isRGB = true,
            .log2ChromaW = 0,
            .log2ChromaH = 0,
            .components =
                {
                    {
                        .plane = 0,
                        .step = 1,
                        .offset = 0,
                        .shift = 0,
                        .depth = 64,
                    }
                },
        };
        pixDesc[static_cast<int>(EPixelFormat::RGB_U64)] = {
            .nbChannels = 3,
            .dataType = EResourceDataType::UInt64,
            .isPlannar = false,
            .isYUV = false,
            .isRGB = true,
            .log2ChromaW = 0,
            .log2ChromaH = 0,
            .components =
                {
                    {
                        .plane = 0,
                        .step = 3,
                        .offset = 0,
                        .shift = 0,
                        .depth = 64,
                    },
                    {
                        .plane = 0,
                        .step = 3,
                        .offset = 1,
                        .shift = 0,
                        .depth = 64,
                    },
                    {
                        .plane = 0,
                        .step = 3,
                        .offset = 2,
                        .shift = 0,
                        .depth = 64,
                    },
                },
        };
        pixDesc[static_cast<int>(EPixelFormat::RGBA_U64)] = {
            .nbChannels = 4,
            .dataType = EResourceDataType::UInt64,
            .isPlannar = false,
            .isYUV = false,
            .isRGB = true,
            .log2ChromaW = 0,
            .log2ChromaH = 0,
            .components =
                {
                    {
                        .plane = 0,
                        .step = 4,
                        .offset = 0,
                        .shift = 0,
                        .depth = 64,
                    },
                    {
                        .plane = 0,
                        .step = 4,
                        .offset = 1,
                        .shift = 0,
                        .depth = 64,
                    },
                    {
                        .plane = 0,
                        .step = 4,
                        .offset = 2,
                        .shift = 0,
                        .depth = 64,
                    },
                    {
                        .plane = 0,
                        .step = 4,
                        .offset = 3,
                        .shift = 0,
                        .depth = 64,
                    },
                },
        };
        pixDesc[static_cast<int>(EPixelFormat::ARGB_U64)] = pixDesc[static_cast<int>(EPixelFormat::RGBA_U64)];
        pixDesc[static_cast<int>(EPixelFormat::BGRA_U64)] = pixDesc[static_cast<int>(EPixelFormat::RGBA_U64)];
        pixDesc[static_cast<int>(EPixelFormat::ABGR_U64)] = pixDesc[static_cast<int>(EPixelFormat::RGBA_U64)];
        pixDesc[static_cast<int>(EPixelFormat::R_FLOAT)] = {
            .nbChannels = 1,
            .dataType = EResourceDataType::Float,
            .isPlannar = false,
            .isYUV = false,
            .isRGB = true,
            .log2ChromaW = 0,
            .log2ChromaH = 0,
            .components =
                {
                    {
                        .plane = 0,
                        .step = 1,
                        .offset = 0,
                        .shift = 0,
                        .depth = 32,
                    }
                },
        };
        pixDesc[static_cast<int>(EPixelFormat::RGB_FLOAT)] = {
            .nbChannels = 3,
            .dataType = EResourceDataType::Float,
            .isPlannar = false,
            .isYUV = false,
            .isRGB = true,
            .log2ChromaW = 0,
            .log2ChromaH = 0,
            .components =
                {
                    {
                        .plane = 0,
                        .step = 3,
                        .offset = 0,
                        .shift = 0,
                        .depth = 32,
                    },
                    {
                        .plane = 0,
                        .step = 3,
                        .offset = 1,
                        .shift = 0,
                        .depth = 32,
                    },
                    {
                        .plane = 0,
                        .step = 3,
                        .offset = 2,
                        .shift = 0,
                        .depth = 32,
                    },
                },
        };
        pixDesc[static_cast<int>(EPixelFormat::RGBA_FLOAT)] = {
            .nbChannels = 4,
            .dataType = EResourceDataType::Float,
            .isPlannar = false,
            .isYUV = false,
            .isRGB = true,
            .log2ChromaW = 0,
            .log2ChromaH = 0,
            .components =
                {
                    {
                        .plane = 0,
                        .step = 4,
                        .offset = 0,
                        .shift = 0,
                        .depth = 32,
                    },
                    {
                        .plane = 0,
                        .step = 4,
                        .offset = 1,
                        .shift = 0,
                        .depth = 32,
                    },
                    {
                        .plane = 0,
                        .step = 4,
                        .offset = 2,
                        .shift = 0,
                        .depth = 32,
                    },
                    {
                        .plane = 0,
                        .step = 4,
                        .offset = 3,
                        .shift = 0,
                        .depth = 32,
                    },
                },
        };
        pixDesc[static_cast<int>(EPixelFormat::ARGB_FLOAT)] = pixDesc[static_cast<int>(EPixelFormat::RGBA_FLOAT)];
        pixDesc[static_cast<int>(EPixelFormat::BGRA_FLOAT)] = pixDesc[static_cast<int>(EPixelFormat::RGBA_FLOAT)];
        pixDesc[static_cast<int>(EPixelFormat::ABGR_FLOAT)] = pixDesc[static_cast<int>(EPixelFormat::RGBA_FLOAT)];
        pixDesc[static_cast<int>(EPixelFormat::R_DOUBLE)] = {
            .nbChannels = 1,
            .dataType = EResourceDataType::Double,
            .isPlannar = false,
            .isYUV = false,
            .isRGB = true,
            .log2ChromaW = 0,
            .log2ChromaH = 0,
            .components =
                {
                    {
                        .plane = 0,
                        .step = 1,
                        .offset = 0,
                        .shift = 0,
                        .depth = 64,
                    }
                },
        };
        pixDesc[static_cast<int>(EPixelFormat::RGB_DOUBLE)] = {
            .nbChannels = 3,
            .dataType = EResourceDataType::Double,
            .isPlannar = false,
            .isYUV = false,
            .isRGB = true,
            .log2ChromaW = 0,
            .log2ChromaH = 0,
            .components =
                {
                    {
                        .plane = 0,
                        .step = 3,
                        .offset = 0,
                        .shift = 0,
                        .depth = 64,
                    },
                    {
                        .plane = 0,
                        .step = 3,
                        .offset = 1,
                        .shift = 0,
                        .depth = 64,
                    },
                    {
                        .plane = 0,
                        .step = 3,
                        .offset = 2,
                        .shift = 0,
                        .depth = 64,
                    },
                },
        };
        pixDesc[static_cast<int>(EPixelFormat::RGBA_DOUBLE)] = {
            .nbChannels = 4,
            .dataType = EResourceDataType::Double,
            .isPlannar = false,
            .isYUV = false,
            .isRGB = true,
            .log2ChromaW = 0,
            .log2ChromaH = 0,
            .components =
                {
                    {
                        .plane = 0,
                        .step = 4,
                        .offset = 0,
                        .shift = 0,
                        .depth = 64,
                    },
                    {
                        .plane = 0,
                        .step = 4,
                        .offset = 1,
                        .shift = 0,
                        .depth = 64,
                    },
                    {
                        .plane = 0,
                        .step = 4,
                        .offset = 2,
                        .shift = 0,
                        .depth = 64,
                    },
                    {
                        .plane = 0,
                        .step = 4,
                        .offset = 3,
                        .shift = 0,
                        .depth = 64,
                    },
                },
        };
        pixDesc[static_cast<int>(EPixelFormat::ARGB_DOUBLE)] = pixDesc[static_cast<int>(EPixelFormat::RGBA_DOUBLE)];
        pixDesc[static_cast<int>(EPixelFormat::BGRA_DOUBLE)] = pixDesc[static_cast<int>(EPixelFormat::RGBA_DOUBLE)];
        pixDesc[static_cast<int>(EPixelFormat::ABGR_DOUBLE)] = pixDesc[static_cast<int>(EPixelFormat::RGBA_DOUBLE)];
        pixDesc[static_cast<int>(EPixelFormat::YUV420P_I8)] = {
            .nbChannels = 3,
            .dataType = EResourceDataType::Int8,
            .isPlannar = false,
            .isYUV = true,
            .isRGB = false,
            .log2ChromaW = 1,
            .log2ChromaH = 1,
            .components =
                {
                    {
                        .plane = 0,
                        .step = 1,
                        .offset = 0,
                        .shift = 0,
                        .depth = 8,
                    },
                    {
                        .plane = 1,
                        .step = 1,
                        .offset = 0,
                        .shift = 0,
                        .depth = 8,
                    },
                    {
                        .plane = 2,
                        .step = 1,
                        .offset = 0,
                        .shift = 0,
                        .depth = 8,
                    },
                },
        };
        pixDesc[static_cast<int>(EPixelFormat::YUV420P_U8)] = {
            .nbChannels = 3,
            .dataType = EResourceDataType::UInt8,
            .isPlannar = false,
            .isYUV = true,
            .isRGB = false,
            .log2ChromaW = 1,
            .log2ChromaH = 1,
            .components =
                {
                    {
                        .plane = 0,
                        .step = 1,
                        .offset = 0,
                        .shift = 0,
                        .depth = 8,
                    },
                    {
                        .plane = 1,
                        .step = 1,
                        .offset = 0,
                        .shift = 0,
                        .depth = 8,
                    },
                    {
                        .plane = 2,
                        .step = 1,
                        .offset = 0,
                        .shift = 0,
                        .depth = 8,
                    },
                },
        };
        pixDesc[static_cast<int>(EPixelFormat::NV12_U8)] = {
            .nbChannels = 2,
            .dataType = EResourceDataType::UInt8,
            .isPlannar = false,
            .isYUV = true,
            .isRGB = false,
            .log2ChromaW = 1,
            .log2ChromaH = 1,
            .components =
                {
                    {
                        .plane = 0,
                        .step = 1,
                        .offset = 0,
                        .shift = 0,
                        .depth = 8,
                    },
                    {
                        .plane = 1,
                        .step = 2,
                        .offset = 0,
                        .shift = 0,
                        .depth = 8,
                    },
                    {
                        .plane = 1,
                        .step = 2,
                        .offset = 1,
                        .shift = 0,
                        .depth = 8,
                    },
                },
        };
    }
} _initialiser;
} // namespace

PixelDesc* getPixelDesc(EPixelFormat pixFmt)
{
    if (static_cast<int>(pixFmt) >= static_cast<int>(EPixelFormat::MAX_PIX_FMT))
    {
        return nullptr;
    }
    return &pixDesc[static_cast<int>(pixFmt)];
}
