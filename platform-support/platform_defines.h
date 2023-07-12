#pragma once

#ifdef WIN32
#include "win32_canvas.h"
using CanvasType = Win32Canvas;
#else
#include "headless_canvas.h"
using CanvasType = HeadlessCanvas;
#endif