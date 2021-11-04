#pragma once
#include <string>
#ifdef _WIN32
#include <Windows.h>
#include <tchar.h>
#else
using TCHAR = char;
#endif

using TString = std::basic_string<TCHAR>;
