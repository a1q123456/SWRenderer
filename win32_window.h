#pragma once
#include "iwindow.h"
#include "tstring.h"
#include "scene_controller.h"
#include "win32_canvas.h"


class Win32Window
{
    TString mTitle;
    TString mWin32WindowClass;
    HWND hWnd;
    WNDCLASSEX mWcex;
    int width = 500;
    int height = 500;
    HDC mHdc;
    std::unique_ptr<SceneController<Win32Canvas>> scene;
    std::chrono::steady_clock::time_point lastTime = std::chrono::steady_clock::now();
    static LRESULT CALLBACK WndProc(
        _In_ HWND hWnd,
        _In_ UINT message,
        _In_ WPARAM wParam,
        _In_ LPARAM lParam);
    void Render();
public:
    ~Win32Window();
    Win32Window(
        HINSTANCE hInstance,
        HINSTANCE hPrevInstance,
        LPSTR lpCmdLine,
        int nShowCmd,
        const TString &title = _T("Software Renderer"),
        const TString &windowClass = _T("SWRenderer"));
    Win32Window(const Win32Window &) = delete;
    Win32Window &operator=(const Win32Window &) = delete;
    int Width() const;
    int Height() const;
    int Exec();
};
