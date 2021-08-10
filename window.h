#pragma once
#include "tstring.h"
#include "scene_controller.h"


class Window
{
    TString mTitle;
    TString mWindowClass;
    HWND hWnd;
    WNDCLASSEX mWcex;
    int width = 500;
    int height = 500;
    HDC mHdc;
    std::unique_ptr<SceneController> scene;
    std::thread renderTh;
    volatile bool stop = false;
    std::chrono::steady_clock::time_point lastTime = std::chrono::steady_clock::now();
    static LRESULT CALLBACK WndProc(
        _In_ HWND hWnd,
        _In_ UINT message,
        _In_ WPARAM wParam,
        _In_ LPARAM lParam);
    void Render();
public:
    ~Window();
    Window(
        HINSTANCE hInstance,
        HINSTANCE hPrevInstance,
        LPSTR lpCmdLine,
        int nShowCmd,
        const TString &title = _T("Software Renderer"),
        const TString &windowClass = _T("SWRenderer"));
    Window(const Window &) = delete;
    Window &operator=(const Window &) = delete;
    int Width() const;
    int Height() const;
    int Exec();
};
