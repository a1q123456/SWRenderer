#pragma once
#include "tstring.h"
#include "swrenderer.h"

class Window
{
    TString mTitle;
    TString mWindowClass;
    HWND hWnd;
    WNDCLASSEX mWcex;
    int width = 500;
    int height = 500;
    HDC mHdc;
    std::unique_ptr<SWRenderer> renderer;
    std::thread renderTh;
    volatile bool stop = false;

    static LRESULT CALLBACK WndProc(
        _In_ HWND hWnd,
        _In_ UINT message,
        _In_ WPARAM wParam,
        _In_ LPARAM lParam);
public:
    ~Window();
    Window(
        HINSTANCE hInstance,
        HINSTANCE hPrevInstance,
        LPSTR lpCmdLine,
        int nShowCmd,
        const TString &title = _T("Software Renderer"),
        const TString &windowClass = _T("SWRenderer"));
    Window(const Window&) = delete;
    Window& operator=(const Window&) = delete;
    int Width() const;
    int Height() const;;
    int Exec();
};
