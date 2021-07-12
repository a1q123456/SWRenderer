#pragma once
#include "tstring.h"
#include "swrenderer.h"

class Window
{
    TString mTitle;
    TString mWindowClass;
    HWND hWnd;
    WNDCLASSEX mWcex;
    int width = 1620;
    int height = 937;
    HDC mHdc;
    std::unique_ptr<SWRenderer> renderer;
    std::thread renderTh;

    static LRESULT CALLBACK WndProc(
        _In_ HWND hWnd,
        _In_ UINT message,
        _In_ WPARAM wParam,
        _In_ LPARAM lParam);
public:
    Window(
        HINSTANCE hInstance,
        HINSTANCE hPrevInstance,
        LPSTR lpCmdLine,
        int nShowCmd,
        const TString &title = _T("Windows Desktop Guided Tour Application"),
        const TString &windowClass = _T("DesktopApp"));
    Window(const Window&) = delete;
    Window& operator=(const Window&) = delete;
    int Width() const;
    int Height() const;;
    int Exec();
};
