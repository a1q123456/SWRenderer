#include "win32_window.h"
#include <WinUser.h>
#include <windowsx.h>
#include "win32exception.h"
#include <chrono>


LRESULT CALLBACK Win32Window::WndProc(
    _In_ HWND hWnd,
    _In_ UINT message,
    _In_ WPARAM wParam,
    _In_ LPARAM lParam)
{
    auto self = reinterpret_cast<Win32Window *>(GetWindowLongPtr(hWnd, GWLP_USERDATA));

    PAINTSTRUCT ps;
    HDC hdc;

    switch (message)
    {
    case WM_PAINT:
    {
        hdc = BeginPaint(hWnd, &ps);
        EndPaint(hWnd, &ps);
    }
    break;
    case WM_LBUTTONDOWN:
        if (self == nullptr)
        {
            break;
        }
        self->scene->MouseDown();
        break;
    case WM_LBUTTONUP:
        if (self == nullptr)
        {
            break;
        }
        self->scene->MouseUp();
        break;
    case WM_MOUSEMOVE:
        if (self == nullptr)
        {
            break;
        }
        self->scene->MouseMove(GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam));
        break;
    case WM_MOUSEWHEEL:
        if (self == nullptr)
        {
            break;
        }
        self->scene->MouseWheel(GET_WHEEL_DELTA_WPARAM(wParam));
        break;
    case WM_DESTROY:
        PostQuitMessage(0);
        break;
    default:
        return DefWindowProc(hWnd, message, wParam, lParam);
        break;
    }
    return 0;
}

void Win32Window::Render()
{
    auto now = std::chrono::steady_clock::now();
    auto dur = now - lastTime;
    lastTime = now;
    auto t = dur.count() * decltype(dur)::duration::period::num / static_cast<float>(decltype(dur)::duration::period::den);
    scene->Render(t);
    HDC src = CreateCompatibleDC(mHdc);       // hdc - Device context for window, I've got earlier with GetDC(hWnd) or GetDC(NULL);
    SelectObject(src, scene->Canvas().Bitmap()); // Inserting picture into our temp HDC
    BitBlt(mHdc,                              // Destination
           0,                                 // x and
           0,                                 // y - upper-left corner of place, where we'd like to copy
           width,                             // width of the region
           height,                            // height
           src,                               // source
           0,                                 // x and
           0,                                 // y of upper left corner  of part of the source, from where we'd like to copy
           SRCCOPY);                          // Defined DWORD to juct copy pixels. Watch more on msdn;

    DeleteDC(src); // Deleting temp HDC
}

Win32Window::~Win32Window()
{
}

Win32Window::Win32Window(
    HINSTANCE hInstance,
    HINSTANCE hPrevInstance,
    LPSTR lpCmdLine,
    int nShowCmd,
    const TString &title,
    const TString &windowClass) : mTitle(title), mWin32WindowClass(windowClass)
{

    mWcex.cbSize = sizeof(WNDCLASSEX);
    mWcex.style = CS_HREDRAW | CS_VREDRAW;
    mWcex.lpfnWndProc = WndProc;
    mWcex.cbClsExtra = 0;
    mWcex.cbWndExtra = 0;
    mWcex.hInstance = hInstance;
    mWcex.hIcon = LoadIcon(mWcex.hInstance, IDI_APPLICATION);
    mWcex.hCursor = LoadCursor(NULL, IDC_ARROW);
    mWcex.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
    mWcex.lpszMenuName = NULL;
    mWcex.lpszClassName = mWin32WindowClass.c_str();
    mWcex.hIconSm = LoadIcon(mWcex.hInstance, IDI_APPLICATION);
    if (!RegisterClassEx(&mWcex))
    {
        throw std::system_error(GetLastError(), win32_error_category());
    }
    hWnd = CreateWindowEx(
        WS_EX_OVERLAPPEDWINDOW,
        mWin32WindowClass.c_str(),
        mTitle.c_str(),
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT,
        width, height,
        NULL,
        NULL,
        hInstance,
        nullptr);

    SetWindowLongPtr(hWnd, GWLP_USERDATA, (LONG_PTR)this);
    if (!hWnd)
    {
        throw std::system_error(GetLastError(), win32_error_category());
    }
    PIXELFORMATDESCRIPTOR pfd = {
        sizeof(PIXELFORMATDESCRIPTOR), // size of this pfd
        1,                             // version number
        PFD_SUPPORT_GDI |
            PFD_DRAW_TO_WINDOW | // support window
            PFD_TYPE_RGBA,       // RGBA type
        32,                      // 24-bit color depth
        0, 0, 0, 0, 0, 0,        // color bits ignored
        0,                       // no alpha buffer
        0,                       // shift bit ignored
        0,                       // no accumulation buffer
        0, 0, 0, 0,              // accum bits ignored
        0,                       // 32-bit z-buffer
        0,                       // no stencil buffer
        0,                       // no auxiliary buffer
        PFD_MAIN_PLANE,          // main layer
        0,                       // reserved
        0, 0, 0                  // layer masks ignored
    };
    mHdc = GetDC(hWnd);
    int iPixelFormat;

    // get the best available match of pixel format for the device context
    iPixelFormat = ChoosePixelFormat(mHdc, &pfd);
    // make that the pixel format of the device context
    SetPixelFormat(mHdc, iPixelFormat, &pfd);
    ShowWindow(hWnd,
               nShowCmd);
    UpdateWindow(hWnd);
    scene = std::make_unique<TestSceneController<Win32Canvas>>(Win32Canvas{mHdc, width, height});
    scene->SetHWND(hWnd);
    scene->CreateBuffer(EPixelFormat::RGBA_U8);
}

int Win32Window::Width() const
{
    return width;
}

int Win32Window::Height() const
{
    return height;
}

int Win32Window::Exec()
{
    MSG msg;
    while (true)
    {
        while (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
        {
            if (msg.message == WM_QUIT)
            {
                break;
            }
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
        if (msg.message == WM_QUIT)
        {
            break;
        }
        Render();
    }
    return (int)msg.wParam;
}
