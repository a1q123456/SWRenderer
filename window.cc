#include "window.h"
#include "win32exception.h"
#include <chrono>

LRESULT CALLBACK Window::WndProc(
    _In_ HWND hWnd,
    _In_ UINT message,
    _In_ WPARAM wParam,
    _In_ LPARAM lParam)
{
    auto self = reinterpret_cast<Window *>(lParam);

    PAINTSTRUCT ps;
    HDC hdc;

    switch (message)
    {
    case WM_PAINT:
        hdc = BeginPaint(hWnd, &ps);
        EndPaint(hWnd, &ps);
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

Window::~Window()
{
    stop = true;
    try
    {
        renderTh.join();
    } catch (...) {}
}

Window::Window(
    HINSTANCE hInstance,
    HINSTANCE hPrevInstance,
    LPSTR lpCmdLine,
    int nShowCmd,
    const TString &title,
    const TString &windowClass) : mTitle(title), mWindowClass(windowClass)
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
    mWcex.lpszClassName = mWindowClass.c_str();
    mWcex.hIconSm = LoadIcon(mWcex.hInstance, IDI_APPLICATION);
    if (!RegisterClassEx(&mWcex))
    {
        throw std::system_error(GetLastError(), win32_error_category());
    }
    hWnd = CreateWindowEx(
        WS_EX_OVERLAPPEDWINDOW,
        mWindowClass.c_str(),
        mTitle.c_str(),
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT,
        width, height,
        NULL,
        NULL,
        hInstance,
        this);
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
    renderer = std::make_unique<SWRenderer>(mHdc, width, height);
    renderer->CreateBuffer(iPixelFormat);

    renderTh = std::thread{[=]()
                           {
                               auto lastTime = std::chrono::steady_clock::now();
                               while (!stop)
                               {
                                   renderer->SwapBuffer();
                                   auto now = std::chrono::steady_clock::now();
                                   auto dur = now - lastTime;
                                   lastTime = now;
                                   auto t = dur.count() / 1'000'000'000.0;
                                   renderer->Render(t);
                                   HDC src = CreateCompatibleDC(mHdc);      // hdc - Device context for window, I've got earlier with GetDC(hWnd) or GetDC(NULL);
                                   SelectObject(src, renderer->GetBitmap()); // Inserting picture into our temp HDC
                                   BitBlt(mHdc,                             // Destination
                                          0,                                // x and
                                          0,                                // y - upper-left corner of place, where we'd like to copy
                                          width,                            // width of the region
                                          height,                           // height
                                          src,                              // source
                                          0,                                // x and
                                          0,                                // y of upper left corner  of part of the source, from where we'd like to copy
                                          SRCCOPY);                         // Defined DWORD to juct copy pixels. Watch more on msdn;

                                   DeleteDC(src); // Deleting temp HDC
                               }
                           }};
}


int Window::Width() const
{
    return width;
}

int Window::Height() const
{
    return height;
}

int Window::Exec()
{
    MSG msg;
    while (GetMessage(&msg, NULL, 0, 0))
    {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }

    return (int)msg.wParam;
}
