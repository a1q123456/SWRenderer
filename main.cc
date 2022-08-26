#include "iwindow.h"
#include "platform_factory.h"
#include "headless_window.h"
#ifdef _WIN32
#include "win32_window.h"
#endif

#include "texture.h"

int 
#ifdef _WIN32
WinMain(
    HINSTANCE hInstance,
    HINSTANCE hPrevInstance,
    LPSTR lpCmdLine,
    int nShowCmd)
#else
main()
#endif
{    
#if defined(_WIN32) && !defined(SWR_HEADLESS_RENDERING)
    Win32Window wnd{hInstance, hPrevInstance, lpCmdLine, nShowCmd};
#else
    HeadlessWindow wnd{500, 500};
#endif
    return wnd.Exec();
}
