#include "iwindow.h"
#include "platform_factory.h"
#include "win32_window.h"

#include "texture.h"

int WinMain(
    HINSTANCE hInstance,
    HINSTANCE hPrevInstance,
    LPSTR lpCmdLine,
    int nShowCmd)
{
    Win32Window wnd{hInstance, hPrevInstance, lpCmdLine, nShowCmd};
    return wnd.Exec();
}
