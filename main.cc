#include "window.h"

int WinMain(
    HINSTANCE hInstance,
    HINSTANCE hPrevInstance,
    LPSTR lpCmdLine,
    int nShowCmd)
{
    try
    {
        Window wnd{hInstance, hPrevInstance, lpCmdLine, nShowCmd};
        return wnd.Exec();
    }
    catch (const std::exception& e)
    {
        MessageBoxA(nullptr, e.what(), "Error", MB_OK);
    }
    return -1;
}
