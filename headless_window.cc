#include "headless_window.h"
#include "scene_controller.h"
#include "headless_canvas.h"

HeadlessWindow::HeadlessWindow(int w, int h) : width(w), height(h)
{

}

int HeadlessWindow::Width() const noexcept
{
    return width;
}

int HeadlessWindow::Height() const noexcept
{
    return height;
}

int HeadlessWindow::Exec()
{
    auto scene = std::make_unique<SceneController<HeadlessCanvas>>(HeadlessCanvas{*this});
    scene->SetHWND(GetNativeHandle());
    scene->CreateBuffer(0);
    scene->MouseDown();
    scene->MouseMove(0, 0);
    scene->MouseMove(100, 100);
    scene->MouseUp();
    scene->Render(0);
    scene->Canvas().Save("a.png");
    return 0;
}

NativeWindowHandle HeadlessWindow::GetNativeHandle()
{
    return this;
}