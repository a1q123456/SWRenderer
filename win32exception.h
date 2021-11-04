#pragma once

class Win32ExceptionCategory : public std::error_category
{
public:
    virtual const char *name() const noexcept;
    virtual std::string message(int _Errval) const;
};

std::error_category& win32_error_category();

