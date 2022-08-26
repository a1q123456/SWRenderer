#pragma once

enum class SWRErrorCode
{
    MisalignedMemoryAccess,
    TypeMismatch,
    IndexOutOfRange,
};


[[noreturn]] void ThrowException(SWRErrorCode ec);
[[noreturn]] void ThrowException(SWRErrorCode ec, const char* message);
[[noreturn]] void ThrowException(SWRErrorCode ec, const std::string& message);