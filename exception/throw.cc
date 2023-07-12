#include "throw.h"
#include <exception>
#include <stdexcept>

class MisalignedMemoryAccess : public std::logic_error
{
public:
    using std::logic_error::logic_error;
};

void ThrowException(SWRErrorCode ec, const char* message)
{
    switch (ec)
    {
        case SWRErrorCode::MisalignedMemoryAccess:
            throw MisalignedMemoryAccess{message};
    }
}

void ThrowException(SWRErrorCode ec, const std::string& message)
{
    switch (ec)
    {
        case SWRErrorCode::MisalignedMemoryAccess:
            throw MisalignedMemoryAccess{message};
    }
}

void ThrowException(SWRErrorCode ec)
{
    switch (ec)
    {
        case SWRErrorCode::MisalignedMemoryAccess:
            throw MisalignedMemoryAccess{"mis-aligned memory access"};
    }
}
