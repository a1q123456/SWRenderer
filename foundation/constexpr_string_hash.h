#pragma once
#include <type_traits>

namespace ConstexprHash
{
    namespace _detail
    {
        static constexpr size_t P = 53;
        static constexpr size_t M = 1e9 + 9;

        template <size_t N>
        constexpr size_t _Get_Hash_Char(const char (&str)[N], size_t idx, size_t pVal)
        {
            return (static_cast<size_t>(str[idx])) * pVal;
        }

        template <size_t Idx, size_t N>
        constexpr typename std::enable_if_t<Idx == (N - 1), size_t> _Get_Hash(const char (&str)[N], size_t pVal)
        {
            return 0;
        }

        template <size_t Idx, size_t N>
        constexpr typename std::enable_if_t<Idx != (N - 1), size_t> _Get_Hash(const char (&str)[N], size_t pVal)
        {
            return _Get_Hash_Char(str, Idx, pVal) + _Get_Hash<Idx + 1>(str, pVal * P);
        }

    }

    template <size_t N>
    constexpr size_t HashString(const char (&str)[N])
    {
        return _detail::_Get_Hash<0>(str, 1) % _detail::M;
    }
}