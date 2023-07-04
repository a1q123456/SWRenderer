#pragma once

template <typename T> class CudaManagedAllocator
{
public:
    using value_type = T;
    using is_always_equal = std::true_type;

    CudaManagedAllocator() = default;

    template <typename U> constexpr CudaManagedAllocator(const CudaManagedAllocator<U>&) noexcept
    {
    }

    [[nodiscard]] constexpr T* allocate(std::size_t n)
    {
        void* ret = nullptr;
        auto err = cudaMallocManaged(&ret, n * sizeof(T));
        if (err != cudaSuccess)
        {
            throw std::bad_alloc{};
        }
        return reinterpret_cast<T*>(ret);
    }

    constexpr void deallocate(T* p, std::size_t n)
    {
        cudaFree(p);
    }
};

template <typename T> bool operator==(const CudaManagedAllocator<T>& a, const CudaManagedAllocator<T>& b)
{
    return true;
}

template <typename T> bool operator!=(const CudaManagedAllocator<T>& a, const CudaManagedAllocator<T>& b)
{
    return false;
}
