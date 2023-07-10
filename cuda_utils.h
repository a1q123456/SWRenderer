#pragma once
#include <memory>
#include <stdexcept>

#ifdef __INTELLISENSE__ 
#define __global__
#define __device__
struct
{
    int x, y;
} threadIdx, blockIdx, blockDim;
#endif

inline void CudaThrowIfFailed(cudaError_t error)
{
    if (error != cudaSuccess)
    {
        throw std::runtime_error{cudaGetErrorString(error)};
    }
}

struct CudaDeleter
{
    template<typename T>
    void operator()(T* ptr)
    {
        ptr->~T();
        cudaFree(ptr);
    }
};

template<typename T>
using CudaPointer = std::unique_ptr<T, CudaDeleter>;

template<typename T, typename... Args>
CudaPointer<T> CudaNewManaged(Args&&... args)
{
    T* ret = nullptr;
    if (cudaMallocManaged(reinterpret_cast<void**>(&ret), sizeof(T)) != cudaSuccess)
    {
        throw std::bad_alloc{};
    }
    new(ret) T{std::forward<Args>(args)...};
    return CudaPointer<T>{ret};
}

template<typename T, typename... Args>
CudaPointer<T> CudaNew(Args&&... args)
{
    T* ret = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&ret), sizeof(T)) != cudaSuccess)
    {
        throw std::bad_alloc{};
    }
    new(ret) T{std::forward<Args>(args)...};
    return CudaPointer<T>{ret};
}

template<typename T>
CudaPointer<T[]> CudaUploadData(const T* data, size_t n)
{
    static_assert(std::is_trivially_copyable_v<T>);
    T* ret = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&ret), sizeof(T) * n) != cudaSuccess)
    {
        throw std::bad_alloc{};
    }
    CudaThrowIfFailed(cudaMemcpy(ret, data, sizeof(T) * n, cudaMemcpyHostToDevice));
    return CudaPointer<T[]>{ret};
}

template<typename T>
CudaPointer<T[]> CudaNewArray(size_t n)
{
    static_assert(std::is_trivial_v<T> && std::is_trivially_destructible_v<T>);

    T* ret = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&ret), n * sizeof(T)) != cudaSuccess)
    {
        throw std::bad_alloc{};
    }
    return CudaPointer<T[]>{ret};
}
