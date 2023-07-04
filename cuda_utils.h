#pragma once
#ifdef __INTELLISENSE__ 
#define __global__
#define __device__
struct
{
    int x, y;
} threadIdx, blockIdx, blockDim;
#endif

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

template<typename T, typename... Args>
CudaPointer<T[]> CudaNewArray(size_t n)
{
    T* ret = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&ret), n * sizeof(T)) != cudaSuccess)
    {
        throw std::bad_alloc{};
    }
    return CudaPointer<T[]>{ret};
}

inline void CudaThrowIfFailed(cudaError_t error)
{
    if (error != cudaSuccess)
    {
        throw std::runtime_error{cudaGetErrorString(error)};
    }
}
