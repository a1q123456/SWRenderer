#pragma once
#include <stdexcept>
#include <driver_types.h>
#include <cuda/std/utility>
#include <cuda.h>
#include <cuda_runtime_api.h>

#ifdef __INTELLISENSE__ 
#define __global__
#define __device__
struct
{
    int x, y;
} threadIdx, blockIdx, blockDim;
#endif

#ifndef __CUDA_ARCH__ 
inline void CudaThrowIfFailed(cudaError_t error)
{
    if (error != cudaSuccess)
    {
        throw std::runtime_error{cudaGetErrorString(error)};
    }
}
#else
inline void CudaThrowIfFailed(cudaError_t error) {}
#endif

template<typename T>
class CudaPointer
{
public:
    using element_type = std::remove_reference_t<T>;
    using const_element_type = std::add_const_t<element_type>;
    using pointer_type = std::add_pointer_t<std::decay_t<T>>;
    using const_pointer_type = std::add_const_t<pointer_type>;
private:
    pointer_type ptr = nullptr;

    __device__ __host__ void Reset(pointer_type val)
    {
        if (ptr != nullptr)
        {
            ptr->~T();
            cudaFree(ptr);
        }

        ptr = val;
    }

    __device__ __host__ void MoveFrom(pointer_type& val)
    {
        Reset(nullptr);
        cuda::std::swap(val, ptr);
    }
public:
    __device__ __host__ CudaPointer(pointer_type val = nullptr) : ptr{val} { }

    CudaPointer(const CudaPointer&) = delete;
    __device__ __host__ CudaPointer(CudaPointer&& rhs) noexcept
    {
        MoveFrom(rhs.ptr);
    }

    CudaPointer& operator=(const CudaPointer&) = delete;
    __device__ __host__ CudaPointer& operator=(CudaPointer&& rhs) noexcept
    {
        MoveFrom(rhs.ptr);
        return *this;
    }

    __device__ __host__ pointer_type operator*()
    {
        return get();
    }

    __device__ __host__ const_pointer_type operator*() const
    {
        return get();
    }

    __device__ __host__ pointer_type operator->()
    {
        return get();
    }

    __device__ __host__ const_pointer_type operator->() const
    {
        return get();
    }

    __device__ __host__ pointer_type get() noexcept
    {
        return ptr;
    }

    __device__ __host__ const_pointer_type get() const noexcept
    {
        return ptr;
    }

    __device__ __host__ void reset(pointer_type val)
    {
        Reset(val);
    }

    __device__ __host__ ~CudaPointer() noexcept
    {
        Reset(nullptr);
    }
};

template<typename T>
class CudaPointer<T[]>
{
public:
    using element_type = std::remove_reference_t<T>;
    using const_element_type = std::add_const_t<element_type>;
    using pointer_type = std::add_pointer_t<element_type>;
    using const_pointer_type = std::add_const_t<pointer_type>;
private:
    pointer_type ptr = nullptr;

    __device__ __host__ void Reset(pointer_type val)
    {
        if (ptr != nullptr)
        {
            ptr->~T();
            cudaFree(ptr);
        }

        ptr = val;
    }

    __device__ __host__ void MoveFrom(pointer_type& val)
    {
        Reset(nullptr);
        cuda::std::swap(val, ptr);
    }
public:
    __device__ __host__ CudaPointer(pointer_type val = nullptr) : ptr{val} { }

    CudaPointer(const CudaPointer&) = delete;
    __device__ __host__ CudaPointer(CudaPointer&& rhs) noexcept
    {
        MoveFrom(rhs.ptr);
    }

    CudaPointer& operator=(const CudaPointer&) = delete;
    __device__ __host__ CudaPointer& operator=(CudaPointer&& rhs) noexcept
    {
        MoveFrom(rhs.ptr);
        return *this;
    }

    __device__ __host__ pointer_type operator*()
    {
        return get();
    }

    __device__ __host__ const_pointer_type operator*() const
    {
        return get();
    }

    __device__ __host__ pointer_type operator->()
    {
        return get();
    }

    __device__ __host__ const_pointer_type operator->() const
    {
        return get();
    }

    __device__ __host__ element_type operator[](std::size_t index)
    {
        return get()[index];
    }

    __device__ __host__ const_element_type operator[](std::size_t index) const
    {
        return get()[index];
    }

    __device__ __host__ pointer_type get() noexcept
    {
        return ptr;
    }

    __device__ __host__ const_pointer_type get() const noexcept
    {
        return ptr;
    }

    __device__ __host__ void reset(pointer_type val)
    {
        Reset(val);
    }

    __device__ __host__ ~CudaPointer() noexcept
    {
        Reset(nullptr);
    }
};

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
