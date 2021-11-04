#pragma once
#include "object.h"

class PlatformFactory
{
private:
    using TypeId = int;
    using ProviderType = std::function<std::shared_ptr<Object>()>;
public:
    PlatformFactory& Instance()
    {
        static PlatformFactory ret;
        return ret;
    }

    template<typename TProvider, typename TInterface, typename T = typename std::remove_reference_t<decltype(*std::declval<TProvider>()())>>
    typename std::enable_if_t<
        std::is_base_of_v<T, Object> &&
        std::is_base_of_v<T, TInterface>, 
    PlatformFactory&>
    Register(TProvider&& provider)
    {
        providers[GetTypeId<TInterface>()] = std::forward<TProvider>(provider);
    }

    template<typename TInterface>
    std::shared_ptr<TInterface> Resolve()
    {
        providers[GetTypeId<TInterface>()]();
    }
private:
    PlatformFactory() = default;

    static inline TypeId curId = 1;
    template<typename T>
    static TypeId GetTypeId()
    {
        static TypeId id = curId++;
        return id;
    }
private:
    std::unordered_map<TypeId, ProviderType> providers;
};
