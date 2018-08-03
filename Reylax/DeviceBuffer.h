#pragma once
#include "ReylaxTypes.h"

namespace Reylax
{
    struct DeviceBuffer
    {
    public:
        DeviceBuffer(u32 size);
        ~DeviceBuffer();
        void copyTo(void* buffer, bool wait);
        void copyFrom(const void* buffer, bool wait);
        template <typename T>
        inline T* ptr() const { return reinterpret_cast<T*>(m_devData); }
        u32 size () const { return m_size; }

    private:
        void* m_devData;
        u32 m_size;
    };
}
