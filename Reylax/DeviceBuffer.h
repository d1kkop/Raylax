#pragma once
#include "Reylax.h"

namespace Reylax
{
    struct DeviceBuffer: public IDeviceBuffer
    {
    public:
        DeviceBuffer(u32 size);
        ~DeviceBuffer();
        void copyTo(void* buffer, bool wait) override;
        void copyFrom(const void* buffer, bool wait) override;
        u32  size() const override { return m_size; };
        void* ptr() const override { return m_devData; };

        template <typename T>
        inline T* ptr() const { return reinterpret_cast<T*>(m_devData); }

    private:
        void* m_devData;
        u32   m_size;
    };
}
