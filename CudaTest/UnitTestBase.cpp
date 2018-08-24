#include "UnitTestBase.h"


namespace UnitTest
{
    double Time()
    { 
        return static_cast<double>(duration_cast<duration<double, milli>>(high_resolution_clock::now().time_since_epoch()).count()); 
    }

    std::vector<UnitTestBase*> g_UnitTests;

    UnitTestBase::UnitTestBase(const char* name):
        m_Name(name)
    {
        g_UnitTests.push_back(this);
    }

    void UnitTestBase::processAll()
    {
        printf("Processing unit tests..\n");
        double startTime = Time();
        for ( auto* ut : g_UnitTests )
        {
            double ts = Time();
            bool b = ut->run();
            double te = Time();
            if ( b ) printf("%s PASSED, time: %fms\n", ut->m_Name.c_str(), (te-ts));
            else     printf("%s FAILED, time: %fms\n", ut->m_Name.c_str(), (te-ts));
        }
        auto endTime = Time();
        printf("Total time is %fms.\n", (endTime -startTime));
    }
}