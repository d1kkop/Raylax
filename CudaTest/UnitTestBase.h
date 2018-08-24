#pragma once
#include <string>
#include <vector>
#include <iostream>
#include <chrono>
using namespace std;
using namespace chrono;


namespace UnitTest
{
    double Time();

	class UnitTestBase
	{
	public:
        UnitTestBase(const char* name);
		virtual bool run() = 0;
		std::string name() { return m_Name; }
		std::string m_Name;
        static void processAll();
	};
}


#define UTESTBEGIN(name) \
namespace UnitTest\
{\
	class UTest##name : public UnitTestBase \
	{\
public:\
		UTest##name(): UnitTestBase(#name) { } \
		bool run() override


#define UNITTESTEND(name)\
	};\
}\
static UnitTest::UTest##name st_##name;