#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <format>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <limits>
#include <numbers>
#include <string>
#include <vector>

#include "OneResult32.hpp"
#include "colors.hpp"
#include "cuda_check.hpp"
#include "custom_asm.hpp"
#include "readbinary.hpp"

struct BinaryCase {
  float a;
  float b;
  const char *alabel;
  const char *blabel;
};

// clang-format off
static constexpr BinaryCase kCases[] = {
  // ---------------------------------------------------------------------
  // From compareb.hip (adjacent input pairs)
  // ---------------------------------------------------------------------
  {1.5f, 1.5001f, "1.5", "1.5001"},
  {1.5001f, -0.0f, "1.5001", "-0"},
  {-0.0f, +0.0f, "-0", "+0"},
  {+0.0f, std::numeric_limits<float>::infinity(), "+0", "+inf"},
  {std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(), "+inf", "+inf"},
  {std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), "+inf", "-inf"},
  {-std::numeric_limits<float>::infinity(), -1.5f, "-inf", "-1.5"},
  {-1.5f, std::numeric_limits<float>::max(), "-1.5", "FLT_MAX"},
  {std::numeric_limits<float>::max(), std::numeric_limits<float>::infinity(), "FLT_MAX", "+inf"},
  {std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN(), "+inf", "qNaN"},
  {std::numeric_limits<float>::quiet_NaN(), 1.0f, "qNaN", "1"},
  {1.0f, 1.0f, "1", "1"},
  {1.0f, std::numeric_limits<float>::quiet_NaN(), "1", "qNaN"},
  {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::min(), "qNaN", "FLT_MIN"},
  {std::numeric_limits<float>::min(), std::numeric_limits<float>::denorm_min(), "FLT_MIN", "denorm_min"},

  // ---------------------------------------------------------------------
  // From div_test.hip
  // ---------------------------------------------------------------------
  {4.0f, 2.0f, "4", "2"},
  {10.0f, 5.0f, "10", "5"},
  {100.0f, 10.0f, "100", "10"},
  {1.0f, 2.0f, "1", "2"},
  {1.0f, 4.0f, "1", "4"},
  {3.0f, 4.0f, "3", "4"},
  {1.0f, 8.0f, "1", "8"},
  {1.0f, 16.0f, "1", "16"},
  {16.0f, 4.0f, "16", "4"},
  {7.0f, 7.0f, "7", "7"},
  {0.123f, 0.123f, "0.123", "0.123"},
  {100.0f, 1.0f, "100", "1"},
  {0.001f, 1.0f, "0.001", "1"},
  {1.0f, 10.0f, "1", "10"},
  {1.0f, 100.0f, "1", "100"},
  {1.0f, 0.5f, "1", "0.5"},
  {1.0f, 0.1f, "1", "0.1"},
  {1.0f, 3.0f, "1", "3"},
  {2.0f, 3.0f, "2", "3"},
  {1.0f, 6.0f, "1", "6"},
  {1.0f, 7.0f, "1", "7"},
  {1.0f, 9.0f, "1", "9"},
  {1.0f, 11.0f, "1", "11"},
  {1.0f, 12.0f, "1", "12"},
  {1.0f, 13.0f, "1", "13"},
  {1.0f, 14.0f, "1", "14"},
  {1.0f, 15.0f, "1", "15"},
  {1.0f, 17.0f, "1", "17"},
  {1.0f, 49.0f, "1", "49"},
  {1.0f, 51.0f, "1", "51"},
  {1.0f, 97.0f, "1", "97"},
  {1.0f, 101.0f, "1", "101"},
  {1.0f, 127.0f, "1", "127"},
  {1.0f, 255.0f, "1", "255"},
  {2.0f, 7.0f, "2", "7"},
  {3.0f, 7.0f, "3", "7"},
  {4.0f, 7.0f, "4", "7"},
  {5.0f, 7.0f, "5", "7"},
  {6.0f, 7.0f, "6", "7"},
  {2.0f, 15.0f, "2", "15"},
  {4.0f, 15.0f, "4", "15"},
  {7.0f, 3.0f, "7", "3"},
  {10.0f, 3.0f, "10", "3"},
  {22.0f, 7.0f, "22", "7"},
  {0.1f, 0.3f, "0.1", "0.3"},
  {1.0f, 1.5f, "1", "1.5"},
  {1.0f, 2.5f, "1", "2.5"},
  {1.0f, 3.5f, "1", "3.5"},
  {1.0f, 5.5f, "1", "5.5"},
  {1.0f, 6.5f, "1", "6.5"},
  {1.0f, 7.5f, "1", "7.5"},
  {3.0f, 5.5f, "3", "5.5"},
  {5.0f, 6.5f, "5", "6.5"},
  {1000.0f, 7.0f, "1000", "7"},
  {1000.0f, 9.0f, "1000", "9"},
  {1000.0f, 11.0f, "1000", "11"},
  {1000.0f, 13.0f, "1000", "13"},
  {123456.0f, 789.0f, "123456", "789"},
  {355.0f, 113.0f, "355", "113"},
  {103993.0f, 33102.0f, "103993", "33102"},
  {-1.0f, 3.0f, "-1", "3"},
  {-10.0f, 2.0f, "-10", "2"},
  {10.0f, -2.0f, "10", "-2"},
  {-10.0f, -2.0f, "-10", "-2"},
  {0.0f, 1.0f, "0", "1"},
  {0.0f, 10.0f, "0", "10"},
  {0.0f, -5.0f, "0", "-5"},
  {-0.0f, 1.0f, "-0", "1"},
  {1.0f, 0.0f, "1", "0"},
  {-1.0f, 0.0f, "-1", "0"},
  {10.0f, 0.0f, "10", "0"},
  {1.0f, -0.0f, "1", "-0"},
  {-1.0f, -0.0f, "-1", "-0"},
  {0.0f, -0.0f, "0", "-0"},
  {1.0f, std::numeric_limits<float>::infinity(), "1", "+inf"},
  {-1.0f, std::numeric_limits<float>::infinity(), "-1", "+inf"},
  {1.0f, -std::numeric_limits<float>::infinity(), "1", "-inf"},
  {-1.0f, -std::numeric_limits<float>::infinity(), "-1", "-inf"},
  {std::numeric_limits<float>::infinity(), 1.0f, "+inf", "1"},
  {-std::numeric_limits<float>::infinity(), 1.0f, "-inf", "1"},
  {-std::numeric_limits<float>::infinity(), -1.0f, "-inf", "-1"},
  {std::numeric_limits<float>::infinity(), 0.0f, "+inf", "0"},
  {-std::numeric_limits<float>::infinity(), 0.0f, "-inf", "0"},
  {std::numeric_limits<float>::infinity(), -0.0f, "+inf", "-0"},
  {-std::numeric_limits<float>::infinity(), -0.0f, "-inf", "-0"},
  {-std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(), "-inf", "+inf"},
  {-std::numeric_limits<float>::quiet_NaN(), 1.0f, "-NaN", "1"},
  {1.0f, -std::numeric_limits<float>::quiet_NaN(), "1", "-NaN"},
  {std::numeric_limits<float>::signaling_NaN(), 1.0f, "sNaN", "1"},
  {1.0f, std::numeric_limits<float>::signaling_NaN(), "1", "sNaN"},
  {-std::numeric_limits<float>::signaling_NaN(), 1.0f, "-sNaN", "1"},
  {std::bit_cast<float>(0x3f7fffffu), 1.0f, "1-ulp", "1"},
  {std::bit_cast<float>(0x3f800001u), 1.0f, "1+ulp", "1"},
  {1.0f, std::bit_cast<float>(0x3f7fffffu), "1", "1-ulp"},
  {1.0f, std::bit_cast<float>(0x3f800001u), "1", "1+ulp"},
  {1.0e-40f, 1.0f, "1e-40", "1"},
  {1.0f, 1.0e-40f, "1", "1e-40"},
  {1.0e-40f, 1.0e-40f, "1e-40", "1e-40"},
  {1.0e-40f, 2.0f, "1e-40", "2"},
  {std::numeric_limits<float>::denorm_min(), 1.0f, "FLT_TRUE_MIN", "1"},
  {1.0f, std::numeric_limits<float>::denorm_min(), "1", "FLT_TRUE_MIN"},
  {std::numeric_limits<float>::denorm_min(), std::numeric_limits<float>::denorm_min(), "TRUE_MIN", "TRUE_MIN"},
  {std::numeric_limits<float>::min(), 1.0f, "FLT_MIN", "1"},
  {1.0f, std::numeric_limits<float>::min(), "1", "FLT_MIN"},
  {std::numeric_limits<float>::min() * 1.5f, 1.0f, "FLT_MIN*1.5", "1"},
  {1.0f, std::numeric_limits<float>::min() * 1.5f, "1", "FLT_MIN*1.5"},
  {std::numeric_limits<float>::min() * 2.0f, 1.0f, "FLT_MIN*2", "1"},
  {1.0f, std::numeric_limits<float>::min() * 2.0f, "1", "FLT_MIN*2"},
  {std::numeric_limits<float>::min() * 3.0f, 1.0f, "FLT_MIN*3", "1"},
  {1.0f, std::numeric_limits<float>::min() * 3.0f, "1", "FLT_MIN*3"},
  {std::numeric_limits<float>::min() * 4.0f, 1.0f, "FLT_MIN*4", "1"},
  {1.0f, std::numeric_limits<float>::min() * 4.0f, "1", "FLT_MIN*4"},
  {std::numeric_limits<float>::min() * 7.0f, 1.0f, "FLT_MIN*7", "1"},
  {1.0f, std::numeric_limits<float>::min() * 7.0f, "1", "FLT_MIN*7"},
  {std::numeric_limits<float>::min() * 8.0f, 1.0f, "FLT_MIN*8", "1"},
  {1.0f, std::numeric_limits<float>::min() * 8.0f, "1", "FLT_MIN*8"},
  {std::numeric_limits<float>::min() * 16.0f, 1.0f, "FLT_MIN*16", "1"},
  {1.0f, std::numeric_limits<float>::min() * 16.0f, "1", "FLT_MIN*16"},
  {std::numeric_limits<float>::min() * 2.0f, std::numeric_limits<float>::min(), "FLT_MIN*2", "FLT_MIN"},
  {std::numeric_limits<float>::min() * 6.0f, std::numeric_limits<float>::min() * 2.0f, "FLT_MIN*6", "FLT_MIN*2"},
  {std::numeric_limits<float>::min() * 3.0f, std::numeric_limits<float>::min() * 7.0f, "FLT_MIN*3", "FLT_MIN*7"},
  {std::numeric_limits<float>::min() / 1.5f, std::numeric_limits<float>::min() / 2.5f, "FLT_MIN/1.5", "FLT_MIN/2.5"},
  {std::numeric_limits<float>::min() / 2.0f, std::numeric_limits<float>::min() / 2.0f, "FLT_MIN/2", "FLT_MIN/2"},
  {std::numeric_limits<float>::min() / 2.0f, std::numeric_limits<float>::min() / 4.0f, "FLT_MIN/2", "FLT_MIN/4"},
  {std::numeric_limits<float>::min() / 3.0f, std::numeric_limits<float>::min() / 7.0f, "FLT_MIN/3", "FLT_MIN/7"},
  {std::numeric_limits<float>::min() / 4.0f, std::numeric_limits<float>::min() / 6.0f, "FLT_MIN/4", "FLT_MIN/6"},
  {std::numeric_limits<float>::min() / 5.0f, std::numeric_limits<float>::min() / 3.0f, "FLT_MIN/5", "FLT_MIN/3"},
  {std::numeric_limits<float>::min() / 7.0f, std::numeric_limits<float>::min() / 2.0f, "FLT_MIN/7", "FLT_MIN/2"},
  {std::numeric_limits<float>::min() / 8.0f, std::numeric_limits<float>::min() / 8.0f, "FLT_MIN/8", "FLT_MIN/8"},
  {std::numeric_limits<float>::min() / 6.0f, std::numeric_limits<float>::min() / 16.0f, "FLT_MIN/6", "FLT_MIN/16"},
  {std::numeric_limits<float>::min() / 11.0f, std::numeric_limits<float>::min() / 13.0f, "FLT_MIN/11", "FLT_MIN/13"},
  {std::numeric_limits<float>::min() / 16.0f, std::numeric_limits<float>::min() / 32.0f, "FLT_MIN/16", "FLT_MIN/32"},
  {std::numeric_limits<float>::min() / 32.0f, std::numeric_limits<float>::min() / 64.0f, "FLT_MIN/32", "FLT_MIN/64"},
  {std::numeric_limits<float>::min() / 1.5f, std::numeric_limits<float>::min() / 1.5f, "a=FLT_MIN/1.5", "a"},
  {std::numeric_limits<float>::min() / 1.5f, std::numeric_limits<float>::min() / 1.5f * 2.0f, "a=FLT_MIN/1.5", "a*2"},
  {std::numeric_limits<float>::min() / 1.5f, std::numeric_limits<float>::min() / 1.5f * 4.0f, "a=FLT_MIN/1.5", "a*4"},
  {std::numeric_limits<float>::min() / 1.5f, std::numeric_limits<float>::min() / 1.5f * 8.0f, "a=FLT_MIN/1.5", "a*8"},
  {std::numeric_limits<float>::min() / 1.5f, std::numeric_limits<float>::min() / 1.5f * 16.0f, "a=FLT_MIN/1.5", "a*16"},
  {std::numeric_limits<float>::min() / 1.5f, std::numeric_limits<float>::min() / 1.5f * 32.0f, "a=FLT_MIN/1.5", "a*32"},
  {std::numeric_limits<float>::min() / 1.5f, std::numeric_limits<float>::min() / 1.5f * 64.0f, "a=FLT_MIN/1.5", "a*64"},
  {std::numeric_limits<float>::min() / 1.5f, std::numeric_limits<float>::min() / 1.5f * 128.0f, "a=FLT_MIN/1.5", "a*128"},
  {std::numeric_limits<float>::min() / 1.5f, std::numeric_limits<float>::min() / 1.5f * 256.0f, "a=FLT_MIN/1.5", "a*256"},
  {std::numeric_limits<float>::min() / 1.5f, std::numeric_limits<float>::min() / 1.5f * 512.0f, "a=FLT_MIN/1.5", "a*512"},
  {std::numeric_limits<float>::min() / 1.5f, std::numeric_limits<float>::min() / 1.5f * 1024.0f, "a=FLT_MIN/1.5", "a*1024"},
  {std::numeric_limits<float>::min() / 1.5f, std::numeric_limits<float>::min() / 1.5f * 3.0f, "a=FLT_MIN/1.5", "a*3"},
  {std::numeric_limits<float>::min() / 1.5f, std::numeric_limits<float>::min() / 1.5f * 6.0f, "a=FLT_MIN/1.5", "a*6"},
  {std::numeric_limits<float>::min() / 1.5f, std::numeric_limits<float>::min() / 1.5f * 12.0f, "a=FLT_MIN/1.5", "a*12"},
  {std::numeric_limits<float>::min() / 1.5f, std::numeric_limits<float>::min() / 1.5f * 24.0f, "a=FLT_MIN/1.5", "a*24"},
  {std::numeric_limits<float>::min() / 1.5f, std::numeric_limits<float>::min() / 1.5f * 48.0f, "a=FLT_MIN/1.5", "a*48"},
  {std::numeric_limits<float>::min() / 1.5f, std::numeric_limits<float>::min() / 1.5f * 96.0f, "a=FLT_MIN/1.5", "a*96"},
  {std::numeric_limits<float>::min() / 1.5f, std::numeric_limits<float>::min() / 1.5f * 192.0f, "a=FLT_MIN/1.5", "a*192"},
  {std::numeric_limits<float>::min() / 1.5f, std::numeric_limits<float>::min() / 1.5f * 384.0f, "a=FLT_MIN/1.5", "a*384"},
  {std::numeric_limits<float>::min() / 1.5f, std::numeric_limits<float>::min() / 1.5f * 768.0f, "a=FLT_MIN/1.5", "a*768"},
  {std::numeric_limits<float>::min() / 1.5f, std::numeric_limits<float>::min() / 1.5f * 1536.0f, "a=FLT_MIN/1.5", "a*1536"},
  {std::numeric_limits<float>::max(), 1.0f, "FLT_MAX", "1"},
  {1.0f, std::numeric_limits<float>::max(), "1", "FLT_MAX"},
  {std::numeric_limits<float>::max(), std::numeric_limits<float>::min(), "FLT_MAX", "FLT_MIN"},
  {std::numeric_limits<float>::min(), std::numeric_limits<float>::max(), "FLT_MIN", "FLT_MAX"},
  {std::numeric_limits<float>::max() / 1.5f, 1.0f, "FLT_MAX/1.5", "1"},
  {1.0f, std::numeric_limits<float>::max() / 1.5f, "1", "FLT_MAX/1.5"},
  {std::numeric_limits<float>::max() / 2.0f, 1.0f, "FLT_MAX/2", "1"},
  {1.0f, std::numeric_limits<float>::max() / 2.0f, "1", "FLT_MAX/2"},
  {std::numeric_limits<float>::max() / 3.0f, 1.0f, "FLT_MAX/3", "1"},
  {1.0f, std::numeric_limits<float>::max() / 3.0f, "1", "FLT_MAX/3"},
  {std::numeric_limits<float>::max() / 4.0f, 1.0f, "FLT_MAX/4", "1"},
  {1.0f, std::numeric_limits<float>::max() / 4.0f, "1", "FLT_MAX/4"},
  {std::numeric_limits<float>::max() / 7.0f, 1.0f, "FLT_MAX/7", "1"},
  {1.0f, std::numeric_limits<float>::max() / 7.0f, "1", "FLT_MAX/7"},
  {std::numeric_limits<float>::max() / 8.0f, 1.0f, "FLT_MAX/8", "1"},
  {1.0f, std::numeric_limits<float>::max() / 8.0f, "1", "FLT_MAX/8"},
  {std::numeric_limits<float>::max() / 16.0f, 1.0f, "FLT_MAX/16", "1"},
  {1.0f, std::numeric_limits<float>::max() / 16.0f, "1", "FLT_MAX/16"},
  {std::numeric_limits<float>::max() / 2.0f, std::numeric_limits<float>::max() / 4.0f, "FLT_MAX/2", "FLT_MAX/4"},
  {std::numeric_limits<float>::max() / 3.0f, std::numeric_limits<float>::max() / 7.0f, "FLT_MAX/3", "FLT_MAX/7"},
  {std::numeric_limits<float>::max() / 4.0f, std::numeric_limits<float>::max() / 6.0f, "FLT_MAX/4", "FLT_MAX/6"},
  {std::numeric_limits<float>::max() / 5.0f, std::numeric_limits<float>::max() / 3.0f, "FLT_MAX/5", "FLT_MAX/3"},
  {std::numeric_limits<float>::max() / 6.0f, std::numeric_limits<float>::max() / 16.0f, "FLT_MAX/6", "FLT_MAX/16"},
  {std::numeric_limits<float>::max() / 7.0f, std::numeric_limits<float>::max() / 2.0f, "FLT_MAX/7", "FLT_MAX/2"},
  {std::numeric_limits<float>::max() / 8.0f, std::numeric_limits<float>::max() / 8.0f, "FLT_MAX/8", "FLT_MAX/8"},
  {std::numeric_limits<float>::max() / 11.0f, std::numeric_limits<float>::max() / 13.0f, "FLT_MAX/11", "FLT_MAX/13"},
  {std::numeric_limits<float>::max() / 16.0f, std::numeric_limits<float>::max() / 32.0f, "FLT_MAX/16", "FLT_MAX/32"},
  {std::numeric_limits<float>::max() / 32.0f, std::numeric_limits<float>::max() / 64.0f, "FLT_MAX/32", "FLT_MAX/64"},
  {std::numeric_limits<float>::max() / 1.5f, std::numeric_limits<float>::max() / 2.5f, "FLT_MAX/1.5", "FLT_MAX/2.5"},
  {std::numeric_limits<float>::max() / 2.0f, std::numeric_limits<float>::max() / 2.0f, "a=(FLT_MAX/2)", "a"},
  {std::numeric_limits<float>::max() / 1536.0f, std::numeric_limits<float>::max() / 1536.0f, "a=FLT_MAX/1536", "a"},
  {std::numeric_limits<float>::max() / 1536.0f, std::numeric_limits<float>::max() / 1536.0f * 2.0f, "a=FLT_MAX/1536", "a*2"},
  {std::numeric_limits<float>::max() / 1536.0f, std::numeric_limits<float>::max() / 1536.0f * 4.0f, "a=FLT_MAX/1536", "a*4"},
  {std::numeric_limits<float>::max() / 1536.0f, std::numeric_limits<float>::max() / 1536.0f * 8.0f, "a=FLT_MAX/1536", "a*8"},
  {std::numeric_limits<float>::max() / 1536.0f, std::numeric_limits<float>::max() / 1536.0f * 16.0f, "a=FLT_MAX/1536", "a*16"},
  {std::numeric_limits<float>::max() / 1536.0f, std::numeric_limits<float>::max() / 1536.0f * 32.0f, "a=FLT_MAX/1536", "a*32"},
  {std::numeric_limits<float>::max() / 1536.0f, std::numeric_limits<float>::max() / 1536.0f * 64.0f, "a=FLT_MAX/1536", "a*64"},
  {std::numeric_limits<float>::max() / 1536.0f, std::numeric_limits<float>::max() / 1536.0f * 128.0f, "a=FLT_MAX/1536", "a*128"},
  {std::numeric_limits<float>::max() / 1536.0f, std::numeric_limits<float>::max() / 1536.0f * 192.0f, "a=FLT_MAX/1536", "a*192"},
  {std::numeric_limits<float>::max() / 1536.0f, std::numeric_limits<float>::max() / 1536.0f * 256.0f, "a=FLT_MAX/1536", "a*256"},
  {std::numeric_limits<float>::max() / 1536.0f, std::numeric_limits<float>::max() / 1536.0f * 384.0f, "a=FLT_MAX/1536", "a*384"},
  {std::numeric_limits<float>::max() / 1536.0f, std::numeric_limits<float>::max() / 1536.0f * 400.0f, "a=FLT_MAX/1536", "a*400"},
  {std::numeric_limits<float>::max() / 1536.0f, std::numeric_limits<float>::max() / 1536.0f * 432.0f, "a=FLT_MAX/1536", "a*432"},
  {std::numeric_limits<float>::max() / 1536.0f, std::numeric_limits<float>::max() / 1536.0f * 448.0f, "a=FLT_MAX/1536", "a*448"},
  {std::numeric_limits<float>::max() / 1536.0f, std::numeric_limits<float>::max() / 1536.0f * 464.0f, "a=FLT_MAX/1536", "a*464"},
  {std::numeric_limits<float>::max() / 1536.0f, std::numeric_limits<float>::max() / 1536.0f * 480.0f, "a=FLT_MAX/1536", "a*480"},
  {std::numeric_limits<float>::max() / 1536.0f, std::numeric_limits<float>::max() / 1536.0f * 496.0f, "a=FLT_MAX/1536", "a*496"},
  {std::numeric_limits<float>::max() / 1536.0f, std::numeric_limits<float>::max() / 1536.0f * 504.0f, "a=FLT_MAX/1536", "a*504"},
  {std::numeric_limits<float>::max() / 1536.0f, std::numeric_limits<float>::max() / 1536.0f * 508.0f, "a=FLT_MAX/1536", "a*508"},
  {std::numeric_limits<float>::max() / 1536.0f, std::numeric_limits<float>::max() / 1536.0f * 510.0f, "a=FLT_MAX/1536", "a*510"},
  {std::numeric_limits<float>::max() / 1536.0f, std::numeric_limits<float>::max() / 1536.0f * 511.0f, "a=FLT_MAX/1536", "a*511"},
  {std::numeric_limits<float>::max() / 1536.0f, std::numeric_limits<float>::max() / 1536.0f * 512.0f, "a=FLT_MAX/1536", "a*512"},
  {std::numeric_limits<float>::max() / 1536.0f, std::numeric_limits<float>::max() / 1536.0f * 768.0f, "a=FLT_MAX/1536", "a*768"},
  {std::numeric_limits<float>::max() / 1536.0f, std::numeric_limits<float>::max() / 1536.0f * 1536.0f, "a=FLT_MAX/1536", "a*1536"},
  {std::numeric_limits<float>::min() * 4.0f, std::numeric_limits<float>::max() / 4.0f, "FLT_MIN*4", "FLT_MAX/4"},
  {std::numeric_limits<float>::max() / 4.0f, std::numeric_limits<float>::min() * 4.0f, "FLT_MAX/4", "FLT_MIN*4"},
  {std::numeric_limits<float>::min() * 3.0f, std::numeric_limits<float>::max() / 3.0f, "FLT_MIN*3", "FLT_MAX/3"},
  {std::numeric_limits<float>::max() / 3.0f, std::numeric_limits<float>::min() * 3.0f, "FLT_MAX/3", "FLT_MIN*3"},
  {std::numeric_limits<float>::min() / 2.0f, std::numeric_limits<float>::max() / 2.0f, "FLT_MIN/2", "FLT_MAX/2"},
  {std::numeric_limits<float>::max() / 2.0f, std::numeric_limits<float>::min() / 2.0f, "FLT_MAX/2", "FLT_MIN/2"},
  {std::numeric_limits<float>::min() / 3.0f, std::numeric_limits<float>::max() / 7.0f, "FLT_MIN/3", "FLT_MAX/7"},
  {std::numeric_limits<float>::max() / 7.0f, std::numeric_limits<float>::min() / 3.0f, "FLT_MAX/7", "FLT_MIN/3"},
  {1e-10f, 1e10f, "1e-10", "1e10"},
  {1e-20f, 1e20f, "1e-20", "1e20"},
  {1.0f, 1e20f, "1", "1e20"},
  {1e10f, 1e-10f, "1e10", "1e-10"},
  {1e20f, 1e-10f, "1e20", "1e-10"},
  {1e30f, 1e-5f, "1e30", "1e-5"},
  {3.4e38f, 0.5f, "3.4e38", "0.5"},
  {1e-38f, 1e10f, "1e-38", "1e10"},
  {1e38f, 2e38f, "1e38", "2e38"},
  {3e38f, 1e38f, "3e38", "1e38"},
  {2e38f, 2e38f, "2e38", "2e38"},

  // ---------------------------------------------------------------------
  // From pow_test.hip
  // ---------------------------------------------------------------------
  {2.0f, 0.0f, "2", "0"},
  {2.0f, 1.0f, "2", "1"},
  {2.0f, 2.0f, "2", "2"},
  {2.0f, 10.0f, "2", "10"},
  {2.0f, -1.0f, "2", "-1"},
  {2.0f, -3.0f, "2", "-3"},
  {2.0f, 0.5f, "2", "0.5"},
  {4.0f, 0.5f, "4", "0.5"},
  {8.0f, 1.0f / 3.0f, "8", "1/3"},
  {2.0f, 1.5f, "2", "1.5"},
  {2.718281828f, 1.0f, "e", "1"},
  {2.718281828f, 2.0f, "e", "2"},
  {2.718281828f, 0.5f, "e", "0.5"},
  {10.0f, 2.0f, "10", "2"},
  {10.0f, 0.5f, "10", "0.5"},
  {10.0f, -1.0f, "10", "-1"},
  {0.5f, 2.0f, "0.5", "2"},
  {0.5f, -2.0f, "0.5", "-2"},
  {0.25f, 0.5f, "0.25", "0.5"},
  {2.0f, 126.0f, "2", "126"},
  {2.0f, -126.0f, "2", "-126"},
  {2.718281828f, 88.0f, "e", "88"},
  {2.718281828f, -88.0f, "e", "-88"},
  {1.0f, -100.0f, "1", "-100"},
  {3.0f, 5.0f, "3", "5"},
  {5.0f, 4.0f, "5", "4"},
  {0.1f, 0.1f, "0.1", "0.1"},
  {0.9f, 10.0f, "0.9", "10"},
  {1.1f, 10.0f, "1.1", "10"},
  {-2.0f, 0.0f, "-2", "0"},
  {std::numeric_limits<float>::quiet_NaN(), 0.0f, "NaN", "0"},
  {0.0f, -1.0f, "0", "-1"},
  {0.0f, 2.0f, "0", "2"},
  {0.0f, -2.0f, "0", "-2"},
  {0.0f, 0.5f, "0", "0.5"},
  {-2.0f, 1.0f, "-2", "1"},
  {-2.0f, 2.0f, "-2", "2"},
  {-2.0f, 3.0f, "-2", "3"},
  {-2.0f, 4.0f, "-2", "4"},
  {-1.0f, 4.0f, "-1", "4"},
  {-2.0f, -1.0f, "-2", "-1"},
  {-2.0f, -2.0f, "-2", "-2"},
  {-2.0f, 0.5f, "-2", "0.5"},
  {-2.0f, 1.5f, "-2", "1.5"},
  {-2.0f, 0.1f, "-2", "0.1"},
  {std::numeric_limits<float>::infinity(), 2.0f, "+inf", "2"},
  {std::numeric_limits<float>::infinity(), -1.0f, "+inf", "-1"},
  {std::numeric_limits<float>::infinity(), 0.5f, "+inf", "0.5"},
  {-std::numeric_limits<float>::infinity(), 2.0f, "-inf", "2"},
  {-std::numeric_limits<float>::infinity(), 3.0f, "-inf", "3"},
  {-std::numeric_limits<float>::infinity(), -2.0f, "-inf", "-2"},
  {std::numeric_limits<float>::quiet_NaN(), 2.0f, "NaN", "2"},
  {std::numeric_limits<float>::quiet_NaN(), 0.5f, "NaN", "0.5"},
  {2.0f, std::numeric_limits<float>::quiet_NaN(), "2", "NaN"},
  {1.175494e-38f, 1.0f, "1.175494e-38", "1"},
  {1.175494e-38f, 2.0f, "1.175494e-38", "2"},
  {0.0f, std::numeric_limits<float>::infinity(), "0", "+inf"},
  {0.0f, -std::numeric_limits<float>::infinity(), "0", "-inf"},
  {-0.0f, std::numeric_limits<float>::infinity(), "-0", "+inf"},
  {-0.0f, -std::numeric_limits<float>::infinity(), "-0", "-inf"},
  {0.0f, -0.5f, "0", "-0.5"},
  {-0.0f, -0.5f, "-0", "-0.5"},
  {-0.0f, 2.0f, "-0", "2"},
  {-0.0f, 3.0f, "-0", "3"},
  {-0.0f, 0.5f, "-0", "0.5"},
  {-2.0f, std::numeric_limits<float>::infinity(), "-2", "+inf"},
  {-2.0f, -std::numeric_limits<float>::infinity(), "-2", "-inf"},
  {0.0f, std::numeric_limits<float>::quiet_NaN(), "0", "NaN"},
  {-std::numeric_limits<float>::quiet_NaN(), 0.0f, "-NaN", "0"},
  {1e-5f, 1.0f, "1e-5", "1"},
  {1e-5f, 2.0f, "1e-5", "2"},
  {1e-5f, 0.5f, "1e-5", "0.5"},
  {1e-5f, -1.0f, "1e-5", "-1"},
  {0.01f, 1.0f, "0.01", "1"},
  {0.01f, 2.0f, "0.01", "2"},
  {0.01f, 0.5f, "0.01", "0.5"},
  {0.01f, -1.0f, "0.01", "-1"},
  {-0.01f, 1.0f, "-0.01", "1"},
  {-0.01f, 2.0f, "-0.01", "2"},
  {-0.01f, 3.0f, "-0.01", "3"},
  {-0.01f, -1.0f, "-0.01", "-1"},
  {-0.01f, -2.0f, "-0.01", "-2"},
  {1e-9f, 1.0f, "1e-9", "1"},
  {1e-9f, 2.0f, "1e-9", "2"},
  {1e-9f, 0.5f, "1e-9", "0.5"},
  {1e-9f, -1.0f, "1e-9", "-1"},
  {-1e-9f, 1.0f, "-1e-9", "1"},
  {-1e-9f, 2.0f, "-1e-9", "2"},
  {-1e-9f, 3.0f, "-1e-9", "3"},
  {-1e-9f, -1.0f, "-1e-9", "-1"},
  {-1e-9f, -2.0f, "-1e-9", "-2"},
  {1.00000001f, 1.0f, "1+1e-8", "1"},
  {1.00000001f, 2.0f, "1+1e-8", "2"},
  {1.00000001f, 100.0f, "1+1e-8", "100"},
  {1.00000001f, -100.0f, "1+1e-8", "-100"},
  {1.00000001f, 1e6f, "1+1e-8", "1e6"},
  {1.00000001f, -1e6f, "1+1e-8", "-1e6"},
  {0.9999999f, 1.0f, "1-1e-7", "1"},
  {0.9999999f, 2.0f, "1-1e-7", "2"},
  {0.9999999f, 100.0f, "1-1e-7", "100"},
  {0.9999999f, -100.0f, "1-1e-7", "-100"},
  {0.9999999f, 1e6f, "1-1e-7", "1e6"},
  {0.9999999f, -1e6f, "1-1e-7", "-1e6"},
  {2.0f, 1e-5f, "2", "1e-5"},
  {2.0f, -1e-5f, "2", "-1e-5"},
  {2.0f, 0.01f, "2", "0.01"},
  {2.0f, -0.01f, "2", "-0.01"},
  {2.0f, 1e-9f, "2", "1e-9"},
  {2.0f, -1e-9f, "2", "-1e-9"},
  {10.0f, 1e-5f, "10", "1e-5"},
  {10.0f, -1e-5f, "10", "-1e-5"},
  {10.0f, 0.01f, "10", "0.01"},
  {10.0f, -0.01f, "10", "-0.01"},
  {0.5f, 1e-5f, "0.5", "1e-5"},
  {0.5f, -1e-5f, "0.5", "-1e-5"},
  {0.5f, 0.01f, "0.5", "0.01"},
  {0.5f, -0.01f, "0.5", "-0.01"},
  {0.0f, 1e-5f, "0", "1e-5"},
  {0.0f, 0.01f, "0", "0.01"},
  {-0.0f, 1e-5f, "-0", "1e-5"},
  {-0.0f, 0.01f, "-0", "0.01"},
  {2.0f, 1.00000001f, "2", "1+1e-8"},
  {2.0f, 0.9999999f, "2", "1-1e-7"},
  {10.0f, 1.00000001f, "10", "1+1e-8"},
  {10.0f, 0.9999999f, "10", "1-1e-7"},
  {0.5f, 1.00000001f, "0.5", "1+1e-8"},
  {0.5f, 0.9999999f, "0.5", "1-1e-7"},
  {100.0f, 1.00000001f, "100", "1+1e-8"},
  {100.0f, 0.9999999f, "100", "1-1e-7"},
  {-2.0f, 1.00000001f, "-2", "1+1e-8"},
  {-2.0f, 0.9999999f, "-2", "1-1e-7"},
  {1e-5f, 1e-5f, "1e-5", "1e-5"},
  {1e-5f, -1e-5f, "1e-5", "-1e-5"},
  {0.01f, 0.01f, "0.01", "0.01"},
  {0.01f, -0.01f, "0.01", "-0.01"},
  {1e-9f, 1e-9f, "1e-9", "1e-9"},
  {1e-9f, -1e-9f, "1e-9", "-1e-9"},
  {1.00000001f, 1.00000001f, "1+1e-8", "1+1e-8"},
  {1.00000001f, 0.9999999f, "1+1e-8", "1-1e-7"},
  {0.9999999f, 1.00000001f, "1-1e-7", "1+1e-8"},
  {0.9999999f, 0.9999999f, "1-1e-7", "1-1e-7"},
  {1.401298e-45f, 0.9999999f, "min_subnorm", "1-1e-7"},
  {1.401298e-45f, 1.00000001f, "min_subnorm", "1+1e-8"},
  {1.401298e-45f, 0.99f, "min_subnorm", "0.99"},
  {1.401298e-45f, 1.01f, "min_subnorm", "1.01"},
  {1.401298e-45f, 0.5f, "min_subnorm", "0.5"},
  {1.401298e-45f, 2.0f, "min_subnorm", "2"},
  {5.877472e-39f, 0.9999999f, "mid_subnorm", "1-1e-7"},
  {5.877472e-39f, 1.00000001f, "mid_subnorm", "1+1e-8"},
  {5.877472e-39f, 0.99f, "mid_subnorm", "0.99"},
  {5.877472e-39f, 1.01f, "mid_subnorm", "1.01"},
  {5.877472e-39f, 0.5f, "mid_subnorm", "0.5"},
  {5.877472e-39f, 2.0f, "mid_subnorm", "2"},
  {1.1754942e-38f, 0.9999999f, "max_subnorm", "1-1e-7"},
  {1.1754942e-38f, 1.00000001f, "max_subnorm", "1+1e-8"},
  {1.1754942e-38f, 0.99f, "max_subnorm", "0.99"},
  {1.1754942e-38f, 1.01f, "max_subnorm", "1.01"},
  {1.1754942e-38f, 0.5f, "max_subnorm", "0.5"},
  {1.1754942e-38f, 2.0f, "max_subnorm", "2"},
  {1.175494e-38f, 0.9999999f, "min_normal", "1-1e-7"},
  {1.175494e-38f, 1.00000001f, "min_normal", "1+1e-8"},
  {1.175494e-38f, 0.99f, "min_normal", "0.99"},
  {1.175494e-38f, 1.01f, "min_normal", "1.01"},
  {1.175494e-38f, 0.5f, "min_normal", "0.5"},
  {1.175494e-38f, -1.0f, "min_normal", "-1"},
  {1.175495e-38f, 0.9999999f, "min_normal+eps", "1-1e-7"},
  {1.175495e-38f, 1.00000001f, "min_normal+eps", "1+1e-8"},
  {1.175495e-38f, 0.99f, "min_normal+eps", "0.99"},
  {1.175495e-38f, 1.01f, "min_normal+eps", "1.01"},
  {2.350989e-38f, 0.9999999f, "2*min_normal", "1-1e-7"},
  {2.350989e-38f, 1.00000001f, "2*min_normal", "1+1e-8"},
  {2.350989e-38f, 0.99f, "2*min_normal", "0.99"},
  {2.350989e-38f, 1.01f, "2*min_normal", "1.01"},
  {1e-37f, 0.9999999f, "1e-37", "1-1e-7"},
  {1e-37f, 1.00000001f, "1e-37", "1+1e-8"},
  {1e-37f, 0.99f, "1e-37", "0.99"},
  {1e-37f, 1.01f, "1e-37", "1.01"},
  {1e-30f, 0.9999999f, "1e-30", "1-1e-7"},
  {1e-30f, 1.00000001f, "1e-30", "1+1e-8"},
  {1e-30f, 0.99f, "1e-30", "0.99"},
  {1e-30f, 1.01f, "1e-30", "1.01"},
  {0x1p-126f, 1.0f, "2^-126 ", "1"},
  {0x1p-126f, 0.5f, "2^-126 ", "0.5"},
  {0x1p-126f, 2.0f, "2^-126 ", "2"},
  {0x1p-126f, -1.0f, "2^-126 ", "-1"},
  {0x1p-124f, 1.0f, "2^-124 ", "1"},
  {0x1p-124f, 0.5f, "2^-124 ", "0.5"},
  {0x1p-124f, 2.0f, "2^-124 ", "2"},
  {0x1p-124f, -1.0f, "2^-124 ", "-1"},
  {0x1p-122f, 1.0f, "2^-122 ", "1"},
  {0x1p-122f, 0.5f, "2^-122 ", "0.5"},
  {0x1p-122f, 2.0f, "2^-122 ", "2"},
  {0x1p-122f, -1.0f, "2^-122 ", "-1"},
  {0x1p-120f, 1.0f, "2^-120 ", "1"},
  {0x1p-120f, 0.5f, "2^-120 ", "0.5"},
  {0x1p-120f, 2.0f, "2^-120 ", "2"},
  {0x1p-120f, -1.0f, "2^-120 ", "-1"},
  {0x1p-118f, 1.0f, "2^-118 ", "1"},
  {0x1p-118f, 0.5f, "2^-118 ", "0.5"},
  {0x1p-118f, 2.0f, "2^-118 ", "2"},
  {0x1p-118f, -1.0f, "2^-118 ", "-1"},
  {0x1p-116f, 1.0f, "2^-116 ", "1"},
  {0x1p-116f, 0.5f, "2^-116 ", "0.5"},
  {0x1p-116f, 2.0f, "2^-116 ", "2"},
  {0x1p-116f, -1.0f, "2^-116 ", "-1"},
  {0x1p-114f, 1.0f, "2^-114 ", "1"},
  {0x1p-114f, 0.5f, "2^-114 ", "0.5"},
  {0x1p-114f, 2.0f, "2^-114 ", "2"},
  {0x1p-114f, -1.0f, "2^-114 ", "-1"},
  {-0x1p-126f, 1.0f, "-2^-126", "1"},
  {-0x1p-126f, 2.0f, "-2^-126", "2"},
  {-0x1p-126f, 3.0f, "-2^-126", "3"},
  {-0x1p-126f, -1.0f, "-2^-126", "-1"},
  {-0x1p-126f, -2.0f, "-2^-126", "-2"},
  {-0x1p-122f, 1.0f, "-2^-122", "1"},
  {-0x1p-122f, 2.0f, "-2^-122", "2"},
  {-0x1p-122f, 3.0f, "-2^-122", "3"},
  {-0x1p-122f, -1.0f, "-2^-122", "-1"},
  {-0x1p-122f, -2.0f, "-2^-122", "-2"},
  {-0x1p-118f, 1.0f, "-2^-118", "1"},
  {-0x1p-118f, 2.0f, "-2^-118", "2"},
  {-0x1p-118f, 3.0f, "-2^-118", "3"},
  {-0x1p-118f, -1.0f, "-2^-118", "-1"},
  {-0x1p-118f, -2.0f, "-2^-118", "-2"},
  {-0x1p-114f, 1.0f, "-2^-114", "1"},
  {-0x1p-114f, 2.0f, "-2^-114", "2"},
  {-0x1p-114f, 3.0f, "-2^-114", "3"},
  {-0x1p-114f, -1.0f, "-2^-114", "-1"},
  {-0x1p-114f, -2.0f, "-2^-114", "-2"},
  {-0.0f, std::numeric_limits<float>::quiet_NaN(), "-0", "NaN"},
  {0.0f, 1.401298e-45f, "0", "+min_subnorm"},
  {0.0f, -1.401298e-45f, "0", "-min_subnorm"},
  {-0.0f, 1.401298e-45f, "-0", "+min_subnorm"},
  {-0.0f, -1.401298e-45f, "-0", "-min_subnorm"},
  {-0.0f, 16777215.0f, "-0", "2^24-1"},
  {-0.0f, 16777216.0f, "-0", "2^24"},
  {-0.0f, -16777215.0f, "-0", "2^24-1"},
  {-0.0f, -16777216.0f, "-0", "2^24"},

  // ---------------------------------------------------------------------
  // Atan2-focused corner cases (y=a, x=b)
  // ---------------------------------------------------------------------
  {+0.0f, +0.0f, "+0", "+0"},
  {-0.0f, -0.0f, "-0", "-0"},
  {+0.0f, -1.0f, "+0", "-1"},
  {-0.0f, -1.0f, "-0", "-1"},
  {std::numeric_limits<float>::denorm_min(), -0x1p-149f, "TRUE_MIN", "-2^-149"},
  {-std::numeric_limits<float>::denorm_min(), -0.0f, "-TRUE_MIN", "-0"},
  {std::numeric_limits<float>::denorm_min(), -1.0f, "TRUE_MIN", "-1"},
  {-std::numeric_limits<float>::denorm_min(), -1.0f, "-TRUE_MIN", "-1"},
  {std::numeric_limits<float>::denorm_min(), -std::numeric_limits<float>::denorm_min(), "TRUE_MIN", "-TRUE_MIN"},
  {-std::numeric_limits<float>::denorm_min(), std::numeric_limits<float>::denorm_min(), "-TRUE_MIN", "TRUE_MIN"},
  {-std::numeric_limits<float>::denorm_min(), -std::numeric_limits<float>::denorm_min(), "-TRUE_MIN", "-TRUE_MIN"},
  {std::numeric_limits<float>::min(), -std::numeric_limits<float>::min(), "FLT_MIN", "-FLT_MIN"},
  {-std::numeric_limits<float>::min(), std::numeric_limits<float>::min(), "-FLT_MIN", "FLT_MIN"},
  {-std::numeric_limits<float>::min(), -std::numeric_limits<float>::min(), "-FLT_MIN", "-FLT_MIN"},
  {std::numeric_limits<float>::max(), std::numeric_limits<float>::denorm_min(), "FLT_MAX", "TRUE_MIN"},
  {-std::numeric_limits<float>::max(), std::numeric_limits<float>::denorm_min(), "-FLT_MAX", "TRUE_MIN"},
  {std::numeric_limits<float>::max(), -std::numeric_limits<float>::denorm_min(), "FLT_MAX", "-TRUE_MIN"},
  {-std::numeric_limits<float>::max(), -std::numeric_limits<float>::denorm_min(), "-FLT_MAX", "-TRUE_MIN"},
  {std::numeric_limits<float>::denorm_min(), std::numeric_limits<float>::max(), "TRUE_MIN", "FLT_MAX"},
  {-std::numeric_limits<float>::denorm_min(), std::numeric_limits<float>::max(), "-TRUE_MIN", "FLT_MAX"},
  {std::numeric_limits<float>::denorm_min(), -std::numeric_limits<float>::max(), "TRUE_MIN", "-FLT_MAX"},
  {-std::numeric_limits<float>::denorm_min(), -std::numeric_limits<float>::max(), "-TRUE_MIN", "-FLT_MAX"},
  {-std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), "-inf", "-inf"},
  {std::numeric_limits<float>::quiet_NaN(), -0.0f, "qNaN", "-0"},
  {std::bit_cast<float>(0x3f7fffffu), -1.0f, "1-ulp", "-1"},
  {std::bit_cast<float>(0x3f800001u), -1.0f, "1+ulp", "-1"},
  {-std::bit_cast<float>(0x3f7fffffu), -1.0f, "-(1-ulp)", "-1"},
  {-std::bit_cast<float>(0x3f800001u), -1.0f, "-(1+ulp)", "-1"},

  // ---------------------------------------------------------------------
  // Copysign-focused corner cases
  // ---------------------------------------------------------------------
  {3.1415927f, -0.0f, "pi", "-0"},
  {-3.1415927f, +0.0f, "-pi", "+0"},
  {0x1p-149f, -std::numeric_limits<float>::infinity(), "2^-149", "-inf"},
  {-0x1p-149f, std::numeric_limits<float>::infinity(), "-2^-149", "+inf"},
  {16777216.0f, -1.0f, "2^24", "-1"},
  {-16777216.0f, 1.0f, "-2^24", "1"},
  {-std::numeric_limits<float>::quiet_NaN(), -std::numeric_limits<float>::infinity(), "-qNaN", "-inf"},
  {-std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(), "-qNaN", "+inf"},

  // ---------------------------------------------------------------------
  // Fmax/Fmin-focused corner cases
  // ---------------------------------------------------------------------
  {std::numeric_limits<float>::signaling_NaN(), -1.0f, "sNaN", "-1"},
  {-1.0f, std::numeric_limits<float>::signaling_NaN(), "-1", "sNaN"},
  {std::numeric_limits<float>::denorm_min(), -0.0f, "TRUE_MIN", "-0"},
  {-std::numeric_limits<float>::denorm_min(), +0.0f, "-TRUE_MIN", "+0"},
  {std::bit_cast<float>(0x7f7fffffu), std::bit_cast<float>(0x7f7ffffeu), "FLT_MAX", "FLT_MAX-ulp"},
  {std::bit_cast<float>(0xff7fffffu), std::bit_cast<float>(0xff7ffffeu), "-FLT_MAX", "-FLT_MAX+ulp"},
  {42.0f, 42.0f, "42", "42"},
  {-42.0f, -42.0f, "-42", "-42"},

  // ---------------------------------------------------------------------
  // Fmod-focused corner cases
  // ---------------------------------------------------------------------
  {7.0f, -2.0f, "7", "-2"},
  {-7.0f, -2.0f, "-7", "-2"},
  {0x1p-126f, 0x1p-149f, "2^-126", "2^-149"},
  {-0x1p-126f, 0x1p-149f, "-2^-126", "2^-149"},
  {0x1p-149f, 0x1p-126f, "2^-149", "2^-126"},
  {std::numeric_limits<float>::max(), 3.0f, "FLT_MAX", "3"},
  {-std::numeric_limits<float>::max(), 3.0f, "-FLT_MAX", "3"},
  {3.0f, std::numeric_limits<float>::denorm_min(), "3", "TRUE_MIN"},
  {-3.0f, std::numeric_limits<float>::denorm_min(), "-3", "TRUE_MIN"},
  {-3.0f, -std::numeric_limits<float>::denorm_min(), "-3", "-TRUE_MIN"},

  // ---------------------------------------------------------------------
  // Hypot-focused corner cases
  // ---------------------------------------------------------------------
  {0x1p127f, 0x1p127f, "2^127", "2^127"},
  {-0x1p127f, 0x1p127f, "-2^127", "2^127"},
  {0x1p-149f, 0x1p-149f, "2^-149", "2^-149"},
  {0x1p-149f, 0x1p127f, "2^-149", "2^127"},
  {3.0f, -4.0f, "3", "-4"},
  {-3.0f, 4.0f, "-3", "4"},
  {std::numeric_limits<float>::quiet_NaN(), -std::numeric_limits<float>::infinity(), "qNaN", "-inf"},
  {std::numeric_limits<float>::signaling_NaN(), std::numeric_limits<float>::infinity(), "sNaN", "+inf"},
  {std::numeric_limits<float>::max() / 1536.0f,
   std::numeric_limits<float>::max() / 1536.0f * 463.0f,
   "probe_a=FLT_MAX/1536",
   "probe_a*463"},
  {std::numeric_limits<float>::max() / 1536.0f,
   std::numeric_limits<float>::max() / 1536.0f * 464.0f,
   "probe_a=FLT_MAX/1536",
   "probe_a*464"},
  {std::numeric_limits<float>::max() / 1536.0f,
   std::numeric_limits<float>::max() / 1536.0f * 465.0f,
   "probe_a=FLT_MAX/1536",
   "probe_a*465"},
  {std::numeric_limits<float>::max() / 1536.0f * 464.0f,
   std::numeric_limits<float>::max() / 1536.0f,
   "probe_a*464",
   "probe_a=FLT_MAX/1536"},
  {2.999e38f, 1.0e38f, "2.999e38", "1e38"},
  {3.000e38f, 1.0e38f, "3e38", "1e38"},
  {3.001e38f, 1.0e38f, "3.001e38", "1e38"},
  {3.000e38f, 0.999e38f, "3e38", "0.999e38"},
  {3.000e38f, 1.001e38f, "3e38", "1.001e38"},
  {1.000e38f, 3.000e38f, "1e38", "3e38"},

  // ---------------------------------------------------------------------
  // Nextafter-focused corner cases
  // ---------------------------------------------------------------------
  {-1.0f, -2.0f, "-1", "-2"},
  {-1.0f, std::numeric_limits<float>::denorm_min(), "-1", "TRUE_MIN"},
  {0x1p-126f, 0.0f, "2^-126", "0"},
  {-0x1p-126f, 0.0f, "-2^-126", "0"},
  {0x1p-149f, std::numeric_limits<float>::infinity(), "2^-149", "+inf"},
  {-0x1p-149f, -std::numeric_limits<float>::infinity(), "-2^-149", "-inf"},
  {std::bit_cast<float>(0x7f7fffffu), 0.0f, "FLT_MAX", "0"},
  {std::bit_cast<float>(0xff7fffffu), 0.0f, "-FLT_MAX", "0"},
  {0.0f, std::numeric_limits<float>::quiet_NaN(), "0", "qNaN"},
  {-0.0f, std::numeric_limits<float>::quiet_NaN(), "-0", "qNaN"},

  // ---------------------------------------------------------------------
  // Remainder-focused corner cases
  // ---------------------------------------------------------------------
  {6.0f, 4.0f, "6", "4"},
  {10.0f, 4.0f, "10", "4"},
  {14.0f, 4.0f, "14", "4"},
  {-6.0f, 4.0f, "-6", "4"},
  {-10.0f, 4.0f, "-10", "4"},
  {-14.0f, 4.0f, "-14", "4"},
  {6.0f, -4.0f, "6", "-4"},
  {-6.0f, -4.0f, "-6", "-4"},
  {std::numeric_limits<float>::max(), 2.0f, "FLT_MAX", "2"},
  {0x1p-149f, 2.0f, "2^-149", "2"},
  {-0x1p-149f, 2.0f, "-2^-149", "2"},

  // ---------------------------------------------------------------------
  // Dim/Fldiv/Cldiv/Root-focused corner cases
  // ---------------------------------------------------------------------
  {5.0f, -3.0f, "5", "-3"},
  {-5.0f, -3.0f, "-5", "-3"},
  {1e-5f, 1e-4f, "1e-5", "1e-4"},
  {1e-4f, 1e-5f, "1e-4", "1e-5"},
  {-7.0f, 2.0f, "-7", "2"},
  {-8.0f, 3.0f, "-8", "3"},
  {-4.0f, 2.0f, "-4", "2"},
  {2.0f, std::numeric_limits<float>::infinity(), "2", "+inf"},
};
// clang-format on

static constexpr size_t kNumBinaryCases = sizeof(kCases) / sizeof(kCases[0]);

enum class BinaryOp {
  Unknown,
  Dim,
  Div,
  Pow,
  Fldiv,
  Cldiv,
  Root,
  Atan2,
  Copysign,
  Fmax,
  Fmin,
  Fmod,
  Hypot,
  Nextafter,
  Remainder,
};

static const char *opName(BinaryOp op) {
  switch (op) {
  case BinaryOp::Dim:
    return "dim";
  case BinaryOp::Div:
    return "div";
  case BinaryOp::Pow:
    return "pow";
  case BinaryOp::Fldiv:
    return "fldiv";
  case BinaryOp::Cldiv:
    return "cldiv";
  case BinaryOp::Root:
    return "root";
  case BinaryOp::Atan2:
    return "atan2";
  case BinaryOp::Copysign:
    return "copysign";
  case BinaryOp::Fmax:
    return "fmax";
  case BinaryOp::Fmin:
    return "fmin";
  case BinaryOp::Fmod:
    return "fmod";
  case BinaryOp::Hypot:
    return "hypot";
  case BinaryOp::Nextafter:
    return "nextafter";
  case BinaryOp::Remainder:
    return "remainder";
  case BinaryOp::Unknown:
    return "unknown";
  }
  return "?";
}

class BinaryTester {
public:
  static constexpr size_t N = kNumBinaryCases;

  float input_a[N];
  float input_b[N];
  float out_cuda[N];
  float out_custom[N];

  __host__ BinaryTester() {
    for (size_t i = 0; i < N; i++) {
      input_a[i] = kCases[i].a;
      input_b[i] = kCases[i].b;
    }
  }

  __host__ void reset() {
    std::memset(out_cuda, 0xff, sizeof(out_cuda));
    std::memset(out_custom, 0xff, sizeof(out_custom));
  }
};

__global__ void testDimCuda(BinaryTester *self) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < BinaryTester::N)
    self->out_cuda[i] = fmaxf(self->input_a[i] - self->input_b[i], 0.0f);
}
__global__ void testDimCustom(BinaryTester *self) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < BinaryTester::N)
    self->out_custom[i] = custom_dimf(self->input_a[i], self->input_b[i]);
}

__global__ void testDivCuda(BinaryTester *self) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < BinaryTester::N)
    self->out_cuda[i] = self->input_a[i] / self->input_b[i];
}
__global__ void testDivCustom(BinaryTester *self) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < BinaryTester::N)
    self->out_custom[i] = custom_fdividef(self->input_a[i], self->input_b[i]);
}

__global__ void testPowCuda(BinaryTester *self) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < BinaryTester::N)
    self->out_cuda[i] = powf(self->input_a[i], self->input_b[i]);
}
__global__ void testPowCustom(BinaryTester *self) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < BinaryTester::N)
    self->out_custom[i] = custom_powf(self->input_a[i], self->input_b[i]);
}

__global__ void testFldivCuda(BinaryTester *self) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < BinaryTester::N)
    self->out_cuda[i] = floorf(self->input_a[i] / self->input_b[i]);
}
__global__ void testFldivCustom(BinaryTester *self) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < BinaryTester::N)
    self->out_custom[i] = custom_fldivf(self->input_a[i], self->input_b[i]);
}

__global__ void testCldivCuda(BinaryTester *self) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < BinaryTester::N)
    self->out_cuda[i] = -floorf((-self->input_a[i]) / self->input_b[i]);
}
__global__ void testCldivCustom(BinaryTester *self) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < BinaryTester::N)
    self->out_custom[i] = custom_cldivf(self->input_a[i], self->input_b[i]);
}

__global__ void testRootCuda(BinaryTester *self) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < BinaryTester::N)
    self->out_cuda[i] = powf(self->input_a[i], 1.0f / self->input_b[i]);
}
__global__ void testRootCustom(BinaryTester *self) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < BinaryTester::N)
    self->out_custom[i] = custom_rootf(self->input_a[i], self->input_b[i]);
}

__global__ void testAtan2Cuda(BinaryTester *self) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < BinaryTester::N)
    self->out_cuda[i] = atan2f(self->input_a[i], self->input_b[i]);
}
__global__ void testAtan2Custom(BinaryTester *self) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < BinaryTester::N)
    self->out_custom[i] = custom_atan2f(self->input_a[i], self->input_b[i]);
}

__global__ void testCopysignCuda(BinaryTester *self) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < BinaryTester::N)
    self->out_cuda[i] = copysignf(self->input_a[i], self->input_b[i]);
}
__global__ void testCopysignCustom(BinaryTester *self) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < BinaryTester::N)
    self->out_custom[i] = custom_copysignf(self->input_a[i], self->input_b[i]);
}

__global__ void testFmaxCuda(BinaryTester *self) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < BinaryTester::N)
    self->out_cuda[i] = fmaxf(self->input_a[i], self->input_b[i]);
}
__global__ void testFmaxCustom(BinaryTester *self) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < BinaryTester::N)
    self->out_custom[i] = custom_fmaxf(self->input_a[i], self->input_b[i]);
}

__global__ void testFminCuda(BinaryTester *self) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < BinaryTester::N)
    self->out_cuda[i] = fminf(self->input_a[i], self->input_b[i]);
}
__global__ void testFminCustom(BinaryTester *self) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < BinaryTester::N)
    self->out_custom[i] = custom_fminf(self->input_a[i], self->input_b[i]);
}

__global__ void testFmodCuda(BinaryTester *self) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < BinaryTester::N)
    self->out_cuda[i] = fmodf(self->input_a[i], self->input_b[i]);
}
__global__ void testFmodCustom(BinaryTester *self) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < BinaryTester::N)
    self->out_custom[i] = custom_fmodf(self->input_a[i], self->input_b[i]);
}

__global__ void testHypotCuda(BinaryTester *self) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < BinaryTester::N)
    self->out_cuda[i] = hypotf(self->input_a[i], self->input_b[i]);
}
__global__ void testHypotCustom(BinaryTester *self) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < BinaryTester::N)
    self->out_custom[i] = custom_hypotf(self->input_a[i], self->input_b[i]);
}

__global__ void testNextafterCuda(BinaryTester *self) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < BinaryTester::N)
    self->out_cuda[i] = nextafterf(self->input_a[i], self->input_b[i]);
}
__global__ void testNextafterCustom(BinaryTester *self) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < BinaryTester::N)
    self->out_custom[i] = custom_nextafterf(self->input_a[i], self->input_b[i]);
}

__global__ void testRemainderCuda(BinaryTester *self) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < BinaryTester::N)
    self->out_cuda[i] = remainderf(self->input_a[i], self->input_b[i]);
}
__global__ void testRemainderCustom(BinaryTester *self) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < BinaryTester::N)
    self->out_custom[i] = custom_remainderf(self->input_a[i], self->input_b[i]);
}

// -- One-value kernels for asm inspection --------------------------------
__global__ void testOneDivCuda(BinaryTester *self) {
  self->out_cuda[0] = self->input_a[0] / self->input_b[0];
}
__global__ void testOneDivCustom(BinaryTester *self) {
  self->out_custom[0] = custom_fdividef(self->input_a[0], self->input_b[0]);
}
__global__ void testOneDimCuda(BinaryTester *self) {
  self->out_cuda[0] = fmaxf(self->input_a[0] - self->input_b[0], 0.0f);
}
__global__ void testOneDimCustom(BinaryTester *self) {
  self->out_custom[0] = custom_dimf(self->input_a[0], self->input_b[0]);
}
__global__ void testOnePowCuda(BinaryTester *self) {
  self->out_cuda[0] = powf(self->input_a[0], self->input_b[0]);
}
__global__ void testOnePowCustom(BinaryTester *self) {
  self->out_custom[0] = custom_powf(self->input_a[0], self->input_b[0]);
}
__global__ void testOneFldivCuda(BinaryTester *self) {
  self->out_cuda[0] = floorf(self->input_a[0] / self->input_b[0]);
}
__global__ void testOneFldivCustom(BinaryTester *self) {
  self->out_custom[0] = custom_fldivf(self->input_a[0], self->input_b[0]);
}
__global__ void testOneCldivCuda(BinaryTester *self) {
  self->out_cuda[0] = -floorf((-self->input_a[0]) / self->input_b[0]);
}
__global__ void testOneCldivCustom(BinaryTester *self) {
  self->out_custom[0] = custom_cldivf(self->input_a[0], self->input_b[0]);
}
__global__ void testOneRootCuda(BinaryTester *self) {
  self->out_cuda[0] = powf(self->input_a[0], 1.0f / self->input_b[0]);
}
__global__ void testOneRootCustom(BinaryTester *self) {
  self->out_custom[0] = custom_rootf(self->input_a[0], self->input_b[0]);
}
__global__ void testOneAtan2Cuda(BinaryTester *self) {
  self->out_cuda[0] = atan2f(self->input_a[0], self->input_b[0]);
}
__global__ void testOneAtan2Custom(BinaryTester *self) {
  self->out_custom[0] = custom_atan2f(self->input_a[0], self->input_b[0]);
}
__global__ void testOneCopysignCuda(BinaryTester *self) {
  self->out_cuda[0] = copysignf(self->input_a[0], self->input_b[0]);
}
__global__ void testOneCopysignCustom(BinaryTester *self) {
  self->out_custom[0] = custom_copysignf(self->input_a[0], self->input_b[0]);
}
__global__ void testOneFmaxCuda(BinaryTester *self) {
  self->out_cuda[0] = fmaxf(self->input_a[0], self->input_b[0]);
}
__global__ void testOneFmaxCustom(BinaryTester *self) {
  self->out_custom[0] = custom_fmaxf(self->input_a[0], self->input_b[0]);
}
__global__ void testOneFminCuda(BinaryTester *self) {
  self->out_cuda[0] = fminf(self->input_a[0], self->input_b[0]);
}
__global__ void testOneFminCustom(BinaryTester *self) {
  self->out_custom[0] = custom_fminf(self->input_a[0], self->input_b[0]);
}
__global__ void testOneFmodCuda(BinaryTester *self) {
  self->out_cuda[0] = fmodf(self->input_a[0], self->input_b[0]);
}
__global__ void testOneFmodCustom(BinaryTester *self) {
  self->out_custom[0] = custom_fmodf(self->input_a[0], self->input_b[0]);
}
__global__ void testOneHypotCuda(BinaryTester *self) {
  self->out_cuda[0] = hypotf(self->input_a[0], self->input_b[0]);
}
__global__ void testOneHypotCustom(BinaryTester *self) {
  self->out_custom[0] = custom_hypotf(self->input_a[0], self->input_b[0]);
}
__global__ void testOneNextafterCuda(BinaryTester *self) {
  self->out_cuda[0] = nextafterf(self->input_a[0], self->input_b[0]);
}
__global__ void testOneNextafterCustom(BinaryTester *self) {
  self->out_custom[0] = custom_nextafterf(self->input_a[0], self->input_b[0]);
}
__global__ void testOneRemainderCuda(BinaryTester *self) {
  self->out_cuda[0] = remainderf(self->input_a[0], self->input_b[0]);
}
__global__ void testOneRemainderCustom(BinaryTester *self) {
  self->out_custom[0] = custom_remainderf(self->input_a[0], self->input_b[0]);
}

bool verbose{};
bool useColor{};
bool quiet{};
bool csvOutput{};

static std::string fp32Hex(float v) {
  uint32_t bits;
  std::memcpy(&bits, &v, sizeof(bits));
  return std::format("0x{:08x}", bits);
}

static void displayResults(const BinaryTester &t, BinaryOp op,
                           const float *torcheager,
                           const float *torchinductor) {
  const char *name = opName(op);

  if (csvOutput) {
    std::cout << std::format("op,idx,a,b,alabel,blabel,row,{0}(cuda),{0}("
                             "custom),torch_eager,torch_inductor\n",
                             name);

    for (size_t i = 0; i < BinaryTester::N; i++) {
      float a = t.input_a[i];
      float b = t.input_b[i];
      float ref = t.out_cuda[i];

      OneResult32 v_custom(ref, t.out_custom[i], true, verbose);
      OneResult32 v_eager(ref, torcheager ? torcheager[i] : 0.0f,
                          torcheager != nullptr, verbose);
      OneResult32 v_inductor(ref, torchinductor ? torchinductor[i] : 0.0f,
                             torchinductor != nullptr, verbose);

      bool allMatch = v_custom.match and v_eager.match and v_inductor.match;
      if (!quiet or !allMatch) {
        std::cout << std::format(
            "{},{},{:g},{:g},\"{}\",\"{}\",VALUE,{:g},{:g},{},{}\n", name, i, a,
            b, kCases[i].alabel, kCases[i].blabel, ref, t.out_custom[i],
            torcheager ? std::format("{:g}", torcheager[i]) : "",
            torchinductor ? std::format("{:g}", torchinductor[i]) : "");
      }
      if (!allMatch) {
        std::cout << std::format("{},{},,,,HEX,{},{},{},{}\n", name, i,
                                 fp32Hex(ref), v_custom.hexValue(),
                                 v_eager.hexValue(), v_inductor.hexValue());
        std::cout << std::format("{},{},,,,DIFF,,{},{},{}\n", name, i,
                                 v_custom.errorString(), v_eager.errorString(),
                                 v_inductor.errorString());
      }
    }
    return;
  }

  if (useColor)
    std::cout << RED;
  std::cout << std::format("BINARY OP: {}\n\n", name);
  if (useColor)
    std::cout << RESET;

  std::cout << std::format(
      "{:>4}{:>16}{:>16}{:>14}{:>14}{:>16}{:>16}{:>16}{:>16}\n", "Idx", "a",
      "b", "a label", "b label", std::format("{}(cuda)", name),
      std::format("{}(custom)", name), torcheager ? "torch-eager" : "",
      torchinductor ? "torch-inductor" : "");
  std::cout << std::string(142, '-') << "\n";

  for (size_t i = 0; i < BinaryTester::N; i++) {
    float a = t.input_a[i];
    float b = t.input_b[i];
    float ref = t.out_cuda[i];

    OneResult32 v_custom(ref, t.out_custom[i], true, verbose);
    OneResult32 v_eager(ref, torcheager ? torcheager[i] : 0.0f,
                        torcheager != nullptr, verbose);
    OneResult32 v_inductor(ref, torchinductor ? torchinductor[i] : 0.0f,
                           torchinductor != nullptr, verbose);

    bool allMatch = v_custom.match and v_eager.match and v_inductor.match;

    if (!quiet or !allMatch) {
      std::cout << std::format(
          "{:>4}{:>16g}{:>16g}{:>14}{:>14}{:>16.6g}{}{}{}\n", i, a, b,
          kCases[i].alabel, kCases[i].blabel, ref, v_custom.value(),
          v_eager.value(), v_inductor.value());
    }

    if (verbose or !allMatch) {
      std::cout << std::format(
          "{:>4}{:>16}{:>16}{:>14}{:>14}{:>16}{:>16}{:>16}{:>16}\n", "",
          fp32Hex(a), fp32Hex(b), "", "", fp32Hex(ref), v_custom.hexValue(),
          v_eager.hexValue(), v_inductor.hexValue());
    }

    if (!allMatch) {
      const char *color = YELLOW;
      std::string es_custom = v_custom.errorString();
      std::string es_eager = v_eager.errorString();
      std::string es_inductor = v_inductor.errorString();
      if ((es_custom == "ERROR") or (es_eager == "ERROR") or
          (es_inductor == "ERROR"))
        color = RED;
      if (useColor)
        std::cout << color;

      std::cout << std::format(
          "{:>4}{:>16}{:>16}{:>14}{:>14}{:>16}{:>16}{:>16}{:>16}\n", "", "", "",
          "", "", "", es_custom, es_eager, es_inductor);
      if (useColor)
        std::cout << RESET;
    }
  }
}

using KernelFn = void (*)(BinaryTester *);

struct KernelPair {
  KernelFn cuda;
  KernelFn custom;
};

static KernelPair kernelsForOp(BinaryOp op) {
  switch (op) {
  case BinaryOp::Dim:
    return {testDimCuda, testDimCustom};
  case BinaryOp::Div:
    return {testDivCuda, testDivCustom};
  case BinaryOp::Pow:
    return {testPowCuda, testPowCustom};
  case BinaryOp::Fldiv:
    return {testFldivCuda, testFldivCustom};
  case BinaryOp::Cldiv:
    return {testCldivCuda, testCldivCustom};
  case BinaryOp::Root:
    return {testRootCuda, testRootCustom};
  case BinaryOp::Atan2:
    return {testAtan2Cuda, testAtan2Custom};
  case BinaryOp::Copysign:
    return {testCopysignCuda, testCopysignCustom};
  case BinaryOp::Fmax:
    return {testFmaxCuda, testFmaxCustom};
  case BinaryOp::Fmin:
    return {testFminCuda, testFminCustom};
  case BinaryOp::Fmod:
    return {testFmodCuda, testFmodCustom};
  case BinaryOp::Hypot:
    return {testHypotCuda, testHypotCustom};
  case BinaryOp::Nextafter:
    return {testNextafterCuda, testNextafterCustom};
  case BinaryOp::Remainder:
    return {testRemainderCuda, testRemainderCustom};
  case BinaryOp::Unknown:
    return {nullptr, nullptr};
  }
  return {nullptr, nullptr};
}

static BinaryOp parseOpName(const std::string &name) {
  if (name == "dim")
    return BinaryOp::Dim;
  if (name == "div")
    return BinaryOp::Div;
  if (name == "pow")
    return BinaryOp::Pow;
  if (name == "fldiv")
    return BinaryOp::Fldiv;
  if (name == "cldiv")
    return BinaryOp::Cldiv;
  if (name == "root")
    return BinaryOp::Root;
  if (name == "atan2")
    return BinaryOp::Atan2;
  if (name == "copysign")
    return BinaryOp::Copysign;
  if (name == "fmax")
    return BinaryOp::Fmax;
  if (name == "fmin")
    return BinaryOp::Fmin;
  if (name == "fmod")
    return BinaryOp::Fmod;
  if (name == "hypot")
    return BinaryOp::Hypot;
  if (name == "nextafter")
    return BinaryOp::Nextafter;
  if (name == "remainder")
    return BinaryOp::Remainder;
  return BinaryOp::Unknown;
}

int main(int argc, char **argv) {
  int c{};
  const char *dumpFile{};
  const char *torchinductorFile{};
  const char *torcheagerFile{};
  BinaryOp selectedOp = BinaryOp::Unknown;

  enum class Options {
    op,
    dim,
    div,
    pow,
    fldiv,
    cldiv,
    root,
    atan2,
    copysign,
    fmax,
    fmin,
    fmod,
    hypot,
    nextafter,
    remainder,
    help,
    verbose,
    quiet,
    color,
    csv,
    dumpinputs,
    torcheager,
    torchinductor,
  };

  constexpr struct option longOptions[] = {
      {"op", 1, nullptr, (int)Options::op},
      {"dim", 0, nullptr, (int)Options::dim},
      {"div", 0, nullptr, (int)Options::div},
      {"pow", 0, nullptr, (int)Options::pow},
      {"fldiv", 0, nullptr, (int)Options::fldiv},
      {"cldiv", 0, nullptr, (int)Options::cldiv},
      {"root", 0, nullptr, (int)Options::root},
      {"atan2", 0, nullptr, (int)Options::atan2},
      {"copysign", 0, nullptr, (int)Options::copysign},
      {"fmax", 0, nullptr, (int)Options::fmax},
      {"fmin", 0, nullptr, (int)Options::fmin},
      {"fmod", 0, nullptr, (int)Options::fmod},
      {"hypot", 0, nullptr, (int)Options::hypot},
      {"nextafter", 0, nullptr, (int)Options::nextafter},
      {"remainder", 0, nullptr, (int)Options::remainder},
      {"help", 0, nullptr, (int)Options::help},
      {"verbose", 0, nullptr, (int)Options::verbose},
      {"quiet", 0, nullptr, (int)Options::quiet},
      {"color", 0, nullptr, (int)Options::color},
      {"csv", 0, nullptr, (int)Options::csv},
      {"dump-inputs", 1, nullptr, (int)Options::dumpinputs},
      {"torcheager", 1, nullptr, (int)Options::torcheager},
      {"torchinductor", 1, nullptr, (int)Options::torchinductor},
      {nullptr, 0, nullptr, 0},
  };

  while (-1 != (c = getopt_long(argc, argv, "", longOptions, nullptr))) {
    switch ((Options)c) {
    case Options::op:
      selectedOp = parseOpName(optarg ? optarg : "");
      break;
    case Options::dim:
      selectedOp = BinaryOp::Dim;
      break;
    case Options::div:
      selectedOp = BinaryOp::Div;
      break;
    case Options::pow:
      selectedOp = BinaryOp::Pow;
      break;
    case Options::fldiv:
      selectedOp = BinaryOp::Fldiv;
      break;
    case Options::cldiv:
      selectedOp = BinaryOp::Cldiv;
      break;
    case Options::root:
      selectedOp = BinaryOp::Root;
      break;
    case Options::atan2:
      selectedOp = BinaryOp::Atan2;
      break;
    case Options::copysign:
      selectedOp = BinaryOp::Copysign;
      break;
    case Options::fmax:
      selectedOp = BinaryOp::Fmax;
      break;
    case Options::fmin:
      selectedOp = BinaryOp::Fmin;
      break;
    case Options::fmod:
      selectedOp = BinaryOp::Fmod;
      break;
    case Options::hypot:
      selectedOp = BinaryOp::Hypot;
      break;
    case Options::nextafter:
      selectedOp = BinaryOp::Nextafter;
      break;
    case Options::remainder:
      selectedOp = BinaryOp::Remainder;
      break;
    case Options::verbose:
      verbose = true;
      break;
    case Options::quiet:
      quiet = true;
      break;
    case Options::color:
      useColor = true;
      break;
    case Options::csv:
      csvOutput = true;
      break;
    case Options::dumpinputs:
      dumpFile = optarg;
      break;
    case Options::torcheager:
      torcheagerFile = optarg;
      break;
    case Options::torchinductor:
      torchinductorFile = optarg;
      break;
    case Options::help:
      std::cout
          << "binary_test"
             " --[dim|div|pow|fldiv|cldiv|root|atan2|copysign|fmax|fmin|fmod|"
             "hypot|nextafter|remainder]"
             " [--op name]"
             " [--verbose]"
             " [--quiet]"
             " [--color]"
             " [--csv]"
             " [--dump-inputs filename]"
             " [--torcheager file.bin]"
             " [--torchinductor file.bin]\n\n"
             "Run with:\n"
             "  complexops/binary_test --dump-inputs ./binarytest.in\n"
             "  ../bin/torchbinary.py --op div --file ./binarytest.in\n"
             "  complexops/binary_test --div --torcheager torcheagerdiv.bin "
             "--torchinductor torchinductordiv.bin --verbose --quiet --color | "
             "less -R\n\n"
             "\t--OP.         (required) one of: dim, div, pow, fldiv, cldiv, "
             "root, atan2, copysign, fmax, fmin, fmod, hypot, nextafter, "
             "remainder\n"
             "\t--op name.    alternative way to select the operation\n"
             "\t--verbose.    Show hex values even when matches\n"
             "\t--quiet.      Suppress rows where all results match\n"
             "\t--color.      Highlight mismatch diagnostics\n"
             "\t--csv.        Emit CSV output (VALUE/HEX/DIFF rows)\n"
             "\t--dump-inputs Write interleaved (a,b) float pairs as binary to "
             "file\n"
             "\t--torcheager  Binary float file with torch eager results\n"
             "\t--torchinductor Binary float file with torch inductor results\n"
             "\t--help.       Show this output and exit\n";
      return 0;
    default:
      std::cerr << "binary_test: unknown option\n";
      return 1;
    }
  }

  if (dumpFile) {
    std::ofstream ofs(dumpFile, std::ios::binary);
    if (!ofs) {
      std::cerr << "Failed to open file for writing: " << dumpFile << std::endl;
      return 2;
    }
    BinaryTester tmp;
    for (size_t i = 0; i < BinaryTester::N; ++i) {
      ofs.write(reinterpret_cast<const char *>(&tmp.input_a[i]), sizeof(float));
      ofs.write(reinterpret_cast<const char *>(&tmp.input_b[i]), sizeof(float));
    }
    ofs.close();
    std::cout << "Wrote interleaved (a,b) input values to " << dumpFile
              << std::endl;
    return 0;
  }

  if (selectedOp == BinaryOp::Unknown) {
    std::cerr << "binary_test: --[op] is required\n";
    return 1;
  }

  std::vector<float> torchinductorOut, torcheagerOut;
  if (torchinductorFile) {
    torchinductorOut = readBinaryFloatFile(torchinductorFile, BinaryTester::N);
    if (torchinductorOut.empty())
      torchinductorFile = nullptr;
  }
  if (torcheagerFile) {
    torcheagerOut = readBinaryFloatFile(torcheagerFile, BinaryTester::N);
    if (torcheagerOut.empty())
      torcheagerFile = nullptr;
  }

  BinaryTester *tester;
  CUDA_CHECK(cudaMallocManaged(&tester, sizeof(BinaryTester)));
  new (tester) BinaryTester();

  constexpr size_t kThreads = 256;
  dim3 blockSize(kThreads);
  dim3 gridSize((BinaryTester::N + kThreads - 1) / kThreads);

  tester->reset();

  auto [cudaKernel, customKernel] = kernelsForOp(selectedOp);
  if (!cudaKernel or !customKernel) {
    std::cerr << "binary_test: no kernels for selected operation\n";
    CUDA_CHECK(cudaFree(tester));
    return 1;
  }

  cudaKernel<<<gridSize, blockSize>>>(tester);
  CUDA_CHECK(cudaDeviceSynchronize());
  customKernel<<<gridSize, blockSize>>>(tester);
  CUDA_CHECK(cudaDeviceSynchronize());

  displayResults(*tester, selectedOp,
                 torcheagerFile ? torcheagerOut.data() : nullptr,
                 torchinductorFile ? torchinductorOut.data() : nullptr);

  CUDA_CHECK(cudaFree(tester));
  return 0;
}

// vim: et ts=2 sw=2
