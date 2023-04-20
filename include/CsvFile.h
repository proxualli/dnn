#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <map>

class CsvFile
{
private:
      std::ofstream os;

public:
    const std::string Separator;
    const std::string Quote;

    CsvFile(const std::string& filename, const std::string& separator = ";", const std::string& quote = "") :
        Separator(separator),
        Quote(quote),
        os()
    {
        std::setlocale(LC_ALL, "");

        os.exceptions(std::ios::failbit | std::ios::badbit);
        os.open(filename);
    }

    ~CsvFile()
    {
        Flush();
        os.close();
    }

    void Flush()
    {
        os.flush();
    }

    void EndRow()
    {
        os << std::endl;
    }

    CsvFile& operator << (CsvFile& (*val)(CsvFile&))
    {
        return val(*this);
    }

    CsvFile& operator << (const char* val)
    {
        os << Quote << val << Quote << Separator;
        return *this;
    }

    CsvFile& operator << (const std::string& val)
    {
        os << Quote << val << Quote << Separator;
        return *this;
    }

    CsvFile& operator << (const bool& val)
    {
        os << (val ? std::string("True") : std::string("False")) << Separator;
        return *this;
    }

    CsvFile& operator << (const float& val)
    {
        std::stringstream stream;
        stream << std::setprecision(std::streamsize(8)) << std::fixed << val;
        os << stream.str() << Separator;
        return *this;
    }

    CsvFile& operator << (const double& val)
    {
        std::stringstream stream;
        stream << std::setprecision(std::streamsize(16)) << std::fixed << val;
        os << stream.str() << Separator;
        return *this;
    }
      
    template<typename T>
    CsvFile& operator << (const T& val)
    {
        os << val << Separator;
        return *this;
    }
};


inline static CsvFile& EndRow(CsvFile& file)
{
    file.EndRow();
    return file;
}

inline static CsvFile& Flush(CsvFile& file)
{
    file.Flush();
    return file;
}

inline static std::string ReadFileToString(const std::string& fileName)
{
    auto file = std::ifstream(fileName);

    if (!file.bad() && file.is_open())
    {
        auto ss = std::ostringstream{};
        ss << file.rdbuf();
        return ss.str();
    }

    std::cerr << std::string("Could not open the file - '") << fileName << std::string("'") << std::endl;

    return std::string("");
}
