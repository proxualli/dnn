#pragma once
#include "Utils.h"

namespace dnn
{
    class CsvFile
    {
    private:
        const std::locale newLocale = std::locale(std::locale(""), new no_separator());
        const std::locale oldLocale;
        std::ofstream os;

    public:
        const char Separator;
        const std::string Quote;

        CsvFile(const std::string& filename, const char separator = ';', const std::string& quote = std::string("")) :
            Separator(separator),
            Quote(quote),
            oldLocale(std::locale::global(newLocale)),
            os(std::ofstream())
        {
            os.exceptions(std::ios::failbit | std::ios::badbit);
            os.open(filename);
        }

        ~CsvFile()
        {
            Flush();
            os.close();
            std::locale::global(oldLocale);
        }

        void Flush()
        {
            os.flush();
        }

        void EndRow()
        {
            // erase last separator
            auto pos = os.tellp();
            pos -= 1;
            os.seekp(pos);

            // add end of line
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
            auto ss = std::stringstream();
            ss.imbue(newLocale);
            ss.precision(std::streamsize(10));
            ss << std::defaultfloat << val;
            os << ss.str() << Separator;
            return *this;
        }

        CsvFile& operator << (const double& val)
        {
            auto ss = std::stringstream();
            ss.imbue(newLocale);
            ss.precision(std::streamsize(16));
            ss << std::defaultfloat << val;
            os << ss.str() << Separator;
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

    inline static auto ReadFileToString(const std::string& fileName)
    {
        auto file = std::ifstream(fileName);

        if (!file.bad() && file.is_open())
        {
            auto oss = std::ostringstream{};
            oss << file.rdbuf();
            file.close();
            return oss.str();
        }

#ifndef NDEBUG
        std::cerr << std::string("CsvFile::") << std::string("ReadFileToString") << std::string("(const std::string& fileName)  -  ") << fileName << std::string("  -  Could not open the file") << std::endl;
#endif

        return std::string("");
    }
}
