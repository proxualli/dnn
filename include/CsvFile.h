#pragma once
#include <iostream>
#include <locale>
#include <fstream>
#include <sstream>
#include <vector>


class CsvFile
{
private:
    struct no_separator : std::numpunct<char>
    {
    protected:
        virtual char do_decimal_point() const
        {
            return ',';
        }
        virtual char do_thousands_sep() const
        {
            return '.';
        }
        virtual std::string do_grouping() const
        {
            return std::string("");
        }
    };

    const std::locale newLocale = std::locale(std::locale(""), new no_separator());
    const std::locale oldLocale;
    std::ofstream os;

public:
    const char Separator;
    const std::string Quote;

    CsvFile(const std::string& filename, const char separator = ';', const std::string& quote = "") :
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
        // end of line
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
        auto stream = std::stringstream();
        stream.imbue(newLocale);
        stream << std::setprecision(std::streamsize(10)) << std::fixed << val;
        os << stream.str() << Separator;
        return *this;
    }

    CsvFile& operator << (const double& val)
    {
        auto stream = std::stringstream();
        stream.imbue(newLocale);
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
