#ifndef UTIL_ENVIRONMENT_VARIABLE_HPP
#define UTIL_ENVIRONMENT_VARIABLE_HPP

#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>

class EnvironmentVariableException : public std::exception {
public:
  EnvironmentVariableException(const std::string &varName)
      : varName_(varName) {}
  virtual const char *what() const throw() {
    // we leak message memory here
    // if caught, catcher can free
    // otherwise, program terminates anyway
    auto msgStr = varName_ + std::string(" was unset with no default");
    char *leak = new char[msgStr.size() + 1];
    msgStr.copy(leak, msgStr.size());
    leak[msgStr.size()] = 0;
    return leak;
  }

private:
  std::string varName_;
};

template <typename T> class EnvironmentVariable {
public:
  EnvironmentVariable(const std::string &varName, T defaultVal)
      : varName_(varName), defaultVal_(defaultVal), hasDefaultVal_(true) {}
  EnvironmentVariable(const std::string &varName)
      : varName_(varName), hasDefaultVal_(false) {}

  T get() {
    auto raw = read_env(varName_);
    if (raw == "") {
      if (hasDefaultVal_) {
        return defaultVal_;
      } else {
        throw EnvironmentVariableException(varName_);
      }
    } else {
      return convert(raw);
    }
  }

private:
  static std::string read_env(const std::string &varName) {
    const char *fromEnv = std::getenv(varName.c_str());
    if (fromEnv && (0 < std::strlen(fromEnv))) {
      return std::string(fromEnv);
    } else { /* not found or empty */
      return "";
    }
  }

  T convert(const std::string &raw);

  std::string varName_;
  T defaultVal_;
  bool hasDefaultVal_;
};

template <>
inline bool EnvironmentVariable<bool>::convert(const std::string &raw) {
  if (raw == "0" || raw == "false" || raw == "FALSE" || raw == "False") {
    return false;
  } else {
    return true;
  }
}

template <>
inline int EnvironmentVariable<int>::convert(const std::string &raw) {
  return std::atoi(raw.c_str());
}

template <>
inline uint32_t EnvironmentVariable<uint32_t>::convert(const std::string &raw) {
  return std::strtoul(raw.c_str(), nullptr /*ignore unparsed characters*/, 10);
}

template <>
inline float EnvironmentVariable<float>::convert(const std::string &raw) {
  return std::atof(raw.c_str());
}

template <>
inline std::string
EnvironmentVariable<std::string>::convert(const std::string &raw) {
  return raw;
}

#endif