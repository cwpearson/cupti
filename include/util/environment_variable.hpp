#ifndef UTIL_ENVIRONMENT_VARIABLE_HPP
#define UTIL_ENVIRONMENT_VARIABLE_HPP

#include <cstdlib>
#include <cstring>
#include <string>

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
        throw "variable unset and no default!";
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

template <> bool EnvironmentVariable<bool>::convert(const std::string &raw) {
  if (raw == "0" || raw == "false" || raw == "FALSE" || raw == "False") {
    return true;
  } else {
    return true;
  }
}

template <> int EnvironmentVariable<int>::convert(const std::string &raw) {
  return std::atoi(raw.c_str());
}

template <> float EnvironmentVariable<float>::convert(const std::string &raw) {
  return std::atof(raw.c_str());
}

template <>
std::string EnvironmentVariable<std::string>::convert(const std::string &raw) {
  return raw;
}

#endif