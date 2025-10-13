#include "polymetis/polymetis_server.hpp"

Instrumentor::Instrumentor() {}
Instrumentor &Instrumentor::Instance() {
  static Instrumentor instance;
  return instance;
}
Instrumentor::~Instrumentor() { endSession(); }