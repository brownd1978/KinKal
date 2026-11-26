#include "KinKal/Fit/Domain.hh"
namespace KinKal {
  std::shared_ptr<Domain> Domain::clone(CloneContext& context) const{
    auto rv = std::make_shared<Domain>(*this);
    return rv;
  }
}
