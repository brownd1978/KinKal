#ifndef KinKal_Domain_hh
#define KinKal_Domain_hh
//
//  domain used to compute magnetic field corrections. Magnetic bending not described by the intrinsic parameterization is assumed
//  to be negligible over the domain.
//  A domain can optionally carry the field gradient (dB_i/dx_j) at its middle. That is used to propagate the trajectory covariance
//  across the domain: a perturbed trajectory sees a different field than the nominal, which the fixed nominal field (bnom) of the
//  domain doesn't describe. This matters when extrapolating through strong gradients; it is ignored if the gradient is not set (default).
//  The transport is skipped where the momentum is nearly perpendicular to the field (|cos(p,bnom)| below mincos), as happens near a
//  magnetic-mirror turning point.  There the LoopHelix t0 and phi0 (defined where the helix crosses local z=0) are ill-conditioned
//  (dt0 ~ z/lambda), so far from z=0 the parameter covariance carries huge, nearly cancelling terms, and the transport would
//  amplify their rounding errors into the physical (position, momentum) covariance.
//
#include <memory>
#include <optional>
#include <cmath>
#include "KinKal/General/CloneContext.hh"
#include "KinKal/General/TimeRange.hh"
#include "KinKal/General/TimeDir.hh"
#include "KinKal/General/Vectors.hh"
#include "KinKal/General/BFieldMap.hh"
#include "KinKal/General/ParticleStateEstimate.hh"
namespace KinKal {
  class Domain {
    public:
      using Grad = BFieldMap::Grad;
      Domain(double lowtime, double range, VEC3 const& bnom) : range_(lowtime,lowtime+range), bnom_(bnom) {}
      Domain(TimeRange const& range, VEC3 const& bnom) : range_(range), bnom_(bnom) {}
      Domain(TimeRange const& range, VEC3 const& bnom, Grad const& grad, double mincos=0.0) : range_(range), bnom_(bnom), grad_(grad), mincos_(mincos) {}
      // clone op for reinstantiation
      Domain(Domain const&) = default;
      std::shared_ptr< Domain > clone(CloneContext&) const;
      bool operator < (Domain const& other) const {return begin() < other.begin(); }
      auto const& range() const { return range_; }
      // forward range functions
      double begin() const { return range_.begin();}
      double end() const { return range_.end();}
      double mid() const { return range_.mid();}
      auto const& bnom() const { return bnom_; }
      auto const& grad() const { return grad_; } // empty if no gradient was set
      bool hasGradient() const { return grad_.has_value(); }
      double minCos() const { return mincos_; }
      // pieces built at the walls of this domain during extrapolation use a local reference (see LoopHelix::LocalReference), if the
      // trajectory supports that
      bool localReference() const { return localref_; }
      void setLocalReference(bool localref) { localref_ = localref; }
      // Parameter-space covariance transport factor K for a trajectory piece with this domain's bnom, constructed (ie with its covariance
      // valid) at time tstart, moving to time tend, through the field gradient of this domain.  The covariance at tend is K C K^T.
      // To first order the perturbed trajectory sees B = bnom + G dx(tmid), with dx(tmid) = dXdPar(tmid) dp.  Two pieces in that field,
      // one matching the nominal state at tstart and one at tend, differ in parameters by (D(tstart)-D(tend)) dB, where D = dPardBdB.
      // So K = 1 + (D(tstart)-D(tend)) G dXdPar(tmid).  The interval is restricted to this domain; the identity is returned if this domain
      // has no gradient, the interval is empty, or |cos| between the piece direction and bnom is below mincos at either end.
      template <class KTRAJ> PSMAT paramTransport(KTRAJ const& piece, double tstart, double tend) const;
      void updateBNom( VEC3 const& bnom) { bnom_ = bnom; }; // used in DomainWall updating
      void setEdge(TimeDir tdir, double time) { range_.extend(time,tdir); } // move the end (forwards) or begin (backwards) to time; used to truncate a track's outermost domain
    private:
      TimeRange range_; // range of this domain
      VEC3 bnom_; // nominal BField for this domain
      std::optional<Grad> grad_; // field gradient dB_i/dx_j at the domain middle, used for covariance transport (empty if not set)
      double mincos_ = 0.0; // minimum |cos(direction,bnom)| for the covariance transport
      bool localref_ = false; // build extrapolated pieces with a local reference
  };
  inline std::ostream& operator <<(std::ostream& ost, KinKal::Domain const& domain) {
    ost << "Domain with " << domain.range() << " and Field " << domain.bnom() << std::endl;
    return ost;
  }
}
namespace KinKal {
  template <class KTRAJ> PSMAT Domain::paramTransport(KTRAJ const& piece, double tstart, double tend) const {
    PSMAT retval = ROOT::Math::SMatrixIdentity();
    if(!grad_) return retval;
    tstart = std::clamp(tstart,begin(),end());
    tend = std::clamp(tend,begin(),end());
    if(tstart == tend) return retval;
    if(mincos_ > 0.0){
      auto bdir = bnom_.Unit();
      if(fabs(piece.direction(tstart).Dot(bdir)) < mincos_ || fabs(piece.direction(tend).Dot(bdir)) < mincos_) return retval;
    }
    double tmid = 0.5*(tstart+tend);
    retval += (piece.dPardBdB(tstart)-piece.dPardBdB(tend))*(*grad_)*piece.dXdPar(tmid);
    return retval;
  }
  // State estimate of an extrapolation piece at the given time, including covariance transport through the domain field gradient.
  // The piece covariance is valid where the piece was constructed: its beginning (end) when extrapolating forwards (backwards).
  // A null domain, or one without a gradient, gives the plain state estimate.
  template <class KTRAJ> ParticleStateEstimate transportedStateEstimate(KTRAJ const& piece, Domain const* domain, double time, TimeDir tdir) {
    if(domain != nullptr && domain->hasGradient()){
      double tstart = tdir == TimeDir::forwards ? piece.range().begin() : piece.range().end();
      auto ptrans = domain->paramTransport(piece,tstart,time);
      if(ptrans != PSMAT(ROOT::Math::SMatrixIdentity())){
        KTRAJ tpiece(piece);
        tpiece.params().covariance() = ROOT::Math::Similarity(ptrans,piece.params().covariance());
        return tpiece.stateEstimate(time);
      }
    }
    return piece.stateEstimate(time);
  }
  // build an extrapolated piece from a state estimate, using a local reference if the domain requests it and the trajectory supports it
  template <class KTRAJ> KTRAJ extrapolatedPiece(ParticleStateEstimate const& pstate, VEC3 const& bnom, TimeRange const& range, Domain const* domain) {
    if constexpr (requires { typename KTRAJ::LocalReference; }) {
      if(domain != nullptr && domain->localReference()) return KTRAJ(pstate,bnom,range,typename KTRAJ::LocalReference{});
    }
    return KTRAJ(pstate,bnom,range);
  }
}
#endif

