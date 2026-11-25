#ifndef KinKal_Domain_hh
#define KinKal_Domain_hh
//
//  domain used to compute magnetic field corrections. Magnetic bending not described by the intrinsic parameterization is assumed
//  to be negligible over the domain.
//  A domain can optionally carry the field gradient (dB_i/dx_j) at its middle. That is used to propagate the trajectory covariance
//  across the domain: a perturbed trajectory sees a different field than the nominal, which the fixed nominal field (bnom) of the
//  domain doesn't describe. This matters when extrapolating through strong gradients; it is ignored if the gradient is null (default).
//
#include <memory>
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
      Domain(TimeRange const& range, VEC3 const& bnom, Grad const& grad) : range_(range), bnom_(bnom), grad_(grad), hasgrad_(true) {}
      // clone op for reinstantiation
      Domain(Domain const&);
      std::shared_ptr< Domain > clone(CloneContext&) const;
      bool operator < (Domain const& other) const {return begin() < other.begin(); }
      auto const& range() const { return range_; }
      // forward range functions
      double begin() const { return range_.begin();}
      double end() const { return range_.end();}
      double mid() const { return range_.mid();}
      auto const& bnom() const { return bnom_; }
      auto const& grad() const { return grad_; }
      bool hasGradient() const { return hasgrad_; }
      // Parameter-space covariance transport factor K for a trajectory piece with this domain's bnom, constructed (ie with its covariance
      // valid) at time tstart, moving to time tend, through the field gradient of this domain.  The covariance at tend is K C K^T.
      // To first order the perturbed trajectory sees B = bnom + G dx(tmid), with dx(tmid) = dXdPar(tmid) dp.  Two pieces in that field,
      // one matching the nominal state at tstart and one at tend, differ in parameters by (D(tstart)-D(tend)) dB, where D = dPardBdB.
      // So K = 1 + (D(tstart)-D(tend)) G dXdPar(tmid).  The interval is restricted to this domain; the identity is returned if this domain
      // has no gradient or the interval is empty.
      template <class KTRAJ> PSMAT paramTransport(KTRAJ const& piece, double tstart, double tend) const;
      void updateBNom( VEC3 const& bnom) { bnom_ = bnom; }; // used in DomainWall updating
      void setEdge(TimeDir tdir, double time) { range_.extend(time,tdir); } // move the end (forwards) or begin (backwards) to time; used to truncate a track's outermost domain
    private:
      TimeRange range_; // range of this domain
      VEC3 bnom_; // nominal BField for this domain
      Grad grad_; // field gradient dB_i/dx_j at the domain middle, used for covariance transport
      bool hasgrad_ = false; // was the gradient set?
  };
  inline std::ostream& operator <<(std::ostream& ost, KinKal::Domain const& domain) {
    ost << "Domain with " << domain.range() << " and Field " << domain.bnom() << std::endl;
    return ost;
  }
}
namespace KinKal {
  template <class KTRAJ> PSMAT Domain::paramTransport(KTRAJ const& piece, double tstart, double tend) const {
    PSMAT retval = ROOT::Math::SMatrixIdentity();
    if(!hasgrad_) return retval;
    tstart = std::clamp(tstart,begin(),end());
    tend = std::clamp(tend,begin(),end());
    if(tstart == tend) return retval;
    double tmid = 0.5*(tstart+tend);
    retval += (piece.dPardBdB(tstart)-piece.dPardBdB(tend))*grad_*piece.dXdPar(tmid);
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
}
#endif

