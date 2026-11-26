// particle state with estimated errors (covariance), used to seed a fit,
// or to express a fit result as a particle state with errors.  Transforming between ParticleStateEstimate
// and a kinematic trajectory is lossless.
#ifndef KinKal_ParticleStateEstimate_hh
#define KinKal_ParticleStateEstimate_hh
#include "KinKal/General/ParticleState.hh"
#include "KinKal/General/MomBasis.hh"

namespace KinKal {
  class ParticleStateEstimate : public ParticleState {
    public:
      // construct from from raw information
      ParticleStateEstimate(ParticleState const& state, DMAT const& scovar) : ParticleState(state), scovar_(scovar) {}
      ParticleStateEstimate(SVEC6 const& state, DMAT const& scovar,double time, double mass, int charge) : ParticleState(state, time, mass, charge), scovar_(scovar) {}
      ParticleStateEstimate() {}
      DMAT const& stateCovariance() const { return scovar_; }
      // project the variance onto the scalar momentum
      double momentumVariance() const;
      // project the position variance in given direction.  Note this will throw if given the momentum direction, as that variance is infinite
      double positionVariance(MomBasis::Direction dir) const;
      // project the position variance onto the Plane defined by the perp direction (u) and the phi direction (v)
      PMAT planeCovariance() const;
      // project the position covariance onto a plane (normal norm, orthogonal in-plane directions udir and vdir) along the momentum direction.
      // This is the covariance of the point where the particle crosses the plane: position errors along the momentum direction drop out.
      // |cos| between the momentum direction and the plane normal must exceed dirtol
      PMAT planeCovariance(VEC3 const& norm, VEC3 const& udir, VEC3 const& vdir, double dirtol) const;
      //
      // Transport through vacuum in a constant magnetic field bnom (Tesla), by time dt (ns, either sign).  The particle follows the exact helix,
      // as the kinematic trajectories do within a domain.
      ParticleState transportState(VEC3 const& bnom, double dt) const;
      // Jacobian of that transport: d(state at time+dt)/d(state at time), with the state ordered as (position, momentum).
      // The time step is fixed, so errors along the trajectory are kept in the transported state covariance
      PSMAT transportJacobian(VEC3 const& bnom, double dt) const;
      // The same, including the field gradient grad (dB_i/dx_j, Tesla/mm): a perturbed particle sees the field bnom + grad dx(time+dt/2),
      // where dx(time+dt/2) is its displacement half way through the step.  This mirrors Domain::paramTransport for the trajectory parameters
      PSMAT transportJacobian(VEC3 const& bnom, double dt, SSMAT const& grad) const;
      // transport the state and its covariance (J C J^T), without and with the field gradient
      ParticleStateEstimate transport(VEC3 const& bnom, double dt) const;
      ParticleStateEstimate transport(VEC3 const& bnom, double dt, SSMAT const& grad) const;
    private:
      DMAT scovar_; // covariance of state vector
  };
}
#endif
