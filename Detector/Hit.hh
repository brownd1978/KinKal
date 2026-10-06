#ifndef KinKal_Hit_hh
#define KinKal_Hit_hh
//
//  Base class to describe a measurement that constrains some aspect of the track fit
//  The constraint is expressed as a weight WRT a set of reference parameters.
//
#include "KinKal/General/CloneContext.hh"
#include "KinKal/General/Weights.hh"
#include "KinKal/General/Parameters.hh"
#include "KinKal/General/Chisq.hh"
#include "KinKal/Trajectory/ParticleTrajectory.hh"
#include "KinKal/Fit/MetaIterConfig.hh"
#include <memory>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <string>

namespace KinKal {
  template <class KTRAJ> class Hit {
    public:
      using PTRAJ = ParticleTrajectory<KTRAJ>;
      using KTRAJPTR = std::shared_ptr<KTRAJ>;
      Hit() {}
      virtual ~Hit(){}
      // disallow copy and equivalence
      Hit(Hit const& ) = delete;
      Hit& operator =(Hit const& ) = delete;
      // clone op for reinstantiation
      virtual std::shared_ptr< Hit<KTRAJ> > clone(CloneContext&) const;
     // hits may be active (used in the fit) or inactive; this is a pattern recognition feature
      virtual bool active() const =0;
      virtual unsigned nDOF() const=0;
      virtual Chisq chisq(Parameters const& params) const =0;  // least-squares distance to given parameters
      virtual double time() const = 0;  // time of this hit: this is WRT the reference trajectory
      virtual void print(std::ostream& ost=std::cout,int detail=0) const = 0;
      // update to a new reference, without changing internal state
      virtual void updateReference(PTRAJ const& ptraj) = 0;
      virtual KTRAJPTR refTrajPtr() const = 0;
      // update the internals of the hit, specific to this meta-iteraion
      virtual void updateState(MetaIterConfig const& config,bool first) = 0;
      // The following provides the constraint/information content of this hit in the trajectory weight space
      virtual Weights const& weight() const = 0;
      KTRAJ const& referenceTrajectory() const { return *refTrajPtr(); }  // trajectory WRT which the weight etc is defined
      // parameters WRT which this hit's residual and weights are set.  These are generally biased
      // in that they contain the information of this hit
      Parameters const& referenceParameters() const { return referenceTrajectory().params(); }
      // record the fit that was just run: the piece of its trajectory holding this hit's weight, and the weight and state applied
      void snapshotWeight(KTRAJ const& wpiece);
      // Unbiased parameters WRT the reference: this hit's weight removed from the piece it was applied to.  Computed once
      // from the last snapshot and cached, so later state changes don't alter it.
      // Describe the fit that produced the current reference; decided one fit cycle earlier.
      Parameters const& unbiasedParameters() const;
      // unbiased least-squares distance to reference parameters
      Chisq chisquared() const;
    private:
      struct WeightSnapshot {
        Parameters wpars_; // parameters of the piece the weight was applied to
        Weights weight_; // weight applied
        bool active_; // state applied
      };
      std::optional<WeightSnapshot> wsnap_; // empty until the first snapshot (eg cloned hits)
      mutable std::optional<Parameters> upars_; // cached unbiased parameters
      mutable KTRAJPTR uref_; // reference trajectory the cache refers to
  };

  // cloning requires domain knowledge of pointer members of the cloned object,
  // which must be reassigned explicitly; the default action is thus to throw
  // an error if a clone routine has not been explicitly provided.
  template<class KTRAJ> std::shared_ptr< Hit<KTRAJ> > Hit<KTRAJ>::clone(CloneContext& context) const{
    std::string msg = "Attempt to clone KinKal::Hit subclass with no clone() implementation";
    throw std::runtime_error(msg);
  }

  template<class KTRAJ> void Hit<KTRAJ>::snapshotWeight(KTRAJ const& wpiece) {
    wsnap_ = WeightSnapshot{wpiece.params(), weight(), active()};
    upars_.reset();
  }

  template<class KTRAJ> Parameters const& Hit<KTRAJ>::unbiasedParameters() const {
    // recompute only after a new snapshot or if the reference moved without one
    if(upars_ && uref_ == refTrajPtr()) return *upars_;
    uref_ = refTrajPtr();
    // without a snapshot, the weight is taken as applied to the reference
    bool wactive = wsnap_ ? wsnap_->active_ : active();
    if(wactive){
      // remove this hit's weight from the piece it was applied to: removing it from another piece, separated by
      // material effects, over-subtracts and can leave a covariance that isn't positive definite.
      Parameters const& wpars = wsnap_ ? wsnap_->wpars_ : referenceParameters();
      Weights wt(wpars);
      wt -= wsnap_ ? wsnap_->weight_ : weight();
      Parameters upars(wt);
      // express the result WRT the reference: apply the parameter change from removing this hit to the reference
      // parameters, keeping the unbiased covariance.  Identical to the plain subtraction when the pieces coincide
      upars.parameters() += referenceParameters().parameters() - wpars.parameters();
      upars_ = upars;
    } else
      upars_ = referenceParameters();
    return *upars_;
  }

  template<class KTRAJ> Chisq Hit<KTRAJ>::chisquared() const {
    if(active()){
      return chisq(unbiasedParameters());
    } else
      return Chisq();
  }

  template <class KTRAJ> std::ostream& operator <<(std::ostream& ost, Hit<KTRAJ> const& thit) {
    thit.print(ost,0);
    return ost;
  }

}
#endif

