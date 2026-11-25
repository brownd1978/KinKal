//
// Test covariance transport when extrapolating through a field gradient.
// 1) check dPardBdB (the parameter Jacobian WRT the nominal field, at fixed particle state) against rebuilding the trajectory
//    from the same state in a perturbed field.  For a field not along the local z axis, the rebuild uses a different local
//    frame, which adds a time-independent (gauge) term to the Jacobian.  The covariance transport only uses the difference
//    dPardBdB(t1)-dPardBdB(t2), where that cancels, so that is what is compared
// 2) extrapolate a trajectory (backwards in time) through the constant gradient of GradientBFieldMap using the same
//    Domain/DomainWall chain as Track::extrapolate, and compare the projected position covariance on a plane against the spread of
//    an ensemble of trajectories sampled from the initial covariance and extrapolated through the same field.  That is done with
//    and without the domain gradient; only the gradient transport is expected to match the ensemble.
//
#include "KinKal/Trajectory/LoopHelix.hh"
#include "KinKal/Trajectory/CentralHelix.hh"
#include "KinKal/Trajectory/ParticleTrajectory.hh"
#include "KinKal/General/BFieldMap.hh"
#include "KinKal/Geometry/Plane.hh"
#include "KinKal/Fit/Domain.hh"
#include "KinKal/Fit/DomainWall.hh"
#include "TRandom3.h"
#include "Math/CholeskyDecomp.h"
#include <iostream>
#include <string>
#include <cmath>
#include <memory>
#include <stdexcept>

using namespace KinKal;

namespace {
  int nfail(0);
  void check(bool ok, std::string const& what) {
    if(!ok){ ++nfail; std::cout << "FAILED: " << what << std::endl; }
  }

  // test geometry: start at z=0 moving to +z, extrapolate backwards to zplane_ through a field rising from 1 to 2 Tesla
  const double zplane = -1200.0;
  const double tstart = 0.0;
  const double domaindt = 0.05; // domain size (ns)

  template <class KTRAJ> void checkdPardBdB(std::string const& name) {
    for(double tilt : {0.0, 0.05}){
      VEC3 bnom(sin(tilt)*cos(0.7), sin(tilt)*sin(0.7), cos(tilt)); bnom *= 1.5;
      VEC4 pos(300.,200.,-400.,5.0);
      MOM4 mom(-60.,55.,73.,0.511);
      KTRAJ traj(pos,mom,-1,bnom,TimeRange(0,10));
      double t1 = 5.3, t2 = 4.6;
      DPDV dpdb = traj.dPardBdB(t1) - traj.dPardBdB(t2);
      double eps = 1e-5, maxdiff(0.0);
      for(int k=0;k<3;k++){
        VEC3 db(k==0?eps:0,k==1?eps:0,k==2?eps:0);
        DVEC truth;
        for(int isign : {1,-1}){
          KTRAJ tr1(traj.state(t1),bnom+isign*db), tr2(traj.state(t2),bnom+isign*db);
          tr1.syncPhi0(traj); tr2.syncPhi0(traj);
          truth += isign*(tr1.params().parameters()-tr2.params().parameters())/(2*eps);
        }
        double scale = 1e-6;
        for(size_t ipar=0;ipar<NParams();ipar++) scale = std::max(scale,fabs(truth[ipar]));
        for(size_t ipar=0;ipar<NParams();ipar++)
          maxdiff = std::max(maxdiff,fabs(dpdb(ipar,k)-truth[ipar])/scale);
      }
      std::cout << name << " tilt " << tilt << " dPardBdB max relative difference to rebuild " << maxdiff << std::endl;
      check(maxdiff < 1e-3, name + " dPardBdB");
    }
  }

  // extrapolate backwards from tstart until zplane is passed, adding domains and walls as Track::extrapolate does.
  // A zero bfield gradient gives the domain field only
  template <class KTRAJ> void extrapolate(ParticleTrajectory<KTRAJ>& ptraj, BFieldMap const& bfield, bool usegrad) {
    // the initial domain: the field at the start, with no gradient (as fit domains)
    auto nextdom = std::make_shared<Domain>(TimeRange(tstart-domaindt,tstart+domaindt),ptraj.front().bnom());
    double time = nextdom->begin();
    // set the initial piece range to the initial domain
    ptraj.front().setRange(nextdom->range());
    unsigned ndomain(0);
    while(ptraj.front().position3(ptraj.range().begin()).Z() > zplane){
      if(++ndomain > 1000) throw std::runtime_error("extrapolation didn't reach the plane");
      TimeRange range(time-domaindt,time);
      VEC3 midpos = ptraj.front().position3(range.mid());
      auto dom = usegrad ? std::make_shared<Domain>(range,bfield.fieldVect(midpos),bfield.fieldGrad(midpos)) :
        std::make_shared<Domain>(range,bfield.fieldVect(midpos));
      DomainWall<KTRAJ> dw(dom,nextdom,ptraj);
      dw.extrapolate(ptraj,TimeDir::backwards);
      nextdom = dom;
      time = dom->begin();
    }
  }

  template <class KTRAJ> double crossing(ParticleTrajectory<KTRAJ> const& ptraj) {
    double tlow = ptraj.range().begin(), thigh = ptraj.range().end();
    for(int i=0;i<100;i++){
      double tmid = 0.5*(tlow+thigh);
      if(ptraj.position3(tmid).Z() < zplane) tlow = tmid; else thigh = tmid;
    }
    return 0.5*(tlow+thigh);
  }

  template <class KTRAJ> void checkEnsemble(std::string const& name) {
    // field from 2 Tesla at zplane to 1 Tesla at z=0
    GradientBFieldMap bfield(2.0,1.0,zplane,0.0);
    VEC4 pos(-450., 300., 0.0, tstart); // circle center well away from the axis, as needed by CentralHelix
    VEC3 bstart = bfield.fieldVect(pos.Vect());
    // pt/pz is chosen so the particle isn't reflected by the magnetic mirror of the rising field
    double pt = 250.*0.299792458*bstart.R(), pz=150.;
    MOM4 mom(pt*sin(0.3), -pt*cos(0.3), pz, 0.511);
    KTRAJ start(pos,mom,-1,bstart,TimeRange(tstart-domaindt,tstart+domaindt));
    // initial covariance: 0.1 mm position and 0.03 MeV momentum, plus a 0.04 ns t0 error (time along the trajectory, which in a
    // gradient samples a different field).  These are ~10 times smaller than typical fit errors, so the ensemble stays in the linear
    // regime; for typical errors, second-order effects are comparable to the (strongly cancelling) linear result
    auto pstate = start.state(tstart);
    DMAT scov;
    for(int i=0;i<3;i++){ scov(i,i) = 1.0e-2; scov(i+3,i+3) = 9.0e-4; }
    KTRAJ seed(ParticleStateEstimate(pstate,scov),bstart,start.range());
    auto pcov = seed.params().covariance();
    pcov(KTRAJ::t0Index(),KTRAJ::t0Index()) += 1.6e-3;
    seed.params().covariance() = pcov;
    // linear transport with and without the gradient
    PMAT plcov[2]; VEC3 xcross;
    for(int igrad=0;igrad<2;igrad++){
      ParticleTrajectory<KTRAJ> ptraj(seed);
      extrapolate(ptraj,bfield,igrad==1);
      double tc = crossing(ptraj);
      xcross = ptraj.position3(tc);
      Plane plane(VEC3(0,0,1),VEC3(1,0,0),xcross);
      plcov[igrad] = ptraj.nearestPiece(tc).planeCovariance(tc,plane,1e-10);
    }
    // ensemble
    ROOT::Math::CholeskyDecomp<double,NParams()> chol(pcov);
    PSMAT lmat; check(chol.getL(lmat), name + " Cholesky decomposition"); // L must be a full (not symmetric) matrix
    TRandom3 rng(23);
    double sx(0),sy(0),sxx(0),syy(0),sxy(0); int nsample(0);
    for(int isample=0;isample<2000;isample++){
      DVEC gaus; for(size_t ipar=0;ipar<NParams();ipar++) gaus[ipar] = rng.Gaus();
      KTRAJ sample(seed);
      sample.params().parameters() += lmat*gaus;
      sample.params().covariance() = DMAT(); // covariance isn't used
      ParticleTrajectory<KTRAJ> ptraj(sample);
      extrapolate(ptraj,bfield,false);
      auto xs = ptraj.position3(crossing(ptraj));
      if(fabs(xs.Z()-zplane) > 1e-3) continue;
      double dx = xs.X()-xcross.X(), dy = xs.Y()-xcross.Y();
      sx += dx; sy += dy; sxx += dx*dx; syy += dy*dy; sxy += dx*dy; ++nsample;
    }
    double mx = sx/nsample, my = sy/nsample;
    double sigx = sqrt(sxx/nsample-mx*mx), sigy = sqrt(syy/nsample-my*my);
    double corr = (sxy/nsample-mx*my)/(sigx*sigy);
    for(int igrad=0;igrad<2;igrad++){
      double su = sqrt(plcov[igrad](0,0)), sv = sqrt(plcov[igrad](1,1));
      std::cout << name << (igrad==0 ? " no gradient  " : " gradient     ") << " sigU " << su << " sigV " << sv
        << " corr " << plcov[igrad](0,1)/(su*sv) << std::endl;
    }
    std::cout << name << " ensemble (" << nsample << ") sigU " << sigx << " sigV " << sigy << " corr " << corr << std::endl;
    check(nsample > 1900, name + " ensemble crossings");
    // The gradient transport is the exact linearization of the domain chain, so it should match the ensemble within statistics
    // (~2% for 2000 samples), although the result is the residual of a strong cancellation. The no-gradient transport is far off
    double su = sqrt(plcov[1](0,0)), sv = sqrt(plcov[1](1,1));
    check(fabs(su/sigx-1.0) < 0.1 && fabs(sv/sigy-1.0) < 0.1, name + " gradient transport vs ensemble");
    check(sqrt(plcov[0](0,0)) > 5*sigx, name + " no-gradient transport overestimate");
    check(fabs(plcov[1](0,1)/(su*sv) - corr) < 0.05, name + " gradient transport correlation vs ensemble");
  }
}

int main(int argc, char** argv){
  checkdPardBdB<LoopHelix>("LoopHelix");
  checkdPardBdB<CentralHelix>("CentralHelix");
  checkEnsemble<LoopHelix>("LoopHelix");
  checkEnsemble<CentralHelix>("CentralHelix");
  if(nfail > 0){
    std::cout << nfail << " ExtrapolationCovariance tests failed" << std::endl;
    return -1;
  }
  std::cout << "ExtrapolationCovariance tests passed" << std::endl;
  return 0;
}
