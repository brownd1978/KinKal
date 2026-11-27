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
// 3) the same through a magnetic-mirror reflection: the trajectory turns around in the rising field, and the covariance is compared on
//    a plane crossed after the turn.  Near local z=0 the transport (without the |cos(p,B)| guard) still matches the ensemble.  Far from
//    z=0 the LoopHelix t0 and phi0 are ill-conditioned near the turning point (dt0 ~ z/lambda), and the domain chain loses precision
//    with or without the transport; that case is printed for information only.  The guard itself is checked directly.
// 4) as a reference, the covariance is also transported in physical (x,p) coordinates along the same domain chain, using the exact
//    constant-field helix Jacobian in each domain (plus the domain-field change from the gradient), with no reference to the
//    trajectory parameters.  It is checked against the ensemble near z=0, and checked to give the same result at z=-4000, where the
//    parameter chain fails.
// 5) LoopHelix with a local reference (LoopHelix::LocalReference, set per Domain): extrapolated pieces define t0 and phi0 where the helix
//    crosses the local z of the piece starting point, instead of z=0.  Scanned in the distance of the turn from z=0, the gradient transport
//    then matches the physical transport, at the crossing plane and in the full state covariance of the piece at the turn.  The
//    representation itself is also checked near pz=0 (round trip and one-domain step vs the exact helix) as a function of lambda and z:
//    with the local reference the precision doesn't depend on z, leaving the intrinsic ~1/lambda^2 loss (1e-5 at lambda = 0.001 mm).
// 6) referenceTime and zTime: the trajectory t0 and z crossing times don't depend on the local reference, and zTime is exact in a tilted field
//
#include "KinKal/Trajectory/LoopHelix.hh"
#include "KinKal/Trajectory/CentralHelix.hh"
#include "KinKal/Trajectory/KinematicLine.hh"
#include "KinKal/Trajectory/ParticleTrajectory.hh"
#include "KinKal/General/BFieldMap.hh"
#include "KinKal/Geometry/Plane.hh"
#include "KinKal/Fit/Domain.hh"
#include "KinKal/Fit/DomainWall.hh"
#include "KinKal/Fit/Config.hh"
#include "TRandom3.h"
#include "Math/CholeskyDecomp.h"
#include <iostream>
#include <string>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <cstdio>
#include <limits>

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
  // magnetic mirror: extrapolate backwards from zstart (moving to +z) through the rising field until the trajectory turns around and
  // comes back up through zcross.  With zcross very negative, stop just after the turn
  template <class KTRAJ> void extrapolateMirror(ParticleTrajectory<KTRAJ>& ptraj, BFieldMap const& bfield, bool usegrad, double zcross, double mincos, bool localref=false) {
    auto nextdom = std::make_shared<Domain>(TimeRange(tstart-domaindt,tstart+domaindt),ptraj.front().bnom());
    double time = nextdom->begin();
    ptraj.front().setRange(nextdom->range());
    double zmin(1e10);
    unsigned ndomain(0);
    while(true){
      double z = ptraj.front().position3(ptraj.range().begin()).Z();
      zmin = std::min(zmin,z);
      if(zmin < zcross - 0.5 && z > zcross) break;
      if(zcross < -1e5 && ptraj.front().momentum3(ptraj.range().begin()).Z() < 0.0 && z > zmin + 1.0) break;
      if(++ndomain > 2000) throw std::runtime_error("mirror extrapolation didn't return to the plane");
      TimeRange range(time-domaindt,time);
      VEC3 midpos = ptraj.front().position3(range.mid());
      auto dom = usegrad ? std::make_shared<Domain>(range,bfield.fieldVect(midpos),bfield.fieldGrad(midpos),mincos) :
        std::make_shared<Domain>(range,bfield.fieldVect(midpos));
      dom->setLocalReference(localref);
      DomainWall<KTRAJ> dw(dom,nextdom,ptraj);
      dw.extrapolate(ptraj,TimeDir::backwards);
      nextdom = dom;
      time = dom->begin();
    }
  }
  // the post-turn crossing of zcross: the trajectory runs backwards in time, so search between its beginning and the turning point
  template <class KTRAJ> double mirrorCrossing(ParticleTrajectory<KTRAJ> const& ptraj, double zcross) {
    double tturn = ptraj.range().end(), zmin = 1e10;
    for(double t = ptraj.range().begin(); t < ptraj.range().end(); t += 0.002){
      double z = ptraj.position3(t).Z();
      if(z < zmin){ zmin = z; tturn = t; }
    }
    double tlow = ptraj.range().begin(), thigh = tturn; // z(tlow) > zcross > z(thigh)
    for(int i=0;i<100;i++){
      double tmid = 0.5*(tlow+thigh);
      if(ptraj.position3(tmid).Z() > zcross) tlow = tmid; else thigh = tmid;
    }
    return 0.5*(tlow+thigh);
  }
  // Physical (x,p) covariance transport, independent of the trajectory parameterization: the exact helix in a constant field, with
  // its Jacobian (by central differences).  This has no reference to local z=0, so it is well conditioned through pz=0
  ParticleState helixStep(ParticleState const& s, VEC3 const& bvec, double dt) {
    double bmag = bvec.R();
    VEC3 bhat = bvec/bmag;
    VEC3 mom = s.momentum3();
    double vscale = CLHEP::c_light/s.energy(); // velocity = vscale * momentum (mm/ns)
    double omega = -s.charge()*BFieldMap::cbar()*vscale*bmag; // rotation rate of the momentum about bhat, from dp/dt = q cbar v x B
    VEC3 ppar = mom.Dot(bhat)*bhat, pperp = mom - ppar, pcross = bhat.Cross(pperp);
    double phi = omega*dt;
    VEC3 newmom = ppar + cos(phi)*pperp + sin(phi)*pcross;
    VEC3 newpos = s.position3() + vscale*(ppar*dt + pperp*(sin(phi)/omega) + pcross*((1.0-cos(phi))/omega));
    return ParticleState(newpos,newmom,s.time()+dt,s.mass(),s.charge());
  }
  PSMAT helixJacobian(ParticleState const& s, VEC3 const& bvec, double dt) {
    PSMAT jac;
    for(int k=0;k<6;k++){
      double eps = k<3 ? 1.0e-3 : 1.0e-4; // mm, MeV/c
      SVEC6 dv; dv[k] = eps;
      auto sp = helixStep(ParticleState(s.state()+dv,s.time(),s.mass(),s.charge()),bvec,dt).state();
      auto sm = helixStep(ParticleState(s.state()-dv,s.time(),s.mass(),s.charge()),bvec,dt).state();
      jac.Place_in_col(SVEC6((sp-sm)/(2.0*eps)),0,k);
    }
    return jac;
  }
  // state Jacobian across part of a domain, from time to time+dt.  With a gradient, add the change of the domain field due to the
  // change of the position at the domain middle (tmid), as the extrapolation sets the domain field there
  PSMAT domainJacobian(ParticleState const& s, VEC3 const& bvec, double dt, bool usegrad, BFieldMap::Grad const& grad, double tmid) {
    auto jac = helixJacobian(s,bvec,dt);
    if(usegrad){
      static constexpr double db = 1.0e-5; // Tesla
      ROOT::Math::SMatrix<double,6,3> dsdb;
      for(int k=0;k<3;k++){
        VEC3 dbvec(k==0?db:0.0, k==1?db:0.0, k==2?db:0.0);
        SVEC6 dstate = (helixStep(s,bvec+dbvec,dt).state() - helixStep(s,bvec-dbvec,dt).state())/(2.0*db);
        dsdb.Place_in_col(dstate,0,k);
      }
      auto dxmid = helixJacobian(s,bvec,tmid-s.time()).Sub<ROOT::Math::SMatrix<double,3,6>>(0,0);
      jac += dsdb*grad*dxmid;
    }
    return jac;
  }
  // transport the seed state covariance from tstart backwards to tc along the extrapolated trajectory, domain (piece) by domain, entirely in
  // (x,p), and project it on the plane z=const at tc.  The central states come from the trajectory pieces; maxdev is the largest
  // difference between the helix step and the piece position at the domain ends (a check of the helix step)
  template <class KTRAJ> DMAT physicalStateCovariance(ParticleTrajectory<KTRAJ> const& ptraj, KTRAJ const& seed, BFieldMap const& bfield,
      bool usegrad, double tc, double& maxdev) {
    DMAT scov = seed.stateEstimate(tstart).stateCovariance();
    double time = tstart;
    maxdev = 0.0;
    int istart = ptraj.nearestIndex(tstart);
    for(int ip = istart; ip >= 0 && time > tc; --ip){
      auto const& piece = ptraj.piece(ip);
      double tlow = std::max(piece.range().begin(),tc);
      double dt = tlow - time;
      if(dt >= 0.0) continue;
      auto state = piece.state(time);
      maxdev = std::max(maxdev,(helixStep(state,piece.bnom(),dt).position3() - piece.position3(tlow)).R());
      // the initial domain has no gradient, as in extrapolateMirror
      bool grad = usegrad && ip != istart;
      double tmid = piece.range().mid();
      auto jac = domainJacobian(state,piece.bnom(),dt,grad,bfield.fieldGrad(piece.position3(tmid)),tmid);
      scov = ROOT::Math::Similarity(jac,scov);
      time = tlow;
    }
    return scov;
  }
  template <class KTRAJ> PMAT physicalCovariance(ParticleTrajectory<KTRAJ> const& ptraj, KTRAJ const& seed, BFieldMap const& bfield,
      bool usegrad, double tc, double& maxdev) {
    auto scov = physicalStateCovariance(ptraj,seed,bfield,usegrad,tc,maxdev);
    // project along the velocity onto the plane z=const, with U=x and V=y as in mirrorCovariance
    auto vel = ptraj.velocity(tc);
    ROOT::Math::SMatrix<double,2,6> proj;
    proj(0,0) = 1.0; proj(0,2) = -vel.X()/vel.Z();
    proj(1,1) = 1.0; proj(1,2) = -vel.Y()/vel.Z();
    return ROOT::Math::Similarity(proj,scov);
  }
  // projected covariance on the post-turn crossing plane, zabove the turning point, for a particle with cos(p,z) = cosz at zstart.
  // The field is translated with zstart, so the physics doesn't depend on it.  Returns the crossing position.  plcov must hold 4
  // matrices: parameter transport without and with the gradient, then physical (x,p) transport without and with the gradient
  double maxdev(0.0); // helix step check of the last physical transport
  // state covariances (gradient transport) at the end of the piece containing the turning point, where the piece was built, from the
  // parameter transport (turncov[0]) and the physical transport (turncov[1])
  DMAT turncov[2];
  double turnlam(0.0), turnzref(0.0), turnz(0.0); // lambda, reference z and z of that piece
  template <class KTRAJ> VEC3 mirrorCovariance(double cosz, double zabove, double zstart, double mincos, PMAT* plcov, KTRAJ& seed, double& zcross, bool localref=false) {
    GradientBFieldMap bfield(2.0,1.0,zplane+zstart,zstart);
    VEC4 pos(-450., 300., zstart, tstart);
    VEC3 bstart = bfield.fieldVect(pos.Vect());
    double ptot = 105.0, pt = ptot*sqrt(1.0-cosz*cosz), pz = ptot*cosz;
    MOM4 mom(pt*sin(0.3), -pt*cos(0.3), pz, 0.511);
    KTRAJ start(pos,mom,-1,bstart,TimeRange(tstart-domaindt,tstart+domaindt));
    DMAT scov;
    for(int i=0;i<3;i++){ scov(i,i) = 1.0e-2; scov(i+3,i+3) = 9.0e-4; }
    seed = KTRAJ(ParticleStateEstimate(start.state(tstart),scov),bstart,start.range());
    auto pcov = seed.params().covariance();
    pcov(KTRAJ::t0Index(),KTRAJ::t0Index()) += 1.6e-3;
    seed.params().covariance() = pcov;
    {  // find the turning point
      ParticleTrajectory<KTRAJ> ptraj(seed);
      extrapolateMirror(ptraj,bfield,false,-1e6,mincos);
      double zmin = 1e10;
      for(double t = ptraj.range().begin(); t < ptraj.range().end(); t += 0.002) zmin = std::min(zmin,ptraj.position3(t).Z());
      zcross = zmin + zabove;
    }
    VEC3 xcross;
    for(int igrad=0;igrad<2;igrad++){
      ParticleTrajectory<KTRAJ> ptraj(seed);
      extrapolateMirror(ptraj,bfield,igrad==1,zcross,mincos,localref);
      double tc = mirrorCrossing(ptraj,zcross);
      if(igrad == 1){
        double tturn = ptraj.range().end(), zmin = 1e10;
        for(double t = ptraj.range().begin(); t < ptraj.range().end(); t += 0.002){
          double z = ptraj.position3(t).Z();
          if(z < zmin){ zmin = z; tturn = t; }
        }
        auto const& tpiece = ptraj.nearestPiece(tturn);
        double tend = tpiece.range().end();
        turncov[0] = tpiece.stateEstimate(tend).stateCovariance();
        double dev;
        turncov[1] = physicalStateCovariance(ptraj,seed,bfield,true,tend,dev);
        turnlam = tpiece.paramVal(1);
        turnz = tpiece.dphi(tend)*tpiece.lam(); // local z - zref
        if constexpr (std::is_same_v<KTRAJ,LoopHelix>) turnzref = tpiece.zref(); else turnzref = 0.0;
      }
      xcross = ptraj.position3(tc);
      Plane plane(VEC3(0,0,1),VEC3(1,0,0),xcross);
      plcov[igrad] = ptraj.nearestPiece(tc).planeCovariance(tc,plane,1e-10);
      // physical (x,p) transport along the same central trajectory (which doesn't depend on the parameter transport)
      plcov[2+igrad] = physicalCovariance(ptraj,seed,bfield,igrad==1,tc,maxdev);
    }
    return xcross;
  }
  template <class KTRAJ> void checkMirror(std::string const& name, double cosz, double zabove) {
    PMAT plcov[4]; KTRAJ seed; double zcross;
    auto xcross = mirrorCovariance(cosz,zabove,0.0,0.0,plcov,seed,zcross);
    double neardev = maxdev;
    GradientBFieldMap bfield(2.0,1.0,zplane,0.0);
    ROOT::Math::CholeskyDecomp<double,NParams()> chol(seed.params().covariance());
    PSMAT lmat; check(chol.getL(lmat), name + " mirror Cholesky decomposition");
    TRandom3 rng(29);
    double sx(0),sy(0),sxx(0),syy(0); int nsample(0);
    for(int isample=0;isample<1000;isample++){
      DVEC gaus; for(size_t ipar=0;ipar<NParams();ipar++) gaus[ipar] = rng.Gaus();
      KTRAJ sample(seed);
      sample.params().parameters() += lmat*gaus;
      sample.params().covariance() = DMAT();
      ParticleTrajectory<KTRAJ> ptraj(sample);
      try { extrapolateMirror(ptraj,bfield,false,zcross,0.0); } catch(std::exception const&) { continue; }
      auto xs = ptraj.position3(mirrorCrossing(ptraj,zcross));
      if(fabs(xs.Z()-zcross) > 1e-3) continue;
      double dx = xs.X()-xcross.X(), dy = xs.Y()-xcross.Y();
      sx += dx; sy += dy; sxx += dx*dx; syy += dy*dy; ++nsample;
    }
    double mx = sx/nsample, my = sy/nsample;
    double sigx = sqrt(sxx/nsample-mx*mx), sigy = sqrt(syy/nsample-my*my);
    auto print = [](PMAT const& cov){ return " sigU " + std::to_string(sqrt(cov(0,0))) + " sigV " + std::to_string(sqrt(cov(1,1))); };
    std::cout << name << " mirror cosz " << cosz << " plane " << zabove << " mm above the turn, ensemble (" << nsample << ") sigU " << sigx << " sigV " << sigy << std::endl
      << "   z=0:     parameter transport: no gradient" << print(plcov[0]) << ", gradient" << print(plcov[1]) << std::endl
      << "            physical transport:  no gradient" << print(plcov[2]) << ", gradient" << print(plcov[3]) << " (helix step check " << neardev << " mm)" << std::endl;
    check(nsample > 950, name + " mirror ensemble crossings");
    double su = sqrt(plcov[1](0,0)), sv = sqrt(plcov[1](1,1));
    check(fabs(su/sigx-1.0) < 0.1 && fabs(sv/sigy-1.0) < 0.1, name + " mirror gradient transport vs ensemble");
    double pu = sqrt(plcov[3](0,0)), pv = sqrt(plcov[3](1,1));
    check(fabs(pu/sigx-1.0) < 0.1 && fabs(pv/sigy-1.0) < 0.1, name + " mirror physical gradient transport vs ensemble");
    check(neardev < 1e-6, name + " mirror helix step vs trajectory");
    // far from z=0, where the LoopHelix t0 and phi0 are ill-conditioned near the turn (see the header).  The physics is the same, so
    // the physical transport, which doesn't use the parameters, should reproduce the z=0 result; the parameter transport is printed for information
    for(double mincos : {0.0, Config().xportmincos_}){
      PMAT farcov[4]; KTRAJ farseed; double farz;
      mirrorCovariance(cosz,zabove,-4000.0,mincos,farcov,farseed,farz);
      std::cout << "   z=-4000: parameter transport (guard " << mincos << "): no gradient" << print(farcov[0]) << ", gradient" << print(farcov[1]) << std::endl;
      if(mincos == 0.0){
        std::cout << "            physical transport:  no gradient" << print(farcov[2]) << ", gradient" << print(farcov[3]) << " (helix step check " << maxdev << " mm)" << std::endl;
        for(int iphys=2;iphys<4;iphys++)
          check(fabs(sqrt(farcov[iphys](0,0)/plcov[iphys](0,0))-1.0) < 0.01 && fabs(sqrt(farcov[iphys](1,1)/plcov[iphys](1,1))-1.0) < 0.01,
              name + " mirror physical transport independent of z");
      }
    }
    // local reference (no guard): extrapolated pieces have t0 and phi0 defined at their starting z instead of z=0.  Scan the distance between
    // the origin and the turn, comparing the plane widths, and the full state covariance at the turning piece, against the physical transport.
    // The covariance difference is max_ij |C_ij - P_ij|/sqrt(P_ii P_jj), with C the parameter and P the physical transport
    auto covdiff = [](DMAT const& c, DMAT const& p){
      double maxd(0.0);
      for(int i=0;i<6;i++)for(int j=0;j<6;j++){
        double d = fabs(c(i,j)-p(i,j))/sqrt(p(i,i)*p(j,j));
        if(!std::isfinite(d)) return std::numeric_limits<double>::infinity();
        maxd = std::max(maxd,d);
      }
      return maxd;
    };
    std::cout << "   reference scan (gradient transport; plane sigU/sigV; turning-piece lambda, local z - zref, state cov difference vs physical):" << std::endl;
    for(double zstart : {0.0, -1000.0, -2000.0, -4000.0, -8000.0}){
      for(bool localref : {false, true}){
        PMAT scov[4]; KTRAJ sseed; double sz;
        mirrorCovariance(cosz,zabove,zstart,0.0,scov,sseed,sz,localref);
        double dturn = covdiff(turncov[0],turncov[1]);
        char line[300];
        snprintf(line,300,"     zstart %6.0f %-6s param %9.4g/%-9.4g physical %7.4f/%-7.4f  turn lambda %8.3g zloc-zref %8.1f covdiff %8.2g",
            zstart, localref ? "local" : "origin", sqrt(scov[1](0,0)), sqrt(scov[1](1,1)), sqrt(scov[3](0,0)), sqrt(scov[3](1,1)),
            turnlam, turnz, dturn);
        if(dturn > 1e-3 && dturn < 1e3){
          std::cout << "       sqrt(diag) param   "; for(int i=0;i<6;i++) std::cout << sqrt(turncov[0](i,i)) << " "; std::cout << std::endl;
          std::cout << "       sqrt(diag) physical"; for(int i=0;i<6;i++) std::cout << sqrt(turncov[1](i,i)) << " "; std::cout << std::endl;
        }
        std::cout << line << std::endl;
        if(localref){
          check(fabs(sqrt(scov[1](0,0)/scov[3](0,0))-1.0) < 0.02 && fabs(sqrt(scov[1](1,1)/scov[3](1,1))-1.0) < 0.02,
              name + " mirror local-reference transport vs physical, zstart " + std::to_string(zstart));
          check(dturn < 0.02, name + " mirror local-reference turn covariance vs physical, zstart " + std::to_string(zstart));
        }
      }
    }
  }
  // conditioning of the LoopHelix representation itself near pz=0: build a piece from a state estimate (origin or local reference),
  // convert back to the state covariance at the construction time (round trip), and at the end of a domain-sized step, compared to the exact
  // helix Jacobian transport of the original state covariance.  Scanned in lambda (at the piece) and z
  void checkConditioning() {
    VEC3 bnom(0.0,0.0,1.5);
    DMAT scov;
    for(int i=0;i<3;i++){ scov(i,i) = 1.0e-2; scov(i+3,i+3) = 9.0e-4; }
    for(int i=0;i<3;i++){ scov(i,i+3) = scov(i+3,i) = 0.5*sqrt(scov(i,i)*scov(i+3,i+3)); } // some position-momentum correlation
    auto covdiff = [](DMAT const& c, DMAT const& p){
      double maxd(0.0);
      for(int i=0;i<6;i++)for(int j=0;j<6;j++){
        double d = fabs(c(i,j)-p(i,j))/sqrt(p(i,i)*p(j,j));
        if(!std::isfinite(d)) return std::numeric_limits<double>::infinity();
        maxd = std::max(maxd,d);
      }
      return maxd;
    };
    std::cout << "LoopHelix conditioning near pz=0 (state cov difference: round trip, and after a " << domaindt << " ns step vs the exact helix):" << std::endl;
    for(double z : {0.0, -4000.0}){
      for(double lam : {10.0, 1.0, 0.1, 0.01, 0.001}){
        double pt = 100.0, pz = lam*BFieldMap::cbar()*bnom.R(); // lambda = pz/Q, |Q| = cbar*B for unit charge
        ParticleState state(VEC3(-450.,300.,z),VEC3(pt*sin(0.3),-pt*cos(0.3),pz),0.0,0.511,-1);
        ParticleStateEstimate pse(state,scov);
        auto exact = ROOT::Math::Similarity(helixJacobian(state,bnom,domaindt),scov);
        char line[300];
        for(bool local : {false,true}){
          LoopHelix piece = local ? LoopHelix(pse,bnom,TimeRange(-1,1),LoopHelix::LocalReference()) : LoopHelix(pse,bnom,TimeRange(-1,1));
          double rt = covdiff(piece.stateEstimate(0.0).stateCovariance(),scov);
          double step = covdiff(piece.stateEstimate(domaindt).stateCovariance(),exact);
          snprintf(line,300,"   z %6.0f lambda %6.3g %-6s round trip %8.2g step %8.2g  (t0 %10.4g phi0 %10.4g)",z,fabs(piece.lam()),local?"local":"origin",rt,step,piece.t0(),piece.phi0());
          std::cout << line << std::endl;
          if(local) check(rt < 1e-4 && step < 1e-4, "LoopHelix local-reference conditioning, z " + std::to_string(z) + " lambda " + std::to_string(lam));
        }
      }
    }
  }
  // reference time and zTime: a local reference doesn't change LoopHelix::referenceTime or the trajectory t0, and zTime finds the z crossing
  template <class KTRAJ> void checkZTime(std::string const& name) {
    VEC3 bnom(0.05,-0.03,1.0); bnom *= 1.2; // tilted, so z is not linear in time
    VEC4 pos(-300., 200., -2500., 3.0);
    MOM4 mom(60.,-55.,40.,0.511);
    KTRAJ traj(pos,mom,-1,bnom,TimeRange(-20.0,20.0));
    double maxdz(0.0);
    for(double zpos : {-2700.0, -2500.0, -2000.0, 0.0}){
      double zt = traj.zTime(zpos,3.0);
      maxdz = std::max(maxdz,fabs(traj.position3(zt).Z()-zpos));
    }
    std::cout << name << " zTime max z deviation " << maxdz << " mm" << std::endl;
    check(maxdz < 1e-5, name + " zTime");
    check(fabs(traj.referenceTime()-traj.t0()) < 1e-12 || std::is_same_v<KTRAJ,LoopHelix>, name + " referenceTime is t0");
  }
  void checkReferenceTime() {
    // LoopHelix: same state, origin and local reference
    VEC3 bnom(0.05,-0.03,1.0); bnom *= 1.2;
    DMAT scov; for(int i=0;i<6;i++) scov(i,i) = i<3 ? 1e-2 : 1e-3;
    ParticleStateEstimate pse(ParticleState(VEC4(-300., 200., -2500., 3.0),MOM4(60.,-55.,40.,0.511),-1),scov);
    LoopHelix origin(pse,bnom,TimeRange(-20,20)), local(pse,bnom,TimeRange(-20,20),LoopHelix::LocalReference());
    std::cout << "LoopHelix reference time origin " << origin.referenceTime() << " local " << local.referenceTime() << " (local t0 " << local.t0() << ")" << std::endl;
    check(fabs(origin.referenceTime()-origin.t0()) < 1e-12, "LoopHelix origin referenceTime is t0");
    check(fabs(origin.referenceTime()-local.referenceTime()) < 1e-9, "LoopHelix local referenceTime");
    check(fabs(origin.zTime(-1000.0,3.0)-local.zTime(-1000.0,3.0)) < 1e-9, "LoopHelix local zTime");
    // extrapolated trajectory: t0 and zTime don't depend on the local reference
    double t0[2], zt[2];
    for(int ilocal=0;ilocal<2;ilocal++){
      GradientBFieldMap bfield(2.0,1.0,zplane,0.0);
      VEC4 spos(-450., 300., 0.0, tstart);
      VEC3 bstart = bfield.fieldVect(spos.Vect());
      LoopHelix seed(spos,MOM4(80.,-50.,50.,0.511),-1,bstart,TimeRange(tstart-domaindt,tstart+domaindt));
      ParticleTrajectory<LoopHelix> ptraj(seed);
      extrapolateMirror(ptraj,bfield,true,-1e6,0.0,ilocal==1);
      t0[ilocal] = ptraj.t0();
      // hint from the seed z velocity: the first (pre-turn) crossing
      double thint = tstart - 150.0/seed.velocity(tstart).Z();
      zt[ilocal] = ptraj.zTime(-150.0,thint);
      check(fabs(ptraj.position3(zt[ilocal]).Z()+150.0) < 1e-5, "LoopHelix piecewise zTime");
      // the front piece t0 is its construction time, inside its range: a search using the piece t0 parameters would stop there
      if(ilocal==1){
        std::cout << "LoopHelix extrapolated with local reference: front piece t0 " << ptraj.front().t0() << " range " << ptraj.front().range() << std::endl;
        check(ptraj.front().t0() >= ptraj.front().range().begin() && ptraj.front().t0() <= ptraj.front().range().end(), "LoopHelix local-reference front piece t0 is in range");
      }
    }
    std::cout << "LoopHelix extrapolated trajectory t0 origin " << t0[0] << " local " << t0[1] << ", zTime(-150) origin " << zt[0] << " local " << zt[1] << std::endl;
    check(fabs(t0[0]-t0[1]) < 1e-9, "LoopHelix extrapolated t0 independent of the local reference");
    check(fabs(zt[0]-zt[1]) < 1e-9, "LoopHelix extrapolated zTime independent of the local reference");
  }
  // the guard: no transport where |cos(direction,bnom)| is below mincos
  template <class KTRAJ> void checkGuard(std::string const& name) {
    GradientBFieldMap bfield(2.0,1.0,zplane,0.0);
    VEC4 pos(-450., 300., -300.0, tstart);
    VEC3 bfld = bfield.fieldVect(pos.Vect());
    MOM4 mom(80.,-60.,15.,0.511); // |cos(p,B)| ~ 0.15
    KTRAJ traj(pos,mom,-1,bfld,TimeRange(-1.0,1.0));
    double cosb = fabs(traj.direction(0.0).Dot(bfld.Unit()));
    PSMAT ident = ROOT::Math::SMatrixIdentity();
    Domain lowguard(TimeRange(-1.0,1.0),bfld,bfield.fieldGrad(pos.Vect()),0.5*cosb);
    Domain highguard(TimeRange(-1.0,1.0),bfld,bfield.fieldGrad(pos.Vect()),2.0*cosb);
    check(lowguard.paramTransport(traj,0.0,-0.5) != ident, name + " transport above the guard");
    check(highguard.paramTransport(traj,0.0,-0.5) == ident, name + " no transport below the guard");
    Domain copy(highguard);
    check(copy.minCos() == highguard.minCos(), name + " Domain copy keeps the guard");
  }
}

int main(int argc, char** argv){
  checkdPardBdB<LoopHelix>("LoopHelix");
  checkdPardBdB<CentralHelix>("CentralHelix");
  checkEnsemble<LoopHelix>("LoopHelix");
  checkEnsemble<CentralHelix>("CentralHelix");
  checkGuard<LoopHelix>("LoopHelix");
  checkGuard<CentralHelix>("CentralHelix");
  checkMirror<LoopHelix>("LoopHelix",0.5,10.0);
  checkMirror<LoopHelix>("LoopHelix",0.3,10.0);
  checkConditioning();
  checkZTime<LoopHelix>("LoopHelix");
  checkZTime<CentralHelix>("CentralHelix");
  checkZTime<KinematicLine>("KinematicLine");
  checkReferenceTime();
  if(nfail > 0){
    std::cout << nfail << " ExtrapolationCovariance tests failed" << std::endl;
    return -1;
  }
  std::cout << "ExtrapolationCovariance tests passed" << std::endl;
  return 0;
}
