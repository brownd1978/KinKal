#include "KinKal/General/ParticleStateEstimate.hh"
#include <stdexcept>
namespace KinKal {
  double ParticleStateEstimate::momentumVariance() const {
    auto momdir = momentum3().Unit();
    DVEC dMdm(0.0, 0.0, 0.0, momdir.X(), momdir.Y(), momdir.Z());
    return ROOT::Math::Similarity(dMdm,scovar_);
  }

  double ParticleStateEstimate::positionVariance(MomBasis::Direction dir) const {
    if(dir == MomBasis::momdir_)throw std::invalid_argument("Variance along momentum direction is undefined");
    auto momdir = momentum3();
    auto pdir = MomBasis::direction(dir,momdir);
    DVEC dPdp(pdir.X(),pdir.Y(),pdir.Z(),0.0, 0.0, 0.0);
    return ROOT::Math::Similarity(dPdp,scovar_);
  }

  PMAT ParticleStateEstimate::planeCovariance() const {
    auto momdir = momentum3();
    auto udir = MomBasis::direction(MomBasis::perpdir_,momdir);
    auto vdir = MomBasis::direction(MomBasis::phidir_,momdir);
    SVEC6 uvec(udir.X(),udir.Y(),udir.Z(),0.0, 0.0, 0.0);
    SVEC6 vvec(vdir.X(),vdir.Y(),vdir.Z(),0.0, 0.0, 0.0);
    PPMAT ppmat;
    ppmat.Place_in_row(uvec,0,0);
    ppmat.Place_in_row(vvec,1,0);
    return ROOT::Math::Similarity(ppmat,scovar_);
  }
}

namespace {
  using namespace KinKal;
  // helix functions of the phase Omega*dt: a = sin(Omega dt)/Omega, b = (1-cos(Omega dt))/Omega and their derivatives WRT Omega,
  // using series expansions for small phase (including Omega = 0, a straight line)
  struct HelixFns { double co, si, a, b, da, db; };
  HelixFns helixFns(double omega, double dt) {
    HelixFns h;
    double phi = omega*dt;
    h.co = cos(phi); h.si = sin(phi);
    if(fabs(phi) > 1.0e-3){
      h.a = h.si/omega; h.b = (1.0-h.co)/omega;
      h.da = (dt*h.co - h.a)/omega;
      h.db = (dt*h.si - h.b)/omega;
    } else {
      double t2 = dt*dt, p2 = phi*phi;
      h.a = dt*(1.0 - p2/6.0 + p2*p2/120.0);
      h.b = omega*t2*(0.5 - p2/24.0 + p2*p2/720.0);
      h.da = -omega*t2*dt*(1.0/3.0 - p2/30.0);
      h.db = t2*(0.5 - p2/8.0 + p2*p2/144.0);
    }
    return h;
  }
  // field direction (z for no field) and signed angular frequency: dp/dt = q v x B = Omega bhat x p
  void helixFrame(ParticleState const& ps, VEC3 const& bnom, VEC3& bhat, double& omega) {
    double bmag = bnom.R();
    bhat = bmag > 0.0 ? bnom/bmag : VEC3(0.0,0.0,1.0);
    omega = -(CLHEP::c_light/1000.0)*ps.charge()*bmag*CLHEP::c_light/ps.energy();
  }
  RMAT outer(VEC3 const& a, VEC3 const& b) {
    RMAT m;
    double av[3] = {a.X(),a.Y(),a.Z()}, bv[3] = {b.X(),b.Y(),b.Z()};
    for(int i=0;i<3;i++) for(int j=0;j<3;j++) m(i,j) = av[i]*bv[j];
    return m;
  }
}

namespace KinKal {
  PMAT ParticleStateEstimate::planeCovariance(VEC3 const& norm, VEC3 const& udir, VEC3 const& vdir, double dirtol) const {
    auto momdir = momentum3().Unit();
    double ndot = norm.Dot(momdir);
    if(fabs(ndot) < dirtol) throw std::invalid_argument("Momentum direction lies in the plane");
    // a perturbation dx moves the crossing point by dx - momdir*(norm.dx)/(norm.momdir)
    auto uproj = udir - norm*(udir.Dot(momdir)/ndot);
    auto vproj = vdir - norm*(vdir.Dot(momdir)/ndot);
    PPMAT ppmat;
    ppmat.Place_in_row(SVEC6(uproj.X(),uproj.Y(),uproj.Z(),0.0,0.0,0.0),0,0);
    ppmat.Place_in_row(SVEC6(vproj.X(),vproj.Y(),vproj.Z(),0.0,0.0,0.0),1,0);
    return ROOT::Math::Similarity(ppmat,scovar_);
  }

  // With P the momentum, b the field direction, P|| = b(b.P), Pperp = P - P||, Q = b x P and E the energy:
  // p(dt) = P|| + cos(Omega dt) Pperp + sin(Omega dt) Q
  // x(dt) = x + (c/E) F, F = P|| dt + a Pperp + b Q (see helixFns)
  ParticleState ParticleStateEstimate::transportState(VEC3 const& bnom, double dt) const {
    VEC3 bhat; double omega;
    helixFrame(*this,bnom,bhat,omega);
    auto h = helixFns(omega,dt);
    auto mom = momentum3();
    auto ppar = bhat*bhat.Dot(mom);
    auto pperp = mom - ppar;
    auto q = bhat.Cross(mom);
    auto newmom = ppar + h.co*pperp + h.si*q;
    auto fvec = ppar*dt + h.a*pperp + h.b*q;
    auto newpos = position3() + fvec*(CLHEP::c_light/energy());
    return ParticleState(newpos,newmom,time()+dt,mass(),charge());
  }

  PSMAT ParticleStateEstimate::transportJacobian(VEC3 const& bnom, double dt) const {
    VEC3 bhat; double omega;
    helixFrame(*this,bnom,bhat,omega);
    auto h = helixFns(omega,dt);
    auto mom = momentum3();
    double energy = this->energy(), cbyE = CLHEP::c_light/energy;
    auto ppar = bhat*bhat.Dot(mom);
    auto pperp = mom - ppar;
    auto q = bhat.Cross(mom);
    auto fvec = ppar*dt + h.a*pperp + h.b*q;
    // the frequency depends on the momentum through the energy: dOmega/dP = -Omega P/E^2
    auto domdp = mom*(-omega/(energy*energy));
    RMAT ident = ROOT::Math::SMatrixIdentity();
    RMAT bb = outer(bhat,bhat);
    RMAT iperp = ident - bb;
    RMAT bx; // bx v = b x v
    bx(0,1) = -bhat.Z(); bx(0,2) = bhat.Y();
    bx(1,0) = bhat.Z(); bx(1,2) = -bhat.X();
    bx(2,0) = -bhat.Y(); bx(2,1) = bhat.X();
    auto dpdom = (-h.si*pperp + h.co*q)*dt;
    RMAT dmdm = bb + h.co*iperp + h.si*bx + outer(dpdom,domdp);
    auto dfdom = h.da*pperp + h.db*q;
    RMAT dxdm = cbyE*(dt*bb + h.a*iperp + h.b*bx + outer(dfdom,domdp)) - outer(fvec,mom)*(cbyE/(energy*energy));
    PSMAT jac;
    jac.Place_at(ident,0,0);
    jac.Place_at(dxdm,0,3);
    jac.Place_at(dmdm,3,3);
    return jac;
  }

  PSMAT ParticleStateEstimate::transportJacobian(VEC3 const& bnom, double dt, SSMAT const& grad) const {
    auto jac = transportJacobian(bnom,dt);
    // d(state at time+dt)/dB at fixed initial state, by central differences of the exact transport
    static constexpr double db = 1.0e-5; // Tesla
    ROOT::Math::SMatrix<double,6,3> dsdb;
    for(int k=0;k<3;k++){
      VEC3 dbvec(k==0?db:0.0, k==1?db:0.0, k==2?db:0.0);
      SVEC6 dstate = (transportState(bnom+dbvec,dt).state() - transportState(bnom-dbvec,dt).state())/(2.0*db); // evaluate now: the expression references temporaries
      dsdb.Place_in_col(dstate,0,k);
    }
    // displacement half way through the step
    auto jmid = transportJacobian(bnom,0.5*dt);
    auto dxmid = jmid.Sub<ROOT::Math::SMatrix<double,3,6>>(0,0);
    jac += dsdb*grad*dxmid;
    return jac;
  }

  ParticleStateEstimate ParticleStateEstimate::transport(VEC3 const& bnom, double dt) const {
    return ParticleStateEstimate(transportState(bnom,dt),ROOT::Math::Similarity(transportJacobian(bnom,dt),scovar_));
  }

  ParticleStateEstimate ParticleStateEstimate::transport(VEC3 const& bnom, double dt, SSMAT const& grad) const {
    return ParticleStateEstimate(transportState(bnom,dt),ROOT::Math::Similarity(transportJacobian(bnom,dt,grad),scovar_));
  }
}
