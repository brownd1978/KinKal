//
// Test Intersection::inbfrac_, the fraction of the projected spatial covariance ellipse inside the surface boundaries.
// The fraction is compared with a toy MC in which the trajectory parameters are randomized according to their covariance,
// the intersection is recomputed, and the fraction of in-bounds intersections is counted.  This tests the normal projection
// of curved surface boundaries as well as the integration.  Simple configurations are also compared with analytic results.
// original author: David Brown (LBNL) 2026
//
#include "KinKal/Geometry/Intersection.hh"
#include "KinKal/Geometry/Intersect.hh"
#include "KinKal/Geometry/Plane.hh"
#include "KinKal/Geometry/Rectangle.hh"
#include "KinKal/Geometry/Annulus.hh"
#include "KinKal/Geometry/Disk.hh"
#include "KinKal/Geometry/Cylinder.hh"
#include "KinKal/Geometry/Frustrum.hh"
#include <iostream>
#include <string>
#include <random>
#include <cmath>
#include <getopt.h>

using namespace KinKal;
using std::cout;
using std::endl;

static struct option long_options[] = {
  {"ntoy",       required_argument, 0, 'n'  },
  {"seed",       required_argument, 0, 's'  },
  {"verbose",    required_argument, 0, 'v'  },
  {NULL, 0,0,0}
};

void print_usage() {
  printf("Usage: IntersectionBounds --ntoy (number of toy MC trials) --seed (random seed) --verbose (verbosity)\n");
}

template <class KTRAJ> void setCovariance(KTRAJ& ktraj, std::array<double,6> const& sigmas) {
  for(size_t ipar=0;ipar < NParams(); ++ipar)
    for(size_t jpar=0;jpar < NParams(); ++jpar)
      ktraj.params().covariance()(ipar,jpar) = ipar == jpar ? sigmas[ipar]*sigmas[ipar] : 0.0;
}

double normcdf(double x) { return 0.5*erfc(-x/sqrt(2.0)); }

static const double tol(1.0e-3); // intersection tolerance (mm): small compared to the projected errors (~1 mm), and keeps the toy MC fast
static const double fractol(0.01); // the target precision of inbfrac_
static const double toytol(0.015); // tolerance on the toy MC comparison: target precision plus statistics

// compare the bounds fraction with the toy MC fraction of in-bounds intersections, and optionally an analytic expectation (if >= 0).
template <class KTRAJ, class SURF> int testBounds(std::string const& name, KTRAJ const& ktraj, SURF const& surf,
    TimeRange const& trange, double expect, unsigned ntoy, unsigned seed, int verbose) {
  auto inter = intersect(ktraj,surf,trange,tol,TimeDir::forwards,true);
  if(!inter.onsurface_ || !inter.hascov_){
    cout << name << ": no intersection covariance " << inter << endl;
    return -1;
  }
  if(inter.inbfrac_ < 0.0 || inter.inbfrac_ > 1.0){
    cout << name << ": fraction out of range " << inter.inbfrac_ << endl;
    return -1;
  }
  // an intersection constructed outside of intersect (only position, time and flags set) must give the same projection
  Intersection hinter;
  hinter.pos_ = inter.pos_;
  hinter.time_ = inter.time_;
  hinter.onsurface_ = inter.onsurface_;
  hinter.inbounds_ = inter.inbounds_;
  fillPlaneCovariance(ktraj,surf,hinter,tol);
  if(!hinter.hascov_ || fabs(hinter.inbfrac_-inter.inbfrac_) > 1.0e-9 || (hinter.norm_-inter.norm_).R() > 1.0e-9
      || (hinter.udir_-inter.udir_).R() > 1.0e-9 || (hinter.vdir_-inter.vdir_).R() > 1.0e-9){
    cout << name << ": hand-built intersection inconsistent " << hinter << endl;
    return -1;
  }
  for(size_t i=0;i<2;++i)for(size_t j=0;j<2;++j)
    if(fabs(hinter.pcov_(i,j)-inter.pcov_(i,j)) > 1.0e-9*(inter.pcov_(0,0)+inter.pcov_(1,1))){
      cout << name << ": hand-built intersection covariance inconsistent" << endl;
      return -1;
    }
  // toy MC; restrict the intersection search to a window around the nominal intersection
  static const double dt(0.5); // ns
  TimeRange toyrange(inter.time_-dt,inter.time_+dt);
  std::mt19937 gen(seed);
  std::normal_distribution<double> gauss(0.0,1.0);
  unsigned nsurf(0), ninb(0);
  for(unsigned itoy=0; itoy < ntoy; ++itoy){
    auto pars = ktraj.params();
    for(size_t ipar=0;ipar < NParams(); ++ipar)
      pars.parameters()[ipar] += gauss(gen)*sqrt(ktraj.params().covariance()(ipar,ipar));
    KTRAJ toytraj(pars,ktraj);
    auto toyinter = intersect(toytraj,surf,toyrange,tol);
    if(!toyinter.onsurface_) continue;
    ++nsurf;
    if(toyinter.inbounds_) ++ninb;
  }
  double toyfrac = nsurf > 0 ? double(ninb)/nsurf : -1.0;
  int retval(0);
  if(nsurf < 0.99*ntoy){
    cout << name << ": too many toy intersections failed " << nsurf << " of " << ntoy << endl;
    retval = -1;
  }
  if(fabs(toyfrac-inter.inbfrac_) > toytol){
    cout << name << ": fraction " << inter.inbfrac_ << " disagrees with toy MC " << toyfrac << endl;
    retval = -1;
  }
  if(expect >= 0.0 && fabs(expect-inter.inbfrac_) > fractol){
    cout << name << ": fraction " << inter.inbfrac_ << " disagrees with analytic " << expect << endl;
    retval = -1;
  }
  if(verbose > 0 || retval != 0)
    cout << name << " inbounds " << inter.inbounds_ << " fraction " << inter.inbfrac_ << " toy " << toyfrac
      << (expect >= 0.0 ? " analytic " + std::to_string(expect) : std::string()) << (retval != 0 ? " FAILED" : "") << endl;
  return retval;
}

template <class KTRAJ> int testTraj(std::string const& tname, KTRAJ const& ktraj, TimeRange const& trange, unsigned ntoy, unsigned seed, int verbose) {
  int status(0);
  // nominal intersection with an (unbounded) tilted plane defines the local frame and covariance
  VEC3 norm = VEC3(0.3,0.2,1.0).Unit();
  VEC3 pudir = norm.Cross(VEC3(0.0,1.0,0.0)).Unit();
  Plane plane(norm,pudir,VEC3(5.0,-3.0,400.0));
  auto inter = intersect(ktraj,plane,trange,tol,TimeDir::forwards,true);
  if(!inter.hascov_){
    cout << tname << ": no plane intersection" << endl;
    return -1;
  }
  status |= testBounds(tname+" Plane",ktraj,plane,trange,1.0,ntoy,seed,verbose);
  auto const& pos = inter.pos_;
  auto const& udir = inter.udir_;
  auto const& vdir = inter.vdir_;
  double su = sqrt(inter.pcov_(0,0));
  double sv = sqrt(inter.pcov_(1,1));
  double rho = inter.pcov_(0,1)/(su*sv);
  if(verbose > 0) cout << tname << " sigma u " << su << " v " << sv << " rho " << rho << endl;
  static const double big(1.0e4);
  // single edge: the +u edge of a rectangle at distance d from the intersection
  for(double dsig : {-1.5, -0.5, 0.0, 0.7, 2.0}){
    double d = dsig*su;
    Rectangle rect(norm,udir,pos+(d-big)*udir,big,big);
    status |= testBounds(tname+" Rectangle edge at "+std::to_string(dsig)+" sigma",ktraj,rect,trange,normcdf(dsig),ntoy,seed,verbose);
  }
  // corner: the intersection is at the corner, (u,v) edges at 0.  For a bivariate Gaussian the quadrant probability is 1/4 + asin(rho)/2pi
  {
    Rectangle rect(norm,udir,pos-big*udir-big*vdir,big,big);
    status |= testBounds(tname+" Rectangle corner",ktraj,rect,trange,0.25+asin(rho)/(2*M_PI),ntoy,seed,verbose);
  }
  // corner offset from the intersection, and a narrow strip (2 parallel edges)
  {
    Rectangle rect(norm,udir,pos+(0.5*su-big)*udir+(-0.8*sv+big)*vdir,big,big);
    status |= testBounds(tname+" Rectangle offset corner",ktraj,rect,trange,-1.0,ntoy,seed,verbose);
    Rectangle strip(norm,udir,pos+0.3*su*udir,0.6*su,big);
    status |= testBounds(tname+" Rectangle strip",ktraj,strip,trange,normcdf(0.9)-normcdf(-0.3),ntoy,seed,verbose);
    Rectangle small(norm,udir,pos+0.2*su*udir-0.3*sv*vdir,0.8*su,0.6*sv);
    status |= testBounds(tname+" Rectangle small (4 edges)",ktraj,small,trange,-1.0,ntoy,seed,verbose);
  }
  // intersection outside the boundary
  {
    Rectangle rect(norm,udir,pos+(-1.0*su-big)*udir,big,big);
    status |= testBounds(tname+" Rectangle out of bounds",ktraj,rect,trange,normcdf(-1.0),ntoy,seed,verbose);
  }
  // disk edge, and an annulus with the intersection in the hole
  {
    double smax = std::max(su,sv);
    Disk disk(norm,udir,pos+3.0*smax*udir,3.0*smax+0.5*su);
    status |= testBounds(tname+" Disk",ktraj,disk,trange,-1.0,ntoy,seed,verbose);
    Annulus ann(norm,udir,pos+0.5*smax*vdir,1.0*smax,10.0*smax);
    status |= testBounds(tname+" Annulus (in hole)",ktraj,ann,trange,-1.0,ntoy,seed,verbose);
  }
  return status;
}

// curved surfaces: the trajectory crosses a cylinder or frustrum near its end
template <class KTRAJ> int testCurved(std::string const& tname, KTRAJ const& ktraj, TimeRange const& trange, unsigned ntoy, unsigned seed, int verbose) {
  int status(0);
  VEC3 zdir(0.0,0.0,1.0);
  double rcyl(200.0), hlen(1000.0);
  Cylinder cyl0(zdir,VEC3(0.0,0.0,0.0),rcyl,hlen);
  auto inter = intersect(ktraj,cyl0,trange,tol,TimeDir::forwards,true);
  if(!inter.hascov_){
    cout << tname << ": no cylinder intersection" << endl;
    return -1;
  }
  // axial sigma of the projected covariance; the axial direction lies in the tangent plane
  double sz = sqrt(ROOT::Math::Similarity(inter.pcov_,ROOT::Math::SVector<double,2>(zdir.Dot(inter.udir_),zdir.Dot(inter.vdir_))));
  if(verbose > 0) cout << tname << " cylinder axial sigma " << sz << endl;
  // restrict to a window around this crossing, so that the shifted surfaces below don't select a crossing on a different loop
  TimeRange crange(inter.time_-0.5,inter.time_+0.5);
  for(double dsig : {-1.0, 0.0, 0.5, 1.5}){
    // place the cylinder so its +z end is dsig sigma from the intersection
    Cylinder cyl(zdir,VEC3(0.0,0.0,inter.pos_.Z()+dsig*sz-hlen),rcyl,hlen);
    status |= testBounds(tname+" Cylinder end at "+std::to_string(dsig)+" sigma",ktraj,cyl,crange,normcdf(dsig),ntoy,seed,verbose);
  }
  // frustrum through the same point, with its +z end near the intersection.  The boundary projection is not exactly axial here
  double dr(100.0);
  for(double dsig : {-0.5, 0.0, 1.0}){
    double zend = inter.pos_.Z()+dsig*sz;
    // radius rcyl at the intersection z: r(z) = rlow + (rhigh-rlow)*(z-zlow)/(2 hlen)
    double rhigh = rcyl + dr*(zend - inter.pos_.Z())/(2*hlen);
    Frustrum fru(zdir,VEC3(0.0,0.0,zend-hlen),rhigh-dr,rhigh,hlen);
    status |= testBounds(tname+" Frustrum end at "+std::to_string(dsig)+" sigma",ktraj,fru,crange,-1.0,ntoy,seed,verbose);
  }
  return status;
}

int main(int argc, char** argv) {
  unsigned ntoy(20000), seed(124223);
  int verbose(0);
  int opt;
  int long_index =0;
  while ((opt = getopt_long_only(argc, argv,"",
          long_options, &long_index )) != -1) {
    switch (opt) {
      case 'n' : ntoy = atoi(optarg);
                 break;
      case 's' : seed = atoi(optarg);
                 break;
      case 'v' : verbose = atoi(optarg);
                 break;
      default: print_usage();
               exit(EXIT_FAILURE);
    }
  }
  VEC3 bnom(0.0,0.0,1.0);
  VEC4 pos(0.0,0.0,0.0,0.0);
  MOM4 mom(40.0,30.0,60.0,0.511);
  TimeRange trange(-10.0,100.0);
  LoopHelix lhelix(pos,mom,-1,bnom,trange);
  setCovariance(lhelix,{0.5,0.5,0.5,0.5,0.002,0.05});
  CentralHelix chelix(pos,mom,-1,bnom,trange);
  setCovariance(chelix,{0.3,0.002,1.0e-5,0.3,0.001,0.05});
  KinematicLine kline(pos,mom,-1,bnom,trange);
  setCovariance(kline,{0.1,0.002,0.1,0.002,0.005,0.05}); // d0, phi0, z0, theta, t0, mom
  int status(0);
  status |= testTraj("LoopHelix",lhelix,trange,ntoy,seed,verbose);
  status |= testTraj("CentralHelix",chelix,trange,ntoy,seed,verbose);
  status |= testTraj("KinematicLine",kline,trange,ntoy,seed,verbose);
  status |= testCurved("LoopHelix",lhelix,trange,ntoy,seed,verbose);
  status |= testCurved("CentralHelix",chelix,trange,ntoy,seed,verbose);
  status |= testCurved("KinematicLine",kline,trange,ntoy,seed,verbose);
  if(status == 0)
    cout << "IntersectionBounds test passed" << endl;
  else
    cout << "IntersectionBounds test FAILED" << endl;
  return status;
}
