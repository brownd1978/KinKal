#ifndef KinKal_BField_hh
#define KinKal_BField_hh
// class defining a BField Map interface for use in KinKal.
#include "KinKal/Vectors.hh"
#include "KinKal/TRange.hh"
#include "CLHEP/Units/PhysicalConstants.h"
#include "Math/SMatrix.h"
#include <vector>
#include <algorithm>
#include <cstdarg>

namespace KinKal {
  class BField {
    public:
      typedef ROOT::Math::SMatrix<float,3> Grad; // field gradient: ie dBi/d(x,y,z)
      // return value of the field at a point
      virtual Vec3 fieldVect(Vec3 const& position) const = 0; 
      // return BField gradient = dB_i/dx_j, at a given point
      virtual Grad fieldGrad(Vec3 const& position) const = 0;
      // return the BField derivative at a given point along a given velocity, WRT time
      virtual Vec3 fieldDeriv(Vec3 const& position, Vec3 const& velocity) const = 0;
      // integrate the magentic force over the given trajectory and range, due to the DIFFERENCE
      // between this field and the field vector referenced by the traj.  Returns the change in momentum
      // ( = - integral of the 'external' force needed to keep the particle onto this trajectory);
      template <class KTRAJ> void integrate(KTRAJ const& ktraj, TRange const& range, Vec3& dmom) const;
      // speed of light in units to convert Tesla to mm (bending radius)
      static float cbar() { static float retval = CLHEP::c_light/1000.0; return retval; }

      virtual ~BField(){}
  };

  // trivial instance of the above, used for testing
  class UniformBField : public BField {
    public:
      virtual Vec3 fieldVect(Vec3 const& position) const override { return fvec_; }
      virtual Grad fieldGrad(Vec3 const& position) const override { return Grad(); }
      virtual Vec3 fieldDeriv(Vec3 const& position, Vec3 const& velocity) const override { return Vec3(); }
      UniformBField(Vec3 const& bnom) : fvec_(bnom) {}
      UniformBField(float BZ) : UniformBField(Vec3(0.0,0.0,BZ)) {}
      virtual ~UniformBField(){}
    private:
      Vec3 fvec_; // constant field
  };

// use superposition to create a composite field
  class CompositeBField : public BField {
    public:
      virtual Vec3 fieldVect(Vec3 const& position) const override;
      virtual Grad fieldGrad(Vec3 const& position) const override;
      virtual Vec3 fieldDeriv(Vec3 const& position, Vec3 const& velocity) const override;
      CompositeBField () {}
      CompositeBField(int fcount, ...);
      void addField(BField const& field) { fields_.push_back(&field); }
      virtual ~CompositeBField() {}
    private:
      std::vector<const BField*> fields_; // fields
  };

  // simple Z gradient field, used to test Field corrections
  class GradBField : public BField {
    public:
      GradBField(float b0, float b1, float zg0, float zg1);
      virtual Vec3 fieldVect(Vec3 const& position) const override;
      virtual Grad fieldGrad(Vec3 const& position) const override { return fgrad_; }
      virtual Vec3 fieldDeriv(Vec3 const& position, Vec3 const& velocity) const override;
      virtual ~GradBField(){}
    private:
      float b0_, b1_;
      float z0_; 
      float grad_; // gradient in tesla/mm, computed from the fvec values
      Grad fgrad_;
  };

  template<class KTRAJ> void BField::integrate(KTRAJ const& ktraj, TRange const& trange, Vec3& dmom) const {
    dmom = Vec3();
    // compare this field with the noiminal at a few points. only bother to integrate if this is outside tolerance.
    std::vector<float> tvals = {trange.low(),trange.mid(),trange.high()};
    std::vector<float> db;
    for(auto tval :tvals){
      Vec3 pos; ktraj.position(tval,pos);
      Vec3 bf = fieldVect(pos);
      db.push_back( (bf - ktraj.bnom(tval)).R());
    }
    if(*std::max_element(db.begin(),db.end()) > 1e-6){ // tolerance should be a parameter FIXME!
      unsigned nsteps(10); // this should be computed from field rate-of-change over this range FIXME!
      float dt = trange.range()/float(nsteps);
      for(unsigned istep=0; istep< nsteps; istep++){
	float tstep = trange.low() + istep*dt;
	Vec3 tpos, vel;
	ktraj.position(tstep,tpos);
	ktraj.velocity(tstep,vel);
	Vec3 bvec = fieldVect(tpos);
	Vec3 db = bvec - ktraj.bnom(tstep);
	dmom += cbar()*ktraj.charge()*dt*vel.Cross(db);
      }
    }
  }

}
#endif
