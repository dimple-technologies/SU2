/*!
 * \file CNSSolver.cpp
 * \brief Main subrotuines for solving Finite-Volume Navier-Stokes flow problems.
 * \author F. Palacios, T. Economon
 * \version 7.0.6 "Blackbird"
 *
 * SU2 Project Website: https://su2code.github.io
 *
 * The SU2 Project is maintained by the SU2 Foundation
 * (http://su2foundation.org)
 *
 * Copyright 2012-2020, SU2 Contributors (cf. AUTHORS.md)
 *
 * SU2 is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * SU2 is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with SU2. If not, see <http://www.gnu.org/licenses/>.
 */

#include "../../include/solvers/CNSSolver.hpp"
#include "../../include/variables/CNSVariable.hpp"
#include "../../../Common/include/toolboxes/printing_toolbox.hpp"
#include "../../../Common/include/toolboxes/geometry_toolbox.hpp"


CNSSolver::CNSSolver(void) : CEulerSolver() { }

CNSSolver::CNSSolver(CGeometry *geometry, CConfig *config, unsigned short iMesh) :
           CEulerSolver(geometry, config, iMesh, true) {

  /*--- This constructor only allocates/inits what is extra to CEulerSolver. ---*/

  unsigned short iMarker, iDim;
  unsigned long iVertex;

  /*--- Store the values of the temperature and the heat flux density at the boundaries,
   used for coupling with a solid donor cell ---*/
  unsigned short nHeatConjugateVar = 4;

  HeatConjugateVar = new su2double** [nMarker];
  for (iMarker = 0; iMarker < nMarker; iMarker++) {
    HeatConjugateVar[iMarker] = new su2double* [nVertex[iMarker]];
    for (iVertex = 0; iVertex < nVertex[iMarker]; iVertex++) {
      HeatConjugateVar[iMarker][iVertex] = new su2double [nHeatConjugateVar]();
      HeatConjugateVar[iMarker][iVertex][0] = config->GetTemperature_FreeStreamND();
    }
  }

  /*--- Allocates a 2D array with variable "outer" sizes and init to 0. ---*/

  auto Alloc2D = [](unsigned long M, const unsigned long* N, su2double**& X) {
    X = new su2double* [M];
    for(unsigned long i = 0; i < M; ++i)
      X[i] = new su2double [N[i]] ();
  };

  /*--- Heat flux in all the markers ---*/

  Alloc2D(nMarker, nVertex, HeatFlux);
  Alloc2D(nMarker, nVertex, HeatFluxTarget);

  /*--- Y plus in all the markers ---*/

  Alloc2D(nMarker, nVertex, YPlus);

  /*--- Skin friction in all the markers ---*/

  CSkinFriction = new su2double** [nMarker];
  for (iMarker = 0; iMarker < nMarker; iMarker++) {
    CSkinFriction[iMarker] = new su2double*[nDim];
    for (iDim = 0; iDim < nDim; iDim++) {
      CSkinFriction[iMarker][iDim] = new su2double[nVertex[iMarker]] ();
    }
  }

  /*--- Non dimensional aerodynamic coefficients ---*/

  ViscCoeff.allocate(nMarker);
  SurfaceViscCoeff.allocate(config->GetnMarker_Monitoring());

  /*--- Heat flux and buffet coefficients ---*/

  HF_Visc = new su2double[nMarker];
  MaxHF_Visc = new su2double[nMarker];

  Surface_HF_Visc = new su2double[config->GetnMarker_Monitoring()];
  Surface_MaxHF_Visc = new su2double[config->GetnMarker_Monitoring()];

  /*--- Buffet sensor in all the markers and coefficients ---*/

  if(config->GetBuffet_Monitoring() || config->GetKind_ObjFunc() == BUFFET_SENSOR){

    Alloc2D(nMarker, nVertex, Buffet_Sensor);
    Buffet_Metric = new su2double[nMarker];
    Surface_Buffet_Metric = new su2double[config->GetnMarker_Monitoring()];

  }

  unsigned long nPointLocal, nPointGlobal;
	nPointLocal = nPointDomain;
#ifdef HAVE_MPI
SU2_MPI::Allreduce(&nPointLocal, &nPointGlobal, 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);
#else
nPointGlobal = nPointLocal; //Total number of points in the domain (no halo nodes considered)
#endif

  if (config->Get_BoolStokesDrag() == true){
	  u_tau = new su2double[nPointGlobal];
  }

  /*--- Read farfield conditions from config ---*/

  Viscosity_Inf   = config->GetViscosity_FreeStreamND();
  Prandtl_Lam     = config->GetPrandtl_Lam();
  Prandtl_Turb    = config->GetPrandtl_Turb();
  Tke_Inf         = config->GetTke_FreeStreamND();

  /*--- Initialize the seed values for forward mode differentiation. ---*/

  switch(config->GetDirectDiff()) {
    case D_VISCOSITY:
      SU2_TYPE::SetDerivative(Viscosity_Inf, 1.0);
      break;
    default:
      /*--- Already done upstream. ---*/
      break;
  }

}

CNSSolver::~CNSSolver(void) {

  unsigned short iMarker, iDim;

  unsigned long iVertex;

  delete [] Buffet_Metric;
  delete [] HF_Visc;
  delete [] MaxHF_Visc;

  delete [] Surface_HF_Visc;
  delete [] Surface_MaxHF_Visc;
  delete [] Surface_Buffet_Metric;

  if (CSkinFriction != nullptr) {
    for (iMarker = 0; iMarker < nMarker; iMarker++) {
      for (iDim = 0; iDim < nDim; iDim++) {
        delete [] CSkinFriction[iMarker][iDim];
      }
      delete [] CSkinFriction[iMarker];
    }
    delete [] CSkinFriction;
  }

  if (HeatConjugateVar != nullptr) {
    for (iMarker = 0; iMarker < nMarker; iMarker++) {
      for (iVertex = 0; iVertex < nVertex[iMarker]; iVertex++) {
        delete [] HeatConjugateVar[iMarker][iVertex];
      }
      delete [] HeatConjugateVar[iMarker];
    }
    delete [] HeatConjugateVar;
  }

  if (Buffet_Sensor != nullptr) {
    for (iMarker = 0; iMarker < nMarker; iMarker++){
      delete [] Buffet_Sensor[iMarker];
    }
    delete [] Buffet_Sensor;
  }

}

void CNSSolver::Preprocessing(CGeometry *geometry, CSolver **solver_container, CConfig *config, unsigned short iMesh,
                              unsigned short iRKStep, unsigned short RunTime_EqSystem, bool Output) {

  unsigned long InnerIter   = config->GetInnerIter();
  bool cont_adjoint         = config->GetContinuous_Adjoint();
  bool limiter_flow         = (config->GetKind_SlopeLimit_Flow() != NO_LIMITER) && (InnerIter <= config->GetLimiterIter());
  bool limiter_turb         = (config->GetKind_SlopeLimit_Turb() != NO_LIMITER) && (InnerIter <= config->GetLimiterIter());
  bool limiter_adjflow      = (cont_adjoint && (config->GetKind_SlopeLimit_AdjFlow() != NO_LIMITER) && (InnerIter <= config->GetLimiterIter()));
  bool van_albada           = config->GetKind_SlopeLimit_Flow() == VAN_ALBADA_EDGE;
  bool wall_functions       = config->GetWall_Functions();

  /*--- Common preprocessing steps (implemented by CEulerSolver) ---*/

  CommonPreprocessing(geometry, solver_container, config, iMesh, iRKStep, RunTime_EqSystem, Output);

  /*--- Compute gradient for MUSCL reconstruction. ---*/

  if (config->GetReconstructionGradientRequired() && (iMesh == MESH_0)) {
    switch (config->GetKind_Gradient_Method_Recon()) {
      case GREEN_GAUSS:
        SetPrimitive_Gradient_GG(geometry, config, true); break;
      case LEAST_SQUARES:
      case WEIGHTED_LEAST_SQUARES:
        SetPrimitive_Gradient_LS(geometry, config, true); break;
      default: break;
    }
  }

  /*--- Compute gradient of the primitive variables ---*/

  if (config->GetKind_Gradient_Method() == GREEN_GAUSS) {
    SetPrimitive_Gradient_GG(geometry, config);
  }
  else if (config->GetKind_Gradient_Method() == WEIGHTED_LEAST_SQUARES) {
    SetPrimitive_Gradient_LS(geometry, config);
  }

  /*--- Compute the limiter in case we need it in the turbulence model or to limit the
   *    viscous terms (check this logic with JST and 2nd order turbulence model) ---*/

  if ((iMesh == MESH_0) && (limiter_flow || limiter_turb || limiter_adjflow) && !Output && !van_albada) {
    SetPrimitive_Limiter(geometry, config);
  }

  /*--- Evaluate the vorticity and strain rate magnitude ---*/

  SU2_OMP_MASTER
  {
    StrainMag_Max = 0.0;
    Omega_Max = 0.0;
  }
  SU2_OMP_BARRIER

  nodes->SetVorticity_StrainMag();

  su2double strainMax = 0.0, omegaMax = 0.0;

  SU2_OMP(for schedule(static,omp_chunk_size) nowait)
  for (unsigned long iPoint = 0; iPoint < nPoint; iPoint++) {

    su2double StrainMag = nodes->GetStrainMag(iPoint);
    const su2double* Vorticity = nodes->GetVorticity(iPoint);
    su2double Omega = sqrt(Vorticity[0]*Vorticity[0]+ Vorticity[1]*Vorticity[1]+ Vorticity[2]*Vorticity[2]);

    strainMax = max(strainMax, StrainMag);
    omegaMax = max(omegaMax, Omega);

  }
  SU2_OMP_CRITICAL
  {
    StrainMag_Max = max(StrainMag_Max, strainMax);
    Omega_Max = max(Omega_Max, omegaMax);
  }

  if ((iMesh == MESH_0) && (config->GetComm_Level() == COMM_FULL)) {
    SU2_OMP_BARRIER
    SU2_OMP_MASTER
    {
      su2double MyOmega_Max = Omega_Max;
      su2double MyStrainMag_Max = StrainMag_Max;

      SU2_MPI::Allreduce(&MyStrainMag_Max, &StrainMag_Max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      SU2_MPI::Allreduce(&MyOmega_Max, &Omega_Max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    }
    SU2_OMP_BARRIER
  }

  /*--- Compute the TauWall from the wall functions ---*/

  if (wall_functions) {
    SetTauWall_WF(geometry, solver_container, config);
  }

}

unsigned long CNSSolver::SetPrimitive_Variables(CSolver **solver_container, CConfig *config, bool Output) {

  /*--- Number of non-physical points, local to the thread, needs
   *    further reduction if function is called in parallel ---*/
  unsigned long nonPhysicalPoints = 0;

  const unsigned short turb_model = config->GetKind_Turb_Model();
  const bool tkeNeeded = (turb_model == SST) || (turb_model == SST_SUST);

  SU2_OMP_FOR_STAT(omp_chunk_size)
  for (unsigned long iPoint = 0; iPoint < nPoint; iPoint ++) {

    /*--- Retrieve the value of the kinetic energy (if needed). ---*/

    su2double eddy_visc = 0.0, turb_ke = 0.0;

    if (turb_model != NONE && solver_container[TURB_SOL] != nullptr) {
      eddy_visc = solver_container[TURB_SOL]->GetNodes()->GetmuT(iPoint);
      if (tkeNeeded) turb_ke = solver_container[TURB_SOL]->GetNodes()->GetSolution(iPoint,0);

      if (config->GetKind_HybridRANSLES() != NO_HYBRIDRANSLES) {
        su2double DES_LengthScale = solver_container[TURB_SOL]->GetNodes()->GetDES_LengthScale(iPoint);
        nodes->SetDES_LengthScale(iPoint, DES_LengthScale);
      }
    }

    /*--- Compressible flow, primitive variables nDim+5, (T, vx, vy, vz, P, rho, h, c, lamMu, eddyMu, ThCond, Cp) ---*/

    bool physical = static_cast<CNSVariable*>(nodes)->SetPrimVar(iPoint, eddy_visc, turb_ke, GetFluidModel());
    nodes->SetSecondaryVar(iPoint, GetFluidModel());

    /*--- Check for non-realizable states for reporting. ---*/

    nonPhysicalPoints += !physical;

  }

  return nonPhysicalPoints;
}

void CNSSolver::Viscous_Residual(unsigned long iEdge, CGeometry *geometry, CSolver **solver_container,
                                 CNumerics *numerics, CConfig *config) {

  const bool implicit  = (config->GetKind_TimeIntScheme() == EULER_IMPLICIT);
  const bool tkeNeeded = (config->GetKind_Turb_Model() == SST) ||
                         (config->GetKind_Turb_Model() == SST_SUST);

  CVariable* turbNodes = nullptr;
  if (tkeNeeded) turbNodes = solver_container[TURB_SOL]->GetNodes();

  /*--- Points, coordinates and normal vector in edge ---*/

  auto iPoint = geometry->edges->GetNode(iEdge,0);
  auto jPoint = geometry->edges->GetNode(iEdge,1);

  numerics->SetCoord(geometry->nodes->GetCoord(iPoint),
                     geometry->nodes->GetCoord(jPoint));

  numerics->SetNormal(geometry->edges->GetNormal(iEdge));

  /*--- Primitive and secondary variables. ---*/

  numerics->SetPrimitive(nodes->GetPrimitive(iPoint),
                         nodes->GetPrimitive(jPoint));

  numerics->SetSecondary(nodes->GetSecondary(iPoint),
                         nodes->GetSecondary(jPoint));

  /*--- Gradients. ---*/

  numerics->SetPrimVarGradient(nodes->GetGradient_Primitive(iPoint),
                               nodes->GetGradient_Primitive(jPoint));

  /*--- Turbulent kinetic energy. ---*/

  if (tkeNeeded)
    numerics->SetTurbKineticEnergy(turbNodes->GetSolution(iPoint,0),
                                   turbNodes->GetSolution(jPoint,0));

  /*--- Wall shear stress values (wall functions) ---*/

  numerics->SetTauWall(nodes->GetTauWall(iPoint),
                       nodes->GetTauWall(iPoint));

  /*--- Compute and update residual ---*/

  auto residual = numerics->ComputeResidual(config);

  if (ReducerStrategy) {
    EdgeFluxes.SubtractBlock(iEdge, residual);
    if (implicit)
      Jacobian.UpdateBlocksSub(iEdge, residual.jacobian_i, residual.jacobian_j);
  }
  else {
    LinSysRes.SubtractBlock(iPoint, residual);
    LinSysRes.AddBlock(jPoint, residual);

    if (implicit)
      Jacobian.UpdateBlocksSub(iEdge, iPoint, jPoint, residual.jacobian_i, residual.jacobian_j);
  }

}

void CNSSolver::Friction_Forces(CGeometry *geometry, CConfig *config) {

  unsigned long iVertex, iPoint, iPointNormal;
  unsigned short Boundary, Monitoring, iMarker, iMarker_Monitoring, iDim, jDim;
  su2double Viscosity = 0.0, div_vel, WallDist[3] = {0.0, 0.0, 0.0},
  Area, WallShearStress, TauNormal, factor, RefTemp, RefVel2, RefDensity, GradTemperature, Density = 0.0, WallDistMod, FrictionVel,
  Mach2Vel, Mach_Motion, UnitNormal[3] = {0.0, 0.0, 0.0}, TauElem[3] = {0.0, 0.0, 0.0}, TauTangent[3] = {0.0, 0.0, 0.0},
  Tau[3][3] = {{0.0, 0.0, 0.0},{0.0, 0.0, 0.0},{0.0, 0.0, 0.0}}, Cp, thermal_conductivity, MaxNorm = 8.0,
  Grad_Vel[3][3] = {{0.0, 0.0, 0.0},{0.0, 0.0, 0.0},{0.0, 0.0, 0.0}}, Grad_Temp[3] = {0.0, 0.0, 0.0},
  delta[3][3] = {{1.0, 0.0, 0.0},{0.0,1.0,0.0},{0.0,0.0,1.0}};
  su2double AxiFactor;
  const su2double *Coord = nullptr, *Coord_Normal = nullptr, *Normal = nullptr;

  string Marker_Tag, Monitoring_Tag;

  su2double Alpha = config->GetAoA()*PI_NUMBER/180.0;
  su2double Beta = config->GetAoS()*PI_NUMBER/180.0;
  su2double RefArea = config->GetRefArea();
  su2double RefLength = config->GetRefLength();
  su2double RefHeatFlux = config->GetHeat_Flux_Ref();
  su2double Gas_Constant = config->GetGas_ConstantND();
  const su2double *Origin = nullptr;

  if (config->GetnMarker_Monitoring() != 0) { Origin = config->GetRefOriginMoment(0); }

  su2double Prandtl_Lam = config->GetPrandtl_Lam();
  bool QCR = config->GetQCR();
  bool axisymmetric = config->GetAxisymmetric();

  /*--- Evaluate reference values for non-dimensionalization.
   For dynamic meshes, use the motion Mach number as a reference value
   for computing the force coefficients. Otherwise, use the freestream values,
   which is the standard convention. ---*/

  RefTemp = Temperature_Inf;
  RefDensity = Density_Inf;
  if (dynamic_grid) {
    Mach2Vel = sqrt(Gamma*Gas_Constant*RefTemp);
    Mach_Motion = config->GetMach_Motion();
    RefVel2 = (Mach_Motion*Mach2Vel)*(Mach_Motion*Mach2Vel);
  } else {
    RefVel2 = 0.0;
    for (iDim = 0; iDim < nDim; iDim++)
      RefVel2  += Velocity_Inf[iDim]*Velocity_Inf[iDim];
  }

  factor = 1.0 / (0.5*RefDensity*RefArea*RefVel2);

  /*--- Variables initialization ---*/

  AllBoundViscCoeff.setZero();
  SurfaceViscCoeff.setZero();

  AllBound_HF_Visc = 0.0;  AllBound_MaxHF_Visc = 0.0;

  for (iMarker_Monitoring = 0; iMarker_Monitoring < config->GetnMarker_Monitoring(); iMarker_Monitoring++) {
    Surface_HF_Visc[iMarker_Monitoring]  = 0.0; Surface_MaxHF_Visc[iMarker_Monitoring]   = 0.0;
  }

  /*--- Loop over the Navier-Stokes markers ---*/

  for (iMarker = 0; iMarker < nMarker; iMarker++) {

    Boundary = config->GetMarker_All_KindBC(iMarker);
    Monitoring = config->GetMarker_All_Monitoring(iMarker);

    /*--- Obtain the origin for the moment computation for a particular marker ---*/

    if (Monitoring == YES) {
      for (iMarker_Monitoring = 0; iMarker_Monitoring < config->GetnMarker_Monitoring(); iMarker_Monitoring++) {
        Monitoring_Tag = config->GetMarker_Monitoring_TagBound(iMarker_Monitoring);
        Marker_Tag = config->GetMarker_All_TagBound(iMarker);
        if (Marker_Tag == Monitoring_Tag)
          Origin = config->GetRefOriginMoment(iMarker_Monitoring);
      }
    }

    if ((Boundary == HEAT_FLUX) || (Boundary == ISOTHERMAL) || (Boundary == HEAT_FLUX) || (Boundary == CHT_WALL_INTERFACE)) {

      /*--- Forces initialization at each Marker ---*/

      ViscCoeff.setZero(iMarker);

      HF_Visc[iMarker] = 0.0;    MaxHF_Visc[iMarker] = 0.0;

      su2double ForceViscous[MAXNDIM] = {0.0}, MomentViscous[MAXNDIM] = {0.0};
      su2double MomentX_Force[MAXNDIM] = {0.0}, MomentY_Force[MAXNDIM] = {0.0}, MomentZ_Force[MAXNDIM] = {0.0};

      /*--- Loop over the vertices to compute the forces ---*/

      for (iVertex = 0; iVertex < geometry->nVertex[iMarker]; iVertex++) {

        iPoint = geometry->vertex[iMarker][iVertex]->GetNode();
        iPointNormal = geometry->vertex[iMarker][iVertex]->GetNormal_Neighbor();

        Coord = geometry->nodes->GetCoord(iPoint);
        Coord_Normal = geometry->nodes->GetCoord(iPointNormal);

        Normal = geometry->vertex[iMarker][iVertex]->GetNormal();

        for (iDim = 0; iDim < nDim; iDim++) {
          for (jDim = 0 ; jDim < nDim; jDim++) {
            Grad_Vel[iDim][jDim] = nodes->GetGradient_Primitive(iPoint,iDim+1, jDim);
          }
          Grad_Temp[iDim] = nodes->GetGradient_Primitive(iPoint,0, iDim);
        }

        Viscosity = nodes->GetLaminarViscosity(iPoint);
        Density = nodes->GetDensity(iPoint);

        Area = 0.0; for (iDim = 0; iDim < nDim; iDim++) Area += Normal[iDim]*Normal[iDim]; Area = sqrt(Area);


        for (iDim = 0; iDim < nDim; iDim++) {
          UnitNormal[iDim] = Normal[iDim]/Area;
        }

        /*--- Evaluate Tau ---*/

        div_vel = 0.0; for (iDim = 0; iDim < nDim; iDim++) div_vel += Grad_Vel[iDim][iDim];

        for (iDim = 0; iDim < nDim; iDim++) {
          for (jDim = 0 ; jDim < nDim; jDim++) {
            Tau[iDim][jDim] = Viscosity*(Grad_Vel[jDim][iDim] + Grad_Vel[iDim][jDim]) - TWO3*Viscosity*div_vel*delta[iDim][jDim];
          }
        }

        /*--- If necessary evaluate the QCR contribution to Tau ---*/

        if (QCR) {
          su2double den_aux, c_cr1=0.3, O_ik, O_jk;
          unsigned short kDim;

          /*--- Denominator Antisymmetric normalized rotation tensor ---*/

          den_aux = 0.0;
          for (iDim = 0 ; iDim < nDim; iDim++)
            for (jDim = 0 ; jDim < nDim; jDim++)
              den_aux += Grad_Vel[iDim][jDim] * Grad_Vel[iDim][jDim];
          den_aux = sqrt(max(den_aux,1E-10));

          /*--- Adding the QCR contribution ---*/

          for (iDim = 0 ; iDim < nDim; iDim++){
            for (jDim = 0 ; jDim < nDim; jDim++){
              for (kDim = 0 ; kDim < nDim; kDim++){
                O_ik = (Grad_Vel[iDim][kDim] - Grad_Vel[kDim][iDim])/ den_aux;
                O_jk = (Grad_Vel[jDim][kDim] - Grad_Vel[kDim][jDim])/ den_aux;
                Tau[iDim][jDim] -= c_cr1 * (O_ik * Tau[jDim][kDim] + O_jk * Tau[iDim][kDim]);
              }
            }
          }
        }

        /*--- Project Tau in each surface element ---*/

        for (iDim = 0; iDim < nDim; iDim++) {
          TauElem[iDim] = 0.0;
          for (jDim = 0; jDim < nDim; jDim++) {
            TauElem[iDim] += Tau[iDim][jDim]*UnitNormal[jDim];
          }
        }

        /*--- Compute wall shear stress (using the stress tensor). Compute wall skin friction coefficient, and heat flux on the wall ---*/

        TauNormal = 0.0; for (iDim = 0; iDim < nDim; iDim++) TauNormal += TauElem[iDim] * UnitNormal[iDim];

        WallShearStress = 0.0;
        for (iDim = 0; iDim < nDim; iDim++) {
          TauTangent[iDim] = TauElem[iDim] - TauNormal * UnitNormal[iDim];
          CSkinFriction[iMarker][iDim][iVertex] = TauTangent[iDim] / (0.5*RefDensity*RefVel2);
          WallShearStress += TauTangent[iDim] * TauTangent[iDim];
        }
        WallShearStress = sqrt(WallShearStress);

        for (iDim = 0; iDim < nDim; iDim++) WallDist[iDim] = (Coord[iDim] - Coord_Normal[iDim]);
        WallDistMod = 0.0; for (iDim = 0; iDim < nDim; iDim++) WallDistMod += WallDist[iDim]*WallDist[iDim]; WallDistMod = sqrt(WallDistMod);

        /*--- Compute y+ and non-dimensional velocity ---*/

        FrictionVel = sqrt(fabs(WallShearStress)/Density);
        if (config->Get_BoolStokesDrag() == true){
        	SetFrictionVel(iPoint, FrictionVel);
        }
        YPlus[iMarker][iVertex] = WallDistMod*FrictionVel/(Viscosity/Density);

        /*--- Compute total and maximum heat flux on the wall ---*/

        GradTemperature = 0.0;
        for (iDim = 0; iDim < nDim; iDim++)
          GradTemperature -= Grad_Temp[iDim]*UnitNormal[iDim];

        Cp = (Gamma / Gamma_Minus_One) * Gas_Constant;
        thermal_conductivity = Cp * Viscosity/Prandtl_Lam;
        HeatFlux[iMarker][iVertex] = -thermal_conductivity*GradTemperature*RefHeatFlux;

        /*--- Note that y+, and heat are computed at the
         halo cells (for visualization purposes), but not the forces ---*/

        if ((geometry->nodes->GetDomain(iPoint)) && (Monitoring == YES)) {

          /*--- Axisymmetric simulations ---*/

          if (axisymmetric) AxiFactor = 2.0*PI_NUMBER*geometry->nodes->GetCoord(iPoint, 1);
          else AxiFactor = 1.0;

          /*--- Force computation ---*/

          su2double Force[MAXNDIM] = {0.0}, MomentDist[MAXNDIM] = {0.0};
          for (iDim = 0; iDim < nDim; iDim++) {
            Force[iDim] = TauElem[iDim] * Area * factor * AxiFactor;
            ForceViscous[iDim] += Force[iDim];
            MomentDist[iDim] = Coord[iDim] - Origin[iDim];
          }

          /*--- Moment with respect to the reference axis ---*/

          if (iDim == 3) {
            MomentViscous[0] += (Force[2]*MomentDist[1] - Force[1]*MomentDist[2])/RefLength;
            MomentX_Force[1] += (-Force[1]*Coord[2]);
            MomentX_Force[2] += (Force[2]*Coord[1]);

            MomentViscous[1] += (Force[0]*MomentDist[2] - Force[2]*MomentDist[0])/RefLength;
            MomentY_Force[2] += (-Force[2]*Coord[0]);
            MomentY_Force[0] += (Force[0]*Coord[2]);
          }
          MomentViscous[2] += (Force[1]*MomentDist[0] - Force[0]*MomentDist[1])/RefLength;
          MomentZ_Force[0] += (-Force[0]*Coord[1]);
          MomentZ_Force[1] += (Force[1]*Coord[0]);

        }

        HF_Visc[iMarker]          += HeatFlux[iMarker][iVertex]*Area;
        MaxHF_Visc[iMarker]       += pow(HeatFlux[iMarker][iVertex], MaxNorm);

      }

      /*--- Project forces and store the non-dimensional coefficients ---*/

      if (Monitoring == YES) {
        if (nDim == 2) {
          ViscCoeff.CD[iMarker]          =  ForceViscous[0]*cos(Alpha) + ForceViscous[1]*sin(Alpha);
          ViscCoeff.CL[iMarker]          = -ForceViscous[0]*sin(Alpha) + ForceViscous[1]*cos(Alpha);
          ViscCoeff.CEff[iMarker]        = ViscCoeff.CL[iMarker] / (ViscCoeff.CD[iMarker]+EPS);
          ViscCoeff.CFx[iMarker]         = ForceViscous[0];
          ViscCoeff.CFy[iMarker]         = ForceViscous[1];
          ViscCoeff.CMz[iMarker]         = MomentViscous[2];
          ViscCoeff.CoPx[iMarker]        = MomentZ_Force[1];
          ViscCoeff.CoPy[iMarker]        = -MomentZ_Force[0];
          ViscCoeff.CT[iMarker]          = -ViscCoeff.CFx[iMarker];
          ViscCoeff.CQ[iMarker]          = -ViscCoeff.CMz[iMarker];
          ViscCoeff.CMerit[iMarker]      = ViscCoeff.CT[iMarker] / (ViscCoeff.CQ[iMarker]+EPS);
          MaxHF_Visc[iMarker]            = pow(MaxHF_Visc[iMarker], 1.0/MaxNorm);
        }
        if (nDim == 3) {
          ViscCoeff.CD[iMarker]      =  ForceViscous[0]*cos(Alpha)*cos(Beta) + ForceViscous[1]*sin(Beta) + ForceViscous[2]*sin(Alpha)*cos(Beta);
          ViscCoeff.CL[iMarker]          = -ForceViscous[0]*sin(Alpha) + ForceViscous[2]*cos(Alpha);
          ViscCoeff.CSF[iMarker]         = -ForceViscous[0]*sin(Beta)*cos(Alpha) + ForceViscous[1]*cos(Beta) - ForceViscous[2]*sin(Beta)*sin(Alpha);
          ViscCoeff.CEff[iMarker]        = ViscCoeff.CL[iMarker]/(ViscCoeff.CD[iMarker] + EPS);
          ViscCoeff.CFx[iMarker]         = ForceViscous[0];
          ViscCoeff.CFy[iMarker]         = ForceViscous[1];
          ViscCoeff.CFz[iMarker]         = ForceViscous[2];
          ViscCoeff.CMx[iMarker]         = MomentViscous[0];
          ViscCoeff.CMy[iMarker]         = MomentViscous[1];
          ViscCoeff.CMz[iMarker]         = MomentViscous[2];
          ViscCoeff.CoPx[iMarker]        = -MomentY_Force[0];
          ViscCoeff.CoPz[iMarker]        = MomentY_Force[2];
          ViscCoeff.CT[iMarker]          = -ViscCoeff.CFz[iMarker];
          ViscCoeff.CQ[iMarker]          = -ViscCoeff.CMz[iMarker];
          ViscCoeff.CMerit[iMarker]      = ViscCoeff.CT[iMarker] / (ViscCoeff.CQ[iMarker] + EPS);
          MaxHF_Visc[iMarker]            = pow(MaxHF_Visc[iMarker], 1.0/MaxNorm);
        }

        AllBoundViscCoeff.CD          += ViscCoeff.CD[iMarker];
        AllBoundViscCoeff.CL          += ViscCoeff.CL[iMarker];
        AllBoundViscCoeff.CSF         += ViscCoeff.CSF[iMarker];
        AllBoundViscCoeff.CFx         += ViscCoeff.CFx[iMarker];
        AllBoundViscCoeff.CFy         += ViscCoeff.CFy[iMarker];
        AllBoundViscCoeff.CFz         += ViscCoeff.CFz[iMarker];
        AllBoundViscCoeff.CMx         += ViscCoeff.CMx[iMarker];
        AllBoundViscCoeff.CMy         += ViscCoeff.CMy[iMarker];
        AllBoundViscCoeff.CMz         += ViscCoeff.CMz[iMarker];
        AllBoundViscCoeff.CoPx        += ViscCoeff.CoPx[iMarker];
        AllBoundViscCoeff.CoPy        += ViscCoeff.CoPy[iMarker];
        AllBoundViscCoeff.CoPz        += ViscCoeff.CoPz[iMarker];
        AllBoundViscCoeff.CT          += ViscCoeff.CT[iMarker];
        AllBoundViscCoeff.CQ          += ViscCoeff.CQ[iMarker];
        AllBound_HF_Visc              += HF_Visc[iMarker];
        AllBound_MaxHF_Visc           += pow(MaxHF_Visc[iMarker], MaxNorm);

        /*--- Compute the coefficients per surface ---*/

        for (iMarker_Monitoring = 0; iMarker_Monitoring < config->GetnMarker_Monitoring(); iMarker_Monitoring++) {
          Monitoring_Tag = config->GetMarker_Monitoring_TagBound(iMarker_Monitoring);
          Marker_Tag = config->GetMarker_All_TagBound(iMarker);
          if (Marker_Tag == Monitoring_Tag) {
            SurfaceViscCoeff.CL[iMarker_Monitoring]      += ViscCoeff.CL[iMarker];
            SurfaceViscCoeff.CD[iMarker_Monitoring]      += ViscCoeff.CD[iMarker];
            SurfaceViscCoeff.CSF[iMarker_Monitoring]     += ViscCoeff.CSF[iMarker];
            SurfaceViscCoeff.CEff[iMarker_Monitoring]    += ViscCoeff.CEff[iMarker];
            SurfaceViscCoeff.CFx[iMarker_Monitoring]     += ViscCoeff.CFx[iMarker];
            SurfaceViscCoeff.CFy[iMarker_Monitoring]     += ViscCoeff.CFy[iMarker];
            SurfaceViscCoeff.CFz[iMarker_Monitoring]     += ViscCoeff.CFz[iMarker];
            SurfaceViscCoeff.CMx[iMarker_Monitoring]     += ViscCoeff.CMx[iMarker];
            SurfaceViscCoeff.CMy[iMarker_Monitoring]     += ViscCoeff.CMy[iMarker];
            SurfaceViscCoeff.CMz[iMarker_Monitoring]     += ViscCoeff.CMz[iMarker];
            Surface_HF_Visc[iMarker_Monitoring]          += HF_Visc[iMarker];
            Surface_MaxHF_Visc[iMarker_Monitoring]       += pow(MaxHF_Visc[iMarker],MaxNorm);
          }
        }

      }

    }
  }

  /*--- Update some global coefficients ---*/

  AllBoundViscCoeff.CEff = AllBoundViscCoeff.CL / (AllBoundViscCoeff.CD + EPS);
  AllBoundViscCoeff.CMerit = AllBoundViscCoeff.CT / (AllBoundViscCoeff.CQ + EPS);
  AllBound_MaxHF_Visc = pow(AllBound_MaxHF_Visc, 1.0/MaxNorm);


#ifdef HAVE_MPI

  /*--- Add AllBound information using all the nodes ---*/

  if (config->GetComm_Level() == COMM_FULL) {

    auto Allreduce = [](su2double x) {
      su2double tmp = x; x = 0.0;
      SU2_MPI::Allreduce(&tmp, &x, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      return x;
    };
    AllBoundViscCoeff.CD = Allreduce(AllBoundViscCoeff.CD);
    AllBoundViscCoeff.CL = Allreduce(AllBoundViscCoeff.CL);
    AllBoundViscCoeff.CSF = Allreduce(AllBoundViscCoeff.CSF);
    AllBoundViscCoeff.CEff = AllBoundViscCoeff.CL / (AllBoundViscCoeff.CD + EPS);

    AllBoundViscCoeff.CMx = Allreduce(AllBoundViscCoeff.CMx);
    AllBoundViscCoeff.CMy = Allreduce(AllBoundViscCoeff.CMy);
    AllBoundViscCoeff.CMz = Allreduce(AllBoundViscCoeff.CMz);

    AllBoundViscCoeff.CFx = Allreduce(AllBoundViscCoeff.CFx);
    AllBoundViscCoeff.CFy = Allreduce(AllBoundViscCoeff.CFy);
    AllBoundViscCoeff.CFz = Allreduce(AllBoundViscCoeff.CFz);

    AllBoundViscCoeff.CoPx = Allreduce(AllBoundViscCoeff.CoPx);
    AllBoundViscCoeff.CoPy = Allreduce(AllBoundViscCoeff.CoPy);
    AllBoundViscCoeff.CoPz = Allreduce(AllBoundViscCoeff.CoPz);

    AllBoundViscCoeff.CT = Allreduce(AllBoundViscCoeff.CT);
    AllBoundViscCoeff.CQ = Allreduce(AllBoundViscCoeff.CQ);
    AllBoundViscCoeff.CMerit = AllBoundViscCoeff.CT / (AllBoundViscCoeff.CQ + EPS);

    AllBound_HF_Visc = Allreduce(AllBound_HF_Visc);
    AllBound_MaxHF_Visc = pow(Allreduce(pow(AllBound_MaxHF_Visc, MaxNorm)), 1.0/MaxNorm);

  }

  /*--- Add the forces on the surfaces using all the nodes ---*/

  if (config->GetComm_Level() == COMM_FULL) {

    int nMarkerMon = config->GetnMarker_Monitoring();

    /*--- Use the same buffer for all reductions. We could avoid the copy back into
     *    the original variable by swaping pointers, but it is safer this way... ---*/

    su2double* buffer = new su2double [nMarkerMon];

    auto Allreduce_inplace = [buffer](int size, su2double* x) {
      SU2_MPI::Allreduce(x, buffer, size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      for(int i=0; i<size; ++i) x[i] = buffer[i];
    };

    Allreduce_inplace(nMarkerMon, SurfaceViscCoeff.CL);
    Allreduce_inplace(nMarkerMon, SurfaceViscCoeff.CD);
    Allreduce_inplace(nMarkerMon, SurfaceViscCoeff.CSF);

    for (iMarker_Monitoring = 0; iMarker_Monitoring < nMarkerMon; iMarker_Monitoring++)
      SurfaceViscCoeff.CEff[iMarker_Monitoring] = SurfaceViscCoeff.CL[iMarker_Monitoring] /
                                                 (SurfaceViscCoeff.CD[iMarker_Monitoring] + EPS);

    Allreduce_inplace(nMarkerMon, SurfaceViscCoeff.CFx);
    Allreduce_inplace(nMarkerMon, SurfaceViscCoeff.CFy);
    Allreduce_inplace(nMarkerMon, SurfaceViscCoeff.CFz);

    Allreduce_inplace(nMarkerMon, SurfaceViscCoeff.CMx);
    Allreduce_inplace(nMarkerMon, SurfaceViscCoeff.CMy);
    Allreduce_inplace(nMarkerMon, SurfaceViscCoeff.CMz);

    Allreduce_inplace(nMarkerMon, Surface_HF_Visc);
    Allreduce_inplace(nMarkerMon, Surface_MaxHF_Visc);

    delete [] buffer;

  }

#endif

  /*--- Update the total coefficients (note that all the nodes have the same value)---*/

  TotalCoeff.CD          += AllBoundViscCoeff.CD;
  TotalCoeff.CL          += AllBoundViscCoeff.CL;
  TotalCoeff.CSF         += AllBoundViscCoeff.CSF;
  TotalCoeff.CEff         = TotalCoeff.CL / (TotalCoeff.CD + EPS);
  TotalCoeff.CFx         += AllBoundViscCoeff.CFx;
  TotalCoeff.CFy         += AllBoundViscCoeff.CFy;
  TotalCoeff.CFz         += AllBoundViscCoeff.CFz;
  TotalCoeff.CMx         += AllBoundViscCoeff.CMx;
  TotalCoeff.CMy         += AllBoundViscCoeff.CMy;
  TotalCoeff.CMz         += AllBoundViscCoeff.CMz;
  TotalCoeff.CoPx        += AllBoundViscCoeff.CoPx;
  TotalCoeff.CoPy        += AllBoundViscCoeff.CoPy;
  TotalCoeff.CoPz        += AllBoundViscCoeff.CoPz;
  TotalCoeff.CT          += AllBoundViscCoeff.CT;
  TotalCoeff.CQ          += AllBoundViscCoeff.CQ;
  TotalCoeff.CMerit       = AllBoundViscCoeff.CT / (AllBoundViscCoeff.CQ + EPS);
  Total_Heat         = AllBound_HF_Visc;
  Total_MaxHeat      = AllBound_MaxHF_Visc;

  /*--- Update the total coefficients per surface (note that all the nodes have the same value)---*/

  for (iMarker_Monitoring = 0; iMarker_Monitoring < config->GetnMarker_Monitoring(); iMarker_Monitoring++) {
    SurfaceCoeff.CL[iMarker_Monitoring]         += SurfaceViscCoeff.CL[iMarker_Monitoring];
    SurfaceCoeff.CD[iMarker_Monitoring]         += SurfaceViscCoeff.CD[iMarker_Monitoring];
    SurfaceCoeff.CSF[iMarker_Monitoring]        += SurfaceViscCoeff.CSF[iMarker_Monitoring];
    SurfaceCoeff.CEff[iMarker_Monitoring]        = SurfaceViscCoeff.CL[iMarker_Monitoring] / (SurfaceCoeff.CD[iMarker_Monitoring] + EPS);
    SurfaceCoeff.CFx[iMarker_Monitoring]        += SurfaceViscCoeff.CFx[iMarker_Monitoring];
    SurfaceCoeff.CFy[iMarker_Monitoring]        += SurfaceViscCoeff.CFy[iMarker_Monitoring];
    SurfaceCoeff.CFz[iMarker_Monitoring]        += SurfaceViscCoeff.CFz[iMarker_Monitoring];
    SurfaceCoeff.CMx[iMarker_Monitoring]        += SurfaceViscCoeff.CMx[iMarker_Monitoring];
    SurfaceCoeff.CMy[iMarker_Monitoring]        += SurfaceViscCoeff.CMy[iMarker_Monitoring];
    SurfaceCoeff.CMz[iMarker_Monitoring]        += SurfaceViscCoeff.CMz[iMarker_Monitoring];
  }


  // Add to each node the computed value of the viscous drag reduction using Stokes method
  // Note: Each node will have info about the drag of the whole system (and not only that of the part of the domain associated to the node).
  // This is possible because after this point there are no additional MPI operations (e.g. Allreduce())
  if (config->Get_BoolStokesDrag() == true){
	su2double delta_cd;
	delta_cd = Compute_ViscCD_StokesMethod(geometry, config); // The TOTAL viscous Cd reduction is computed by every processor (no need to sum them up).
	if (config->GetnMarker_Monitoring() == 1){// Currently works only if there is only 1 marker that is monitored
      iMarker_Monitoring = 0;
	  SurfaceCoeff.CD[iMarker_Monitoring] += delta_cd;
	  TotalCoeff.CD += delta_cd; //add/subtract drag increase/reduction from viscous drag coefficient
	}
  }

}

void CNSSolver::Buffet_Monitoring(CGeometry *geometry, CConfig *config) {

  unsigned long iVertex;
  unsigned short Boundary, Monitoring, iMarker, iMarker_Monitoring, iDim;
  su2double *Vel_FS = config->GetVelocity_FreeStream();
  su2double VelMag_FS = 0.0, SkinFrictionMag = 0.0, SkinFrictionDot = 0.0, *Normal, Area, Sref = config->GetRefArea();
  su2double k   = config->GetBuffet_k(),
             lam = config->GetBuffet_lambda();
  string Marker_Tag, Monitoring_Tag;

  for (iDim = 0; iDim < nDim; iDim++){
    VelMag_FS += Vel_FS[iDim]*Vel_FS[iDim];
  }
  VelMag_FS = sqrt(VelMag_FS);

  /*-- Variables initialization ---*/

  Total_Buffet_Metric = 0.0;

  for (iMarker_Monitoring = 0; iMarker_Monitoring < config->GetnMarker_Monitoring(); iMarker_Monitoring++) {
    Surface_Buffet_Metric[iMarker_Monitoring] = 0.0;
  }

  /*--- Loop over the Euler and Navier-Stokes markers ---*/

  for (iMarker = 0; iMarker < nMarker; iMarker++) {

    Buffet_Metric[iMarker] = 0.0;

    Boundary   = config->GetMarker_All_KindBC(iMarker);
    Monitoring = config->GetMarker_All_Monitoring(iMarker);

    if ((Boundary == HEAT_FLUX) || (Boundary == ISOTHERMAL) || (Boundary == HEAT_FLUX) || (Boundary == CHT_WALL_INTERFACE)) {

      /*--- Loop over the vertices to compute the buffet sensor ---*/

      for (iVertex = 0; iVertex < geometry->nVertex[iMarker]; iVertex++) {

        /*--- Perform dot product of skin friction with freestream velocity ---*/

        SkinFrictionMag = 0.0;
        SkinFrictionDot = 0.0;
        for(iDim = 0; iDim < nDim; iDim++){
          SkinFrictionMag += CSkinFriction[iMarker][iDim][iVertex]*CSkinFriction[iMarker][iDim][iVertex];
          SkinFrictionDot += CSkinFriction[iMarker][iDim][iVertex]*Vel_FS[iDim];
        }
        SkinFrictionMag = sqrt(SkinFrictionMag);

        /*--- Normalize the dot product ---*/

        SkinFrictionDot /= SkinFrictionMag*VelMag_FS;

        /*--- Compute Heaviside function ---*/

        Buffet_Sensor[iMarker][iVertex] = 1./(1. + exp(2.*k*(SkinFrictionDot + lam)));

        /*--- Integrate buffet sensor ---*/

        if(Monitoring == YES){

          Normal = geometry->vertex[iMarker][iVertex]->GetNormal();
          Area = 0.0;
          for(iDim = 0; iDim < nDim; iDim++) Area += Normal[iDim]*Normal[iDim];
          Area = sqrt(Area);

          Buffet_Metric[iMarker] += Buffet_Sensor[iMarker][iVertex]*Area/Sref;

        }

      }

      if(Monitoring == YES){

        Total_Buffet_Metric += Buffet_Metric[iMarker];

        /*--- Per surface buffet metric ---*/

        for (iMarker_Monitoring = 0; iMarker_Monitoring < config->GetnMarker_Monitoring(); iMarker_Monitoring++) {
          Monitoring_Tag = config->GetMarker_Monitoring_TagBound(iMarker_Monitoring);
          Marker_Tag = config->GetMarker_All_TagBound(iMarker);
          if (Marker_Tag == Monitoring_Tag) Surface_Buffet_Metric[iMarker_Monitoring] = Buffet_Metric[iMarker];
        }

      }

    }

  }

#ifdef HAVE_MPI

  /*--- Add buffet metric information using all the nodes ---*/

  su2double MyTotal_Buffet_Metric = Total_Buffet_Metric;
  SU2_MPI::Allreduce(&MyTotal_Buffet_Metric, &Total_Buffet_Metric, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  /*--- Add the buffet metric on the surfaces using all the nodes ---*/

  su2double *MySurface_Buffet_Metric = new su2double[config->GetnMarker_Monitoring()];

  for (iMarker_Monitoring = 0; iMarker_Monitoring < config->GetnMarker_Monitoring(); iMarker_Monitoring++) {
    MySurface_Buffet_Metric[iMarker_Monitoring] = Surface_Buffet_Metric[iMarker_Monitoring];
  }

  SU2_MPI::Allreduce(MySurface_Buffet_Metric, Surface_Buffet_Metric, config->GetnMarker_Monitoring(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  delete [] MySurface_Buffet_Metric;

#endif

}

void CNSSolver::Evaluate_ObjFunc(CConfig *config) {

  unsigned short iMarker_Monitoring, Kind_ObjFunc;
  su2double Weight_ObjFunc;

  /*--- Evaluate objective functions common to Euler and NS solvers ---*/

  CEulerSolver::Evaluate_ObjFunc(config);

  /*--- Evaluate objective functions specific to NS solver ---*/

  for (iMarker_Monitoring = 0; iMarker_Monitoring < config->GetnMarker_Monitoring(); iMarker_Monitoring++) {

    Weight_ObjFunc = config->GetWeight_ObjFunc(iMarker_Monitoring);
    Kind_ObjFunc = config->GetKind_ObjFunc(iMarker_Monitoring);

    switch(Kind_ObjFunc) {
      case BUFFET_SENSOR:
          Total_ComboObj +=Weight_ObjFunc*Surface_Buffet_Metric[iMarker_Monitoring];
          break;
      default:
          break;
    }
  }

}

void CNSSolver::SetRoe_Dissipation(CGeometry *geometry, CConfig *config){

  const unsigned short kind_roe_dissipation = config->GetKind_RoeLowDiss();

  SU2_OMP_FOR_STAT(omp_chunk_size)
  for (unsigned long iPoint = 0; iPoint < nPoint; iPoint++) {

    if (kind_roe_dissipation == FD || kind_roe_dissipation == FD_DUCROS){

      su2double wall_distance = geometry->nodes->GetWall_Distance(iPoint);

      nodes->SetRoe_Dissipation_FD(iPoint, wall_distance);

    } else if (kind_roe_dissipation == NTS || kind_roe_dissipation == NTS_DUCROS) {

      const su2double delta = geometry->nodes->GetMaxLength(iPoint);
      assert(delta > 0 && "Delta must be initialized and non-negative");
      nodes->SetRoe_Dissipation_NTS(iPoint, delta, config->GetConst_DES());
    }
  }

}

void CNSSolver::AddDynamicGridResidualContribution(unsigned long iPoint, unsigned long Point_Normal,
                                                   CGeometry* geometry,  const su2double* UnitNormal,
                                                   su2double Area, const su2double* GridVel,
                                                   su2double** Jacobian_i, su2double& Res_Conv,
                                                   su2double& Res_Visc) const {

  su2double ProjGridVel = Area * GeometryToolbox::DotProduct(nDim, GridVel, UnitNormal);

  /*--- Retrieve other primitive quantities and viscosities ---*/

  su2double Density = nodes->GetDensity(iPoint);
  su2double Pressure = nodes->GetPressure(iPoint);
  su2double laminar_viscosity = nodes->GetLaminarViscosity(iPoint);
  su2double eddy_viscosity = nodes->GetEddyViscosity(iPoint);
  su2double total_viscosity = laminar_viscosity + eddy_viscosity;

  const auto Grad_Vel = &nodes->GetGradient_Primitive(iPoint)[1];

  /*--- Divergence of the velocity ---*/

  su2double div_vel = 0.0;
  for (auto iDim = 0u; iDim < nDim; iDim++)
    div_vel += Grad_Vel[iDim][iDim];

  /*--- Compute the viscous stress tensor ---*/

  su2double tau[MAXNDIM][MAXNDIM] = {{0.0}};
  for (auto iDim = 0u; iDim < nDim; iDim++) {
    for (auto jDim = 0u; jDim < nDim; jDim++) {
      tau[iDim][jDim] = total_viscosity * (Grad_Vel[jDim][iDim] + Grad_Vel[iDim][jDim]);
    }
    tau[iDim][iDim] -= TWO3*total_viscosity*div_vel;
  }

  /*--- Dot product of the stress tensor with the grid velocity ---*/

  su2double tau_vel[MAXNDIM] = {0.0};
  for (auto iDim = 0u; iDim < nDim; iDim++)
    tau_vel[iDim] = GeometryToolbox::DotProduct(nDim, tau[iDim], GridVel);

  /*--- Compute the convective and viscous residuals (energy eqn.) ---*/

  Res_Conv += Pressure*ProjGridVel;
  Res_Visc += GeometryToolbox::DotProduct(nDim, tau_vel, UnitNormal) * Area;

  /*--- Implicit Jacobian contributions due to moving walls ---*/

  if (Jacobian_i != nullptr) {

    /*--- Jacobian contribution related to the pressure term ---*/

    su2double GridVel2 = GeometryToolbox::SquaredNorm(nDim, GridVel);

    Jacobian_i[nDim+1][0] += 0.5*(Gamma-1.0)*GridVel2*ProjGridVel;

    for (auto jDim = 0u; jDim < nDim; jDim++)
      Jacobian_i[nDim+1][jDim+1] += -(Gamma-1.0)*GridVel[jDim]*ProjGridVel;

    Jacobian_i[nDim+1][nDim+1] += (Gamma-1.0)*ProjGridVel;

    /*--- Now the Jacobian contribution related to the shear stress ---*/

    /*--- Get coordinates of i & nearest normal and compute distance ---*/

    const auto Coord_i = geometry->nodes->GetCoord(iPoint);
    const auto Coord_j = geometry->nodes->GetCoord(Point_Normal);

    su2double dist_ij = GeometryToolbox::Distance(nDim, Coord_i, Coord_j);

    const su2double theta2 = 1.0;

    su2double factor = total_viscosity*Area/(Density*dist_ij);

    if (nDim == 2) {
      su2double thetax = theta2 + UnitNormal[0]*UnitNormal[0]/3.0;
      su2double thetay = theta2 + UnitNormal[1]*UnitNormal[1]/3.0;

      su2double etaz = UnitNormal[0]*UnitNormal[1]/3.0;

      su2double pix = GridVel[0]*thetax + GridVel[1]*etaz;
      su2double piy = GridVel[0]*etaz   + GridVel[1]*thetay;

      Jacobian_i[nDim+1][0] += factor*(-pix*GridVel[0]+piy*GridVel[1]);
      Jacobian_i[nDim+1][1] += factor*pix;
      Jacobian_i[nDim+1][2] += factor*piy;
    }
    else {
      su2double thetax = theta2 + UnitNormal[0]*UnitNormal[0]/3.0;
      su2double thetay = theta2 + UnitNormal[1]*UnitNormal[1]/3.0;
      su2double thetaz = theta2 + UnitNormal[2]*UnitNormal[2]/3.0;

      su2double etaz = UnitNormal[0]*UnitNormal[1]/3.0;
      su2double etax = UnitNormal[1]*UnitNormal[2]/3.0;
      su2double etay = UnitNormal[0]*UnitNormal[2]/3.0;

      su2double pix = GridVel[0]*thetax + GridVel[1]*etaz   + GridVel[2]*etay;
      su2double piy = GridVel[0]*etaz   + GridVel[1]*thetay + GridVel[2]*etax;
      su2double piz = GridVel[0]*etay   + GridVel[1]*etax   + GridVel[2]*thetaz;

      Jacobian_i[nDim+1][0] += factor*(-pix*GridVel[0]+piy*GridVel[1]+piz*GridVel[2]);
      Jacobian_i[nDim+1][1] += factor*pix;
      Jacobian_i[nDim+1][2] += factor*piy;
      Jacobian_i[nDim+1][3] += factor*piz;
    }
  }
}

void CNSSolver::BC_HeatFlux_Wall(CGeometry *geometry, CSolver **solver_container, CNumerics *conv_numerics,
                                 CNumerics *visc_numerics, CConfig *config, unsigned short val_marker) {

  /*--- Identify the boundary by string name and get the specified wall
   heat flux from config as well as the wall function treatment. ---*/

  const bool implicit = (config->GetKind_TimeIntScheme() == EULER_IMPLICIT);
  const auto Marker_Tag = config->GetMarker_All_TagBound(val_marker);
  su2double Wall_HeatFlux = config->GetWall_HeatFlux(Marker_Tag)/config->GetHeat_Flux_Ref();

//  Wall_Function = config->GetWallFunction_Treatment(Marker_Tag);
//  if (Wall_Function != NO_WALL_FUNCTION) {
//    SU2_MPI::Error("Wall function treament not implemented yet", CURRENT_FUNCTION);
//  }

  /*--- Jacobian, initialized to zero if needed. ---*/
  su2double **Jacobian_i = nullptr;
  if (dynamic_grid && implicit) {
    Jacobian_i = new su2double* [nVar];
    for (auto iVar = 0u; iVar < nVar; iVar++)
      Jacobian_i[iVar] = new su2double [nVar] ();
  }

  /*--- Loop over all of the vertices on this boundary marker ---*/

  SU2_OMP_FOR_DYN(OMP_MIN_SIZE)
  for (auto iVertex = 0u; iVertex < geometry->nVertex[val_marker]; iVertex++) {

    const auto iPoint = geometry->vertex[val_marker][iVertex]->GetNode();

    /*--- Check if the node belongs to the domain (i.e, not a halo node) ---*/

    if (!geometry->nodes->GetDomain(iPoint)) continue;

    /*--- If it is a customizable patch, retrieve the specified wall heat flux. ---*/

    if (config->GetMarker_All_PyCustom(val_marker))
      Wall_HeatFlux = geometry->GetCustomBoundaryHeatFlux(val_marker, iVertex);

    /*--- Compute dual-grid area and boundary normal ---*/

    const auto Normal = geometry->vertex[val_marker][iVertex]->GetNormal();

    su2double Area = GeometryToolbox::Norm(nDim, Normal);

    su2double UnitNormal[MAXNDIM] = {0.0};
    for (auto iDim = 0u; iDim < nDim; iDim++)
      UnitNormal[iDim] = -Normal[iDim]/Area;

    /*--- Apply a weak boundary condition for the energy equation.
     Compute the residual due to the prescribed heat flux.
     The convective part will be zero if the grid is not moving. ---*/

    su2double Res_Conv = 0.0;
    su2double Res_Visc = Wall_HeatFlux * Area;

    /*--- Impose the value of the velocity as a strong boundary
     condition (Dirichlet). Fix the velocity and remove any
     contribution to the residual at this node. ---*/

    if (dynamic_grid) {
      nodes->SetVelocity_Old(iPoint, geometry->nodes->GetGridVel(iPoint));
    }
    else {
      su2double zero[MAXNDIM] = {0.0};
      nodes->SetVelocity_Old(iPoint, zero);
    }

    for (auto iDim = 0u; iDim < nDim; iDim++)
      LinSysRes.SetBlock_Zero(iPoint, iDim+1);
    nodes->SetVel_ResTruncError_Zero(iPoint);

    /*--- If the wall is moving, there are additional residual contributions
     due to pressure (p v_wall.n) and shear stress (tau.v_wall.n). ---*/

    if (dynamic_grid) {
      if (implicit) {
        for (auto iVar = 0u; iVar < nVar; ++iVar)
          Jacobian_i[nDim+1][iVar] = 0.0;
      }

      const auto Point_Normal = geometry->vertex[val_marker][iVertex]->GetNormal_Neighbor();

      AddDynamicGridResidualContribution(iPoint, Point_Normal, geometry, UnitNormal,
                                         Area, geometry->nodes->GetGridVel(iPoint),
                                         Jacobian_i, Res_Conv, Res_Visc);
    }

    /*--- Convective and viscous contributions to the residual at the wall ---*/

    LinSysRes(iPoint, nDim+1) += Res_Conv - Res_Visc;

    /*--- Enforce the no-slip boundary condition in a strong way by
     modifying the velocity-rows of the Jacobian (1 on the diagonal).
     And add the contributions to the Jacobian due to energy. ---*/

    if (implicit) {
      if (dynamic_grid) {
        Jacobian.AddBlock2Diag(iPoint, Jacobian_i);
      }

      for (auto iVar = 1u; iVar <= nDim; iVar++) {
        auto total_index = iPoint*nVar+iVar;
        Jacobian.DeleteValsRowi(total_index);
      }
    }
  }

  if (Jacobian_i)
    for (auto iVar = 0u; iVar < nVar; iVar++)
      delete [] Jacobian_i[iVar];
  delete [] Jacobian_i;

}

su2double CNSSolver::GetCHTWallTemperature(const CConfig* config, unsigned short val_marker,
                                           unsigned long iVertex, su2double thermal_conductivity,
                                           su2double dist_ij, su2double There,
                                           su2double Temperature_Ref) const {

  /*--- Compute the normal gradient in temperature using Twall ---*/

  const su2double Tconjugate = GetConjugateHeatVariable(val_marker, iVertex, 0) / Temperature_Ref;

  su2double Twall = 0.0;

  if ((config->GetKind_CHT_Coupling() == AVERAGED_TEMPERATURE_NEUMANN_HEATFLUX) ||
      (config->GetKind_CHT_Coupling() == AVERAGED_TEMPERATURE_ROBIN_HEATFLUX)) {

    /*--- Compute wall temperature from both temperatures ---*/

    su2double HF_FactorHere = thermal_conductivity*config->GetViscosity_Ref()/dist_ij;
    su2double HF_FactorConjugate = GetConjugateHeatVariable(val_marker, iVertex, 2);

    Twall = (There*HF_FactorHere + Tconjugate*HF_FactorConjugate)/(HF_FactorHere + HF_FactorConjugate);
  }
  else if ((config->GetKind_CHT_Coupling() == DIRECT_TEMPERATURE_NEUMANN_HEATFLUX) ||
           (config->GetKind_CHT_Coupling() == DIRECT_TEMPERATURE_ROBIN_HEATFLUX)) {

    /*--- (Directly) Set wall temperature to conjugate temperature. ---*/

    Twall = Tconjugate;
  }
  else {
    SU2_MPI::Error("Unknown CHT coupling method.", CURRENT_FUNCTION);
  }

  return Twall;
}

void CNSSolver::BC_Isothermal_Wall_Generic(CGeometry *geometry, CSolver **solver_container,
                                           CNumerics *conv_numerics, CNumerics *visc_numerics,
                                           CConfig *config, unsigned short val_marker, bool cht_mode) {

  const bool implicit = (config->GetKind_TimeIntScheme() == EULER_IMPLICIT);
  const su2double Temperature_Ref = config->GetTemperature_Ref();
  const su2double Prandtl_Lam = config->GetPrandtl_Lam();
  const su2double Prandtl_Turb = config->GetPrandtl_Turb();
  const su2double Gas_Constant = config->GetGas_ConstantND();
  const su2double Cp = (Gamma / Gamma_Minus_One) * Gas_Constant;

  /*--- Identify the boundary and retrieve the specified wall temperature from
   the config (for non-CHT problems) as well as the wall function treatment. ---*/

  const auto Marker_Tag = config->GetMarker_All_TagBound(val_marker);
  su2double Twall = 0.0;
  if (!cht_mode) {
    Twall = config->GetIsothermal_Temperature(Marker_Tag) / Temperature_Ref;
  }

//  Wall_Function = config->GetWallFunction_Treatment(Marker_Tag);
//  if (Wall_Function != NO_WALL_FUNCTION) {
//    SU2_MPI::Error("Wall function treament not implemented yet", CURRENT_FUNCTION);
//  }

  su2double **Jacobian_i = nullptr;
  if (implicit) {
    Jacobian_i = new su2double* [nVar];
    for (auto iVar = 0u; iVar < nVar; iVar++)
      Jacobian_i[iVar] = new su2double [nVar] ();
  }

  /*--- Loop over boundary points ---*/

  SU2_OMP_FOR_DYN(OMP_MIN_SIZE)
  for (auto iVertex = 0u; iVertex < geometry->nVertex[val_marker]; iVertex++) {

    const auto iPoint = geometry->vertex[val_marker][iVertex]->GetNode();

    if (!geometry->nodes->GetDomain(iPoint)) continue;

    /*--- Compute dual-grid area and boundary normal ---*/

    const auto Normal = geometry->vertex[val_marker][iVertex]->GetNormal();

    su2double Area = GeometryToolbox::Norm(nDim, Normal);

    su2double UnitNormal[MAXNDIM] = {0.0};
    for (auto iDim = 0u; iDim < nDim; iDim++)
      UnitNormal[iDim] = -Normal[iDim]/Area;

    /*--- Compute closest normal neighbor ---*/

    const auto Point_Normal = geometry->vertex[val_marker][iVertex]->GetNormal_Neighbor();

    /*--- Get coordinates of i & nearest normal and compute distance ---*/

    const auto Coord_i = geometry->nodes->GetCoord(iPoint);
    const auto Coord_j = geometry->nodes->GetCoord(Point_Normal);

    su2double dist_ij = GeometryToolbox::Distance(nDim, Coord_i, Coord_j);

    /*--- Store the corrected velocity at the wall which will
     be zero (v = 0), unless there is grid motion (v = u_wall)---*/

    if (dynamic_grid) {
      nodes->SetVelocity_Old(iPoint, geometry->nodes->GetGridVel(iPoint));
    }
    else {
      su2double zero[MAXNDIM] = {0.0};
      nodes->SetVelocity_Old(iPoint, zero);
    }

    for (auto iDim = 0u; iDim < nDim; iDim++)
      LinSysRes.SetBlock_Zero(iPoint, iDim+1);
    nodes->SetVel_ResTruncError_Zero(iPoint);

    /*--- Get transport coefficients ---*/

    su2double laminar_viscosity    = nodes->GetLaminarViscosity(iPoint);
    su2double eddy_viscosity       = nodes->GetEddyViscosity(iPoint);
    su2double thermal_conductivity = Cp * (laminar_viscosity/Prandtl_Lam + eddy_viscosity/Prandtl_Turb);

    // work in progress on real-gases...
    //thermal_conductivity = nodes->GetThermalConductivity(iPoint);
    //Cp = nodes->GetSpecificHeatCp(iPoint);
    //thermal_conductivity += Cp*eddy_viscosity/Prandtl_Turb;

    /*--- If it is a customizable or CHT patch, retrieve the specified wall temperature. ---*/

    const su2double There = nodes->GetTemperature(Point_Normal);

    if (cht_mode) {
      Twall = GetCHTWallTemperature(config, val_marker, iVertex, dist_ij,
                                    thermal_conductivity, There, Temperature_Ref);
    }
    else if (config->GetMarker_All_PyCustom(val_marker)) {
      Twall = geometry->GetCustomBoundaryTemperature(val_marker, iVertex);
    }

    /*--- Compute the normal gradient in temperature using Twall ---*/

    su2double dTdn = -(There - Twall)/dist_ij;

    /*--- Apply a weak boundary condition for the energy equation.
     Compute the residual due to the prescribed heat flux. ---*/

    su2double Res_Conv = 0.0;
    su2double Res_Visc = thermal_conductivity * dTdn * Area;

    /*--- Calculate Jacobian for implicit time stepping ---*/

    if (implicit) {

      /*--- Add contributions to the Jacobian from the weak enforcement of the energy equations. ---*/

      su2double Density = nodes->GetDensity(iPoint);
      su2double Vel2 = GeometryToolbox::SquaredNorm(nDim, &nodes->GetPrimitive(iPoint)[1]);
      su2double dTdrho = 1.0/Density * ( -Twall + (Gamma-1.0)/Gas_Constant*(Vel2/2.0) );

      Jacobian_i[nDim+1][0] = thermal_conductivity/dist_ij * dTdrho * Area;

      for (auto jDim = 0u; jDim < nDim; jDim++)
        Jacobian_i[nDim+1][jDim+1] = 0.0;

      Jacobian_i[nDim+1][nDim+1] = thermal_conductivity/dist_ij * (Gamma-1.0)/(Gas_Constant*Density) * Area;
    }

    /*--- If the wall is moving, there are additional residual contributions
     due to pressure (p v_wall.n) and shear stress (tau.v_wall.n). ---*/

    if (dynamic_grid) {
      AddDynamicGridResidualContribution(iPoint, Point_Normal, geometry, UnitNormal,
                                         Area, geometry->nodes->GetGridVel(iPoint),
                                         Jacobian_i, Res_Conv, Res_Visc);
    }

    /*--- Convective and viscous contributions to the residual at the wall ---*/

    LinSysRes(iPoint, nDim+1) += Res_Conv - Res_Visc;

    /*--- Enforce the no-slip boundary condition in a strong way by
     modifying the velocity-rows of the Jacobian (1 on the diagonal).
     And add the contributions to the Jacobian due to energy. ---*/

    if (implicit) {
      Jacobian.AddBlock2Diag(iPoint, Jacobian_i);

      for (auto iVar = 1u; iVar <= nDim; iVar++) {
        auto total_index = iPoint*nVar+iVar;
        Jacobian.DeleteValsRowi(total_index);
      }
    }
  }

  if (Jacobian_i)
    for (auto iVar = 0u; iVar < nVar; iVar++)
      delete [] Jacobian_i[iVar];
  delete [] Jacobian_i;

}

void CNSSolver::BC_Isothermal_Wall(CGeometry *geometry, CSolver **solver_container, CNumerics *conv_numerics,
                                   CNumerics *visc_numerics, CConfig *config, unsigned short val_marker) {

  BC_Isothermal_Wall_Generic(geometry, solver_container, conv_numerics, visc_numerics, config, val_marker);
}

void CNSSolver::BC_ConjugateHeat_Interface(CGeometry *geometry, CSolver **solver_container, CNumerics *conv_numerics,
                                           CConfig *config, unsigned short val_marker) {

  BC_Isothermal_Wall_Generic(geometry, solver_container, conv_numerics, nullptr, config, val_marker, true);
}

void CNSSolver::SetTauWall_WF(CGeometry *geometry, CSolver **solver_container, CConfig *config) {

  const su2double Gas_Constant = config->GetGas_ConstantND();
  const su2double Cp = (Gamma / Gamma_Minus_One) * Gas_Constant;

  constexpr unsigned short max_iter = 10;
  const su2double tol = 1e-6;
  const su2double relax = 0.25;

  /*--- Compute the recovery factor ---*/
  // Double-check: laminar or turbulent Pr for this?
  const su2double Recovery = pow(config->GetPrandtl_Lam(), (1.0/3.0));

  /*--- Typical constants from boundary layer theory ---*/

  const su2double kappa = 0.4;
  const su2double B = 5.5;

  for (auto iMarker = 0u; iMarker < config->GetnMarker_All(); iMarker++) {

    if (!config->GetViscous_Wall(iMarker)) continue;

    /*--- Identify the boundary by string name ---*/

    const auto Marker_Tag = config->GetMarker_All_TagBound(iMarker);

    /*--- Get the specified wall heat flux from config ---*/

    // Wall_HeatFlux = config->GetWall_HeatFlux(Marker_Tag);

    /*--- Loop over all of the vertices on this boundary marker ---*/

    SU2_OMP_FOR_DYN(OMP_MIN_SIZE)
    for (auto iVertex = 0u; iVertex < geometry->nVertex[iMarker]; iVertex++) {

      const auto iPoint = geometry->vertex[iMarker][iVertex]->GetNode();
      const auto Point_Normal = geometry->vertex[iMarker][iVertex]->GetNormal_Neighbor();

      /*--- Check if the node belongs to the domain (i.e, not a halo node)
       and the neighbor is not part of the physical boundary ---*/

      if (!geometry->nodes->GetDomain(iPoint)) continue;

      /*--- Get coordinates of the current vertex and nearest normal point ---*/

      const auto Coord = geometry->nodes->GetCoord(iPoint);
      const auto Coord_Normal = geometry->nodes->GetCoord(Point_Normal);

      /*--- Compute dual-grid area and boundary normal ---*/

      const auto Normal = geometry->vertex[iMarker][iVertex]->GetNormal();

      su2double Area = GeometryToolbox::Norm(nDim, Normal);

      su2double UnitNormal[MAXNDIM] = {0.0};
      for (auto iDim = 0u; iDim < nDim; iDim++)
        UnitNormal[iDim] = -Normal[iDim]/Area;

      /*--- Get the velocity, pressure, and temperature at the nearest
       (normal) interior point. ---*/

      su2double Vel[MAXNDIM] = {0.0};
      for (auto iDim = 0u; iDim < nDim; iDim++)
        Vel[iDim] = nodes->GetVelocity(Point_Normal,iDim);
      su2double P_Normal = nodes->GetPressure(Point_Normal);
      su2double T_Normal = nodes->GetTemperature(Point_Normal);

      /*--- Compute the wall-parallel velocity at first point off the wall ---*/

      su2double VelNormal = GeometryToolbox::DotProduct(nDim, Vel, UnitNormal);

      su2double VelTang[MAXNDIM] = {0.0};
      for (auto iDim = 0u; iDim < nDim; iDim++)
        VelTang[iDim] = Vel[iDim] - VelNormal*UnitNormal[iDim];

      su2double VelTangMod = GeometryToolbox::Norm(int(MAXNDIM), VelTang);

      /*--- Compute normal distance of the interior point from the wall ---*/

      su2double WallDist[MAXNDIM] = {0.0};
      GeometryToolbox::Distance(nDim, Coord, Coord_Normal, WallDist);

      su2double WallDistMod = GeometryToolbox::Norm(int(MAXNDIM), WallDist);

      /*--- Compute mach number ---*/

      // M_Normal = VelTangMod / sqrt(Gamma * Gas_Constant * T_Normal);

      /*--- Compute the wall temperature using the Crocco-Buseman equation ---*/

      //T_Wall = T_Normal * (1.0 + 0.5*Gamma_Minus_One*Recovery*M_Normal*M_Normal);
      su2double T_Wall = T_Normal + Recovery*pow(VelTangMod,2.0)/(2.0*Cp);

      /*--- Extrapolate the pressure from the interior & compute the
       wall density using the equation of state ---*/

      su2double P_Wall = P_Normal;
      su2double Density_Wall = P_Wall/(Gas_Constant*T_Wall);

      /*--- Compute the shear stress at the wall in the regular fashion
       by using the stress tensor on the surface ---*/

      su2double Lam_Visc_Wall = nodes->GetLaminarViscosity(iPoint);

      const auto GradVel = &nodes->GetGradient_Primitive(iPoint)[1];

      su2double div_vel = 0.0;
      for (auto iDim = 0u; iDim < nDim; iDim++)
        div_vel += GradVel[iDim][iDim];

      su2double tau[MAXNDIM][MAXNDIM] = {{0.0}}, TauElem[MAXNDIM] = {0.0};
      for (auto iDim = 0u; iDim < nDim; iDim++) {
        for (auto jDim = 0u; jDim < nDim; jDim++) {
          tau[iDim][jDim] = Lam_Visc_Wall * (GradVel[jDim][iDim] + GradVel[iDim][jDim]);
        }
        tau[iDim][iDim] -= TWO3*Lam_Visc_Wall*div_vel;

        TauElem[iDim] = GeometryToolbox::DotProduct(nDim, tau[iDim], UnitNormal);
      }

      /*--- Compute wall shear stress as the magnitude of the wall-tangential
       component of the shear stress tensor---*/

      su2double TauNormal = GeometryToolbox::DotProduct(nDim, TauElem, UnitNormal);

      su2double TauTangent[MAXNDIM] = {0.0};
      for (auto iDim = 0u; iDim < nDim; iDim++)
        TauTangent[iDim] = TauElem[iDim] - TauNormal * UnitNormal[iDim];

      su2double Tau_Wall = GeometryToolbox::Norm(int(MAXNDIM), TauTangent);

      /*--- Calculate the quantities from boundary layer theory and
       iteratively solve for a new wall shear stress. Use the current wall
       shear stress as a starting guess for the wall function. ---*/

      su2double Tau_Wall_Old = Tau_Wall;
      unsigned short counter = 0;
      su2double diff = 1.0;

      while (diff > tol) {

        /*--- Friction velocity and u+ ---*/

        su2double U_Tau = sqrt(Tau_Wall_Old/Density_Wall);
        su2double U_Plus = VelTangMod/U_Tau;

        /*--- Gamma, Beta, Q, and Phi, defined by Nichols & Nelson (2004) ---*/

        su2double Gam  = Recovery*pow(U_Tau,2)/(2.0*Cp*T_Wall);
        su2double Beta = 0.0; // For adiabatic flows only
        su2double Q    = sqrt(Beta*Beta + 4.0*Gam);
        su2double Phi  = asin(-1.0*Beta/Q);

        /*--- Y+ defined by White & Christoph (compressibility and heat transfer)
         negative value for (2.0*Gam*U_Plus - Beta)/Q ---*/

        su2double Y_Plus_White = exp((kappa/sqrt(Gam))*(asin((2.0*Gam*U_Plus - Beta)/Q) - Phi))*exp(-1.0*kappa*B);

        /*--- Spalding's universal form for the BL velocity with the
         outer velocity form of White & Christoph above. ---*/

        su2double kUp = kappa*U_Plus;
        su2double Y_Plus = U_Plus + Y_Plus_White - exp(-1.0*kappa*B) * (1.0 + kUp*(1.0 + 0.5*kUp + pow(kUp,2)/6.0));

        /*--- Calculate an updated value for the wall shear stress using the y+ value,
         the definition of y+, and the definition of the friction velocity. ---*/

        Tau_Wall = (1.0/Density_Wall)*pow(Y_Plus*Lam_Visc_Wall/WallDistMod,2.0);

        /*--- Difference between the old and new Tau. Update old value. ---*/

        diff = fabs(Tau_Wall-Tau_Wall_Old);
        Tau_Wall_Old += relax * (Tau_Wall-Tau_Wall_Old);

        counter++;
        if (counter > max_iter) {
          cout << "WARNING: Tau_Wall evaluation has not converged in CNSSolver.cpp" << endl;
          cout << Tau_Wall_Old << " " << Tau_Wall << " " << diff << endl;
          break;
        }
      }

      /*--- Store this value for the wall shear stress at the node.  ---*/

      nodes->SetTauWall(iPoint, Tau_Wall);

    }

  }

}

su2double CNSSolver::Compute_ViscCD_StokesMethod(CGeometry *geometry, CConfig *config) {

	unsigned long iPoin, jPoin, kPoin, nPoin_x, nPoin_y, nPoin_z, local_index;
	unsigned long nPointLocal, nPointGlobal;
	su2double nu;
	su2double ***wm_plus, ***y_plus, ***x_pos, ***local_wm_plus, ***local_y_plus, ***local_x_pos;
	su2double ***tke_approx, ***local_tke_approx, ***u_streamwise, ***local_u_streamwise, ***nu_turb, ***local_nu_turb;

	su2double Grad_Vel[3][3] = {{0.0, 0.0, 0.0},{0.0, 0.0, 0.0},{0.0, 0.0, 0.0}}, div_vel,
	delta_ij[3][3] = {{1.0, 0.0, 0.0},{0.0,1.0,0.0},{0.0,0.0,1.0}},
	Sij_star[3][3] = {{0.0, 0.0, 0.0},{0.0, 0.0, 0.0},{0.0, 0.0, 0.0}};
	unsigned short iDim, jDim;

	nPointLocal = nPointDomain;
#ifdef HAVE_MPI
  SU2_MPI::Allreduce(&nPointLocal, &nPointGlobal, 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);
#else
  nPointGlobal = nPointLocal; //Total number of points in the domain (no halo nodes considered)
#endif

	nPoin_x = config->Get_nPoinx_samplingLines();
	nPoin_y = config->Get_nPoiny_samplingLines() - 1; //exclude mesh points at the wall
	nPoin_z = config->Get_nPoinz_samplingLines();

	local_wm_plus = new su2double **[nPoin_x];
	local_y_plus = new su2double **[nPoin_x];
	local_x_pos = new su2double **[nPoin_x];
	local_tke_approx = new su2double **[nPoin_x];
	local_u_streamwise = new su2double **[nPoin_x];
	local_nu_turb = new su2double **[nPoin_x];
	wm_plus = new su2double **[nPoin_x];
	y_plus = new su2double **[nPoin_x];
	x_pos = new su2double **[nPoin_x];
	tke_approx = new su2double **[nPoin_x];
	u_streamwise = new su2double **[nPoin_x];
	nu_turb = new su2double **[nPoin_x];

	su2double **local_u_tau, **u_tau;
	local_u_tau = new su2double *[nPoin_x];
	u_tau = new su2double *[nPoin_x];
	unsigned long local_index_b;

	/*--- Compute the spanwise velocity (wm), wall distance (y_plus), the friction velocity ---*/
	for (iPoin = 0; iPoin < nPoin_x; ++iPoin){

		local_u_tau[iPoin] = new su2double [nPoin_z];
		local_wm_plus[iPoin] = new su2double *[nPoin_y];
		local_y_plus[iPoin] = new su2double *[nPoin_y];
		local_x_pos[iPoin] = new su2double *[nPoin_y];
		local_tke_approx[iPoin] = new su2double *[nPoin_y];
		local_u_streamwise[iPoin] = new su2double *[nPoin_y];
		local_nu_turb[iPoin] = new su2double *[nPoin_y];
		u_tau[iPoin] = new su2double [nPoin_z];
		wm_plus[iPoin] = new su2double *[nPoin_y];
		y_plus[iPoin] = new su2double *[nPoin_y];
		x_pos[iPoin] = new su2double *[nPoin_y];
		tke_approx[iPoin] = new su2double *[nPoin_y];
		u_streamwise[iPoin] = new su2double *[nPoin_y];
		nu_turb[iPoin] = new su2double *[nPoin_y];

		for (jPoin = 0; jPoin < nPoin_y; ++jPoin){

			local_wm_plus[iPoin][jPoin] = new su2double [nPoin_z];
			local_y_plus[iPoin][jPoin] = new su2double [nPoin_z];
			local_x_pos[iPoin][jPoin] = new su2double [nPoin_z];
			local_tke_approx[iPoin][jPoin] = new su2double [nPoin_z];
			local_u_streamwise[iPoin][jPoin] = new su2double [nPoin_z];
			local_nu_turb[iPoin][jPoin] = new su2double [nPoin_z];
			wm_plus[iPoin][jPoin] = new su2double [nPoin_z];
			y_plus[iPoin][jPoin] = new su2double [nPoin_z];
			x_pos[iPoin][jPoin] = new su2double [nPoin_z];
			tke_approx[iPoin][jPoin] = new su2double [nPoin_z];
			u_streamwise[iPoin][jPoin] = new su2double [nPoin_z];
			nu_turb[iPoin][jPoin] = new su2double [nPoin_z];

			for (kPoin = 0; kPoin < nPoin_z; ++kPoin){
				local_index = config->Get_samplingLines(iPoin, jPoin, kPoin);
				local_index_b = config->Get_samplingLines(iPoin, nPoin_y, kPoin); //points at the wall are stored at index nPoin_y

				if (local_index == nPointGlobal+1){ //in CTurbSolver::ReadSamplingLines() local_index is set to nPointGlobal+1 if it does not belong to the subdomain.
					local_wm_plus[iPoin][jPoin][kPoin] = 0;
					local_y_plus[iPoin][jPoin][kPoin] = 0;
					local_x_pos[iPoin][jPoin][kPoin] = 0;
					local_tke_approx[iPoin][jPoin][kPoin] = 0; // tke approx inspired by SA QCR (2013) model : https://turbmodels.larc.nasa.gov/spalart.html
					local_u_streamwise[iPoin][jPoin][kPoin] = 0;
					local_nu_turb[iPoin][jPoin][kPoin] = 0;
					local_u_tau[iPoin][kPoin] = GetFrictionVel(local_index_b);
				}
				else{
					nu 	= nodes->GetLaminarViscosity(local_index) / nodes->GetDensity(local_index);
					local_wm_plus[iPoin][jPoin][kPoin] = nodes->GetVelocity(local_index, 2); // spanwise velocity (w)
					local_y_plus[iPoin][jPoin][kPoin] = geometry->nodes->GetWall_Distance(local_index) / nu; // wall distance same formula as in CNSsolver::Firction_Forces()
					local_x_pos[iPoin][jPoin][kPoin] = geometry->nodes->GetCoord(local_index, 0);	// streamwise coordinate (x)
					local_u_tau[iPoin][kPoin] = GetFrictionVel(local_index_b);
					local_u_streamwise[iPoin][jPoin][kPoin] = nodes->GetVelocity(local_index, 0);
					local_nu_turb[iPoin][jPoin][kPoin] = nu;

					/*--- Compute gradient of velocity, divergence and strain rate in order to approximate t.k.e. ---*/
				    for (iDim = 0; iDim < nDim; iDim++) {
				      for (jDim = 0 ; jDim < nDim; jDim++) {
				        Grad_Vel[iDim][jDim] = nodes->GetGradient_Primitive(local_index, iDim+1, jDim);
				      }
				    }

				    div_vel = 0.0; for (iDim = 0; iDim < nDim; iDim++) div_vel += Grad_Vel[iDim][iDim];

			        for (iDim = 0; iDim < nDim; iDim++) {
			          for (jDim = 0 ; jDim < nDim; jDim++) {
			        	  Sij_star[iDim][jDim] = (Grad_Vel[jDim][iDim] + Grad_Vel[iDim][jDim])/2.0 - 1.0/3.0*div_vel*delta_ij[iDim][jDim];
			          }
			        }

					local_tke_approx[iPoin][jPoin][kPoin] = nodes->GetEddyViscosity(local_index)/nodes->GetDensity(local_index) *
							                                sqrt(2.0 * (Sij_star[0][0]*Sij_star[0][0] + Sij_star[1][1]*Sij_star[1][1] + Sij_star[2][2]*Sij_star[2][2]) );

				}
			}
		}
	}

/*--- Populate all the matrices by sharing info among the cores ---*/

#ifdef HAVE_MPI

	for (iPoin = 0; iPoin < nPoin_x; ++iPoin){
		SU2_MPI::Allreduce(local_u_tau[iPoin], u_tau[iPoin], nPoin_z, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		for (jPoin = 0; jPoin < nPoin_y; ++jPoin){
			SU2_MPI::Allreduce(local_wm_plus[iPoin][jPoin], wm_plus[iPoin][jPoin], nPoin_z, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
			SU2_MPI::Allreduce(local_y_plus[iPoin][jPoin], y_plus[iPoin][jPoin], nPoin_z, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
			SU2_MPI::Allreduce(local_x_pos[iPoin][jPoin], x_pos[iPoin][jPoin], nPoin_z, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
			SU2_MPI::Allreduce(local_tke_approx[iPoin][jPoin], tke_approx[iPoin][jPoin], nPoin_z, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
			SU2_MPI::Allreduce(local_u_streamwise[iPoin][jPoin], u_streamwise[iPoin][jPoin], nPoin_z, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
			SU2_MPI::Allreduce(local_nu_turb[iPoin][jPoin], nu_turb[iPoin][jPoin], nPoin_z, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		}
	}

#else
	&wm_plus= &local_wm_plus;
	&y_plus = &local_y_plus;
	&x_pos 	= &local_x_pos;
	&u_tau 	= &local_u_tau;
	&tke_approx	= &local_tke_approx;
	&u_streamwise	= &local_u_streamwise;
	&nu_turb= &local_nu_turb;
#endif

	/* --- Divide by u_tau: wm_plus = wm / u_tau, y_plus = wall_dist * u_tau / nu ---*/
	for (iPoin = 0; iPoin < nPoin_x; ++iPoin){
		for (jPoin = 0; jPoin < nPoin_y; ++jPoin){
			for (kPoin = 0; kPoin < nPoin_z; ++kPoin){
				y_plus[iPoin][jPoin][kPoin] *= u_tau[iPoin][kPoin]; // wall distance
				wm_plus[iPoin][jPoin][kPoin] /= u_tau[iPoin][kPoin]; // spanwise velocity (w)
			}
		}
	}

	/*--- Fit the exponential to compute the 'equivalent spanwise velocity at the wall' (Wm_plus);
	 * the relationship is wm+ = exp(Wm+, -A*y+), where Wm+ and A need to be fitted. ---*/

	su2double **Wm_plus; // initialize Matrix of equivalent spanwise velocity (i.e. a wave) at the wall
	su2double **x_at_wall; // streamwise coordinates at the wall along a slice
	su2double **B;
	unsigned long ***peaks, ***peaks_tke;

	Wm_plus = new su2double*[nPoin_x];				//each wave is evaluated along the streamwise direction (x-coord)
	x_at_wall = new su2double*[nPoin_x];
	B = new su2double*[nPoin_x];
	peaks = new unsigned long**[nPoin_x];
	peaks_tke = new unsigned long**[nPoin_x];
	for (iPoin = 0; iPoin < nPoin_x; iPoin++){
		Wm_plus[iPoin] = new su2double[nPoin_z];	// one wave for every spanwise (z_coord) slice
		x_at_wall[iPoin] = new su2double[nPoin_z];
		B[iPoin] = new su2double[nPoin_z];
		peaks[iPoin] = new unsigned long*[3]; 		// one element to store whether it's a peak or a through; one to store the index; one to store change of sign index;
		peaks_tke[iPoin] = new unsigned long*[3];
		for (jPoin = 0; jPoin < 3; jPoin++){
			peaks[iPoin][jPoin] = new unsigned long[nPoin_z];
			peaks_tke[iPoin][jPoin] = new unsigned long[nPoin_z];
		}
	}

	/*--- Find tke inflection point closest to wall ---*/
	Find_peak_closest_to_wall(tke_approx, nPoin_x, nPoin_y, nPoin_z, 1e-6, peaks_tke);

	/*--- Find inflection point closest to wall ---*/
	Find_peak_closest_to_wall(wm_plus, nPoin_x, nPoin_y, nPoin_z, 1e-6, peaks);

	/*--- Find index of closest point to inflection point where wm_plus changes sign ---*/
	Find_change_of_sign(wm_plus, nPoin_x, nPoin_y, nPoin_z, peaks);

//	//FOR VALIDATION ONLY
//	kPoin = 15;
//
//	ofstream myfile1;
//
//	/*--- Uncomment if need to debug ---*/
//	if (rank == MASTER_NODE){
//		myfile1.open ("1_validation_fit_exponential/peaksP.dat", ios::out);
//		for (iPoin = 0; iPoin<nPoin_x-1; iPoin++){
//			myfile1 << peaks[iPoin][1][kPoin] << ", ";
//		}
//		myfile1 << peaks[nPoin_x-1][1][kPoin] << endl;
//		myfile1.close();
//	}
//
//	/*--- Uncomment if need to debug ---*/
//	if (rank == MASTER_NODE){
//		myfile1.open ("1_validation_fit_exponential/sign_changeP.dat", ios::out);
//		for (iPoin = 0; iPoin<nPoin_x-1; iPoin++){
//			myfile1 << peaks[iPoin][2][kPoin] << ", " ;
//		}
//		myfile1 << peaks[nPoin_x-1][2][kPoin] << endl;
//		myfile1.close();
//	}
//
//	if (rank == MASTER_NODE){
//		myfile1.open ("1_validation_fit_exponential/y_plusP.dat", ios::out);
//		for (iPoin=0; iPoin < nPoin_x; iPoin++){
//			for (jPoin=0; jPoin < nPoin_y-1; jPoin++){
//				myfile1 << y_plus[iPoin][jPoin][kPoin] << ", ";
//			}
//			myfile1 << y_plus[iPoin][nPoin_y-1][kPoin] << endl;
//		}
//		myfile1.close();
//	}
//
//	if (rank == MASTER_NODE){
//		myfile1.open ("1_validation_fit_exponential/xpos.dat", ios::out);
//		for (iPoin=0; iPoin < nPoin_x; iPoin++){
//			for (jPoin=0; jPoin < nPoin_y-1; jPoin++){
//				myfile1 << x_pos[iPoin][jPoin][kPoin] << ", ";
//			}
//			myfile1 << x_pos[iPoin][nPoin_y-1][kPoin] << endl;
//		}
//		myfile1.close();
//	}
//
//	if (rank == MASTER_NODE){
//		myfile1.open ("1_validation_fit_exponential/wm_plusP.dat", ios::out);
//		for (iPoin=0; iPoin < nPoin_x; iPoin++){
//			for (jPoin=0; jPoin < nPoin_y-1; jPoin++){
//				myfile1 << wm_plus[iPoin][jPoin][kPoin] << ", ";
//			}
//			myfile1 << wm_plus[iPoin][nPoin_y-1][kPoin] << endl;
//		}
//		myfile1.close();
//	}
//
//	if (rank == MASTER_NODE){
//		myfile1.open ("6_tke_approx_validation/tke.dat", ios::out);
//		for (iPoin=0; iPoin < nPoin_x; iPoin++){
//			for (jPoin=0; jPoin < nPoin_y-1; jPoin++){
//				myfile1 << tke_approx[iPoin][jPoin][kPoin] << ", ";
//			}
//			myfile1 << tke_approx[iPoin][nPoin_y-1][kPoin] << endl;
//		}
//		myfile1.close();
//	}
//
//	if (rank == MASTER_NODE){
//		myfile1.open ("6_tke_approx_validation/y_plusP.dat", ios::out);
//		for (iPoin=0; iPoin < nPoin_x; iPoin++){
//			for (jPoin=0; jPoin < nPoin_y-1; jPoin++){
//				myfile1 << y_plus[iPoin][jPoin][kPoin] << ", ";
//			}
//			myfile1 << y_plus[iPoin][nPoin_y-1][kPoin] << endl;
//		}
//		myfile1.close();
//	}
//
//	/*--- Uncomment if need to debug ---*/
//	if (rank == MASTER_NODE){
//		myfile1.open ("6_tke_approx_validation/peaks_tke.dat", ios::out);
//		for (iPoin = 0; iPoin<nPoin_x-1; iPoin++){
//			myfile1 << peaks_tke[iPoin][1][kPoin] << ", " ;
//		}
//		myfile1 << peaks_tke[nPoin_x-1][1][kPoin] << endl;
//		myfile1.close();
//	}
//
//  //END VALIDATION

	/*--- Compute equivalent spanwise oscialltion at the wall ---*/
	Fit_exponential(wm_plus, y_plus, x_pos, nPoin_x, nPoin_y, nPoin_z, Wm_plus, x_at_wall, B, peaks);

//	//FOR VALIDATION ONLY: For each slice, for each x, plot the actual wm_plus and the fitted exponential. Visually check they make sense.
//
//	for (iPoin=0; iPoin < nPoin_x; iPoin++){
//		cout << "(" << iPoin << "): Aint = " << Wm_plus[iPoin][kPoin] << " , Bint = " << B[iPoin][kPoin] << endl;;
//	}
//	cout << endl;
//	//END VALIDATION

	/*--- Compute Fourier transform of Wm+ for each slice ---*/

	// Transform distance into time //
	su2double **vel_peak, **nu_peak;

	vel_peak = new su2double *[nPoin_x];
	nu_peak = new su2double *[nPoin_x];
	for (iPoin = 0; iPoin < nPoin_x; ++iPoin){
		vel_peak[iPoin] = new su2double [nPoin_z];
		nu_peak[iPoin] = new su2double [nPoin_z];
		for (kPoin = 0; kPoin < nPoin_z; ++kPoin){
			vel_peak[iPoin][kPoin] = 0.0;
			nu_peak[iPoin][kPoin] = 0.0;
		}
	}

	/*--- Retrieve streamwise velocity and laminar viscosity in correspondence of tke peaks ---*/
	for (iPoin = 0; iPoin < nPoin_x; ++iPoin){
		for (kPoin = 0; kPoin < nPoin_z; ++kPoin){
			jPoin = peaks_tke[iPoin][1][kPoin];
			vel_peak[iPoin][kPoin] = u_streamwise[iPoin][jPoin][kPoin];
			nu_peak[iPoin][kPoin] = nu_turb[iPoin][jPoin][kPoin];
		}
	}


	su2double avg_vel, avg_utau, avg_nu;
	su2double min_t, max_t, freq_spacing;
	su2double *time, *wm_plus_for_fft;
	time = new su2double[nPoin_x];
	wm_plus_for_fft = new su2double[nPoin_x];

	unsigned long count, n, fft_npoin;
	count = 0; n = nPoin_x;
	while( n != 0){
		n >>= 1;
		count += 1;
	}
	fft_npoin = pow(2, count);

	su2double *tmp_time, *tmp_wmplus, **fft_freq, **fft_period, **fft_wmplus, h, val;
	su2double freq;
	tmp_time = new su2double[fft_npoin];
	tmp_wmplus = new su2double[fft_npoin];
	fft_freq = new su2double*[fft_npoin/2]; // fft_npoin/2 only when analyzing a real-valued signal
	fft_period = new su2double*[fft_npoin/2];
	fft_wmplus = new su2double*[fft_npoin/2];
	for (iPoin=0; iPoin < fft_npoin/2; iPoin++) {
		fft_freq[iPoin] = new su2double[nPoin_z];
		fft_period[iPoin] = new su2double[nPoin_z];
		fft_wmplus[iPoin] = new su2double[nPoin_z];
	}

	su2double *xx, *yy, **zz, *xint;
	su2double **tmp_R, **weight;
	su2double R, sum_tmp_R=0, sum_weight=0;

	tmp_R = new su2double* [fft_npoin/2];
	weight = new su2double* [fft_npoin/2];
	xint = new su2double [2];
	for (iPoin=0; iPoin < fft_npoin/2; iPoin++) {
		tmp_R[iPoin] = new su2double[nPoin_z];
		weight[iPoin] = new su2double[nPoin_z];
	}

	// Allocate Ricco data
	xx = new su2double [config->Get_nPoinx_Ricco()];
	yy = new su2double [config->Get_nPoiny_Ricco()];
	zz = new su2double* [config->Get_nPoinx_Ricco()];

	for (iPoin = 0; iPoin < config->Get_nPoinx_Ricco(); iPoin++){
		xx[iPoin] = config->Get_RiccoField(0, iPoin, 0);			// only one row is necessary for bilinear interp on an *ordered grid* of points.
		zz[iPoin] = new su2double [config->Get_nPoiny_Ricco()];
		for (jPoin = 0; jPoin < config->Get_nPoiny_Ricco(); jPoin++){
			yy[jPoin] = config->Get_RiccoField(jPoin, 0, 1);	// only one column is necessary for bilinear interp on an *ordered grid* of points.
			zz[iPoin][jPoin] = config->Get_RiccoField(jPoin, iPoin, 2);
		}
	}

	su2double u_tau_mean = 0;

	/*--- Loop over each spanwise slice to compute FFT of the equivalent wall oscillations ---*/
	for (kPoin = 0; kPoin < nPoin_z; ++kPoin){

		avg_vel=0; avg_utau=0; avg_nu=0;

		for (iPoin = 0; iPoin < nPoin_x; ++iPoin){
			avg_vel  += vel_peak[iPoin][kPoin];
			avg_utau += u_tau[iPoin][kPoin];
			avg_nu   += nu_peak[iPoin][kPoin];
		}
		avg_vel /= nPoin_x;
		avg_utau /= nPoin_x;
		avg_nu /= nPoin_x;

		u_tau_mean += avg_utau;

		for (iPoin = 0; iPoin < nPoin_x; ++iPoin){
			time[iPoin] = x_at_wall[iPoin][kPoin] / avg_vel;
			wm_plus_for_fft[iPoin] = Wm_plus[iPoin][kPoin];
		}

		// Interpolate each Wm+ and T+ on n equally-spaced points, where n is an integer power of 2. //
		min_t = time[0];
		max_t = time[0];
		for (iPoin=0; iPoin < nPoin_x; iPoin++){
			if(time[iPoin] > max_t){ max_t = time[iPoin];} // If current element is greater than max
			if(time[iPoin] < min_t){ min_t = time[iPoin];} // If current element is smaller than min
		}


		h = (max_t - min_t) / (fft_npoin - 1); //spacing;

		for (iPoin=0, val = min_t; iPoin < fft_npoin; iPoin++, val += h) {
			tmp_time[iPoin] = val;
			tmp_wmplus[iPoin] = LinearInterp(time, nPoin_x, wm_plus_for_fft, tmp_time[iPoin]);
		}

//		//FOR VALIDATION ONLY: For each slice, plot the equivalent wall oscillation
//
//		ofstream myfile_fft;
//
//		/*--- Uncomment if need to debug ---*/
//		unsigned long kkk = 68;
//		if (rank == MASTER_NODE && kPoin == kkk){
//			myfile_fft.open ("7_fft_validation/wm_plus.dat", ios::out);
//			for (iPoin = 0; iPoin<fft_npoin-1; iPoin++){
//				myfile_fft << tmp_wmplus[iPoin]<< ", ";
//			}
//			myfile_fft << tmp_wmplus[fft_npoin-1] << endl;
//			myfile_fft.close();
//		}
//
//		if (rank == MASTER_NODE && kPoin == kkk){
//			myfile_fft.open ("7_fft_validation/time.dat", ios::out);
//			for (iPoin = 0; iPoin<fft_npoin-1; iPoin++){
//				myfile_fft << tmp_time[iPoin]<< ", ";
//			}
//			myfile_fft << tmp_time[fft_npoin-1] << endl;
//			myfile_fft.close();
//		}
//		//END VALIDATION

		//Compute the FFT of the data
		realft(tmp_wmplus, 1.0, fft_npoin); 	// 1.0: indicates we want to map from the time to the frequency domain;

		freq_spacing = 1.0/( h * (fft_npoin-1) );
		for (iPoin=0, freq=0; iPoin < fft_npoin/2; iPoin++, freq += freq_spacing) {
			fft_freq[iPoin][kPoin] = freq;
			if (iPoin == 0){
				fft_period[iPoin][kPoin] = 0;
			}
			else{
				fft_period[iPoin][kPoin] = 1.0 / fft_freq[iPoin][kPoin];
				fft_period[iPoin][kPoin] *= (avg_utau*avg_utau) / avg_nu; 		 	// Normalize period;
			}
			fft_wmplus[iPoin][kPoin] = sqrt(tmp_wmplus[2*iPoin]*tmp_wmplus[2*iPoin] +
					                   tmp_wmplus[2*iPoin+1]*tmp_wmplus[2*iPoin+1]); //compute fft magnitude: sqrt(Re(z)**2+Im(z)**2)
			fft_wmplus[iPoin][kPoin] /= fft_npoin/2;							 	 // Normalize
		}

		/*--- Compute R-factor by interpolating from Gatti and Quadrio (2016) diagram (Wm+ vs. T+ vs. R) ---*/
		/*--- R represents the % friction drag reduction compared to a flat plate ---*/
		for (iPoin=0; iPoin < fft_npoin/2; iPoin++) {
			// if T+, Wm+ are within the interpolating domain and is not the zero-frequency case (iPoin=0)
			if (fft_wmplus[iPoin][kPoin] >= yy[0] &&
				fft_wmplus[iPoin][kPoin] <= yy[config->Get_nPoiny_Ricco()-1] &&
				fft_period[iPoin][kPoin] >= xx[0] &&
				fft_period[iPoin][kPoin] <= xx[config->Get_nPoinx_Ricco()-1] &&
				iPoin != 0){

				xint[0] = fft_period[iPoin][kPoin];
				xint[1] = fft_wmplus[iPoin][kPoin];
				tmp_R[iPoin][kPoin] = BilinearInterp(xx, config->Get_nPoinx_Ricco(), yy, config->Get_nPoiny_Ricco(), zz, xint);
				weight[iPoin][kPoin] = fft_wmplus[iPoin][kPoin];
			}
			else{
				tmp_R[iPoin][kPoin] = 0;
				weight[iPoin][kPoin] = 0;
			}
			sum_tmp_R += weight[iPoin][kPoin]*tmp_R[iPoin][kPoin];
			sum_weight += weight[iPoin][kPoin];
//			cout << "i,k = " << iPoin << ", " << kPoin << ": T+ = " << fft_period[iPoin][kPoin] << ", Wm+ = " << fft_wmplus[iPoin][kPoin]
//				 <<	", R = " << tmp_R[iPoin][kPoin] << ", w = " <<  weight[iPoin][kPoin] << endl;
		}
	}

	u_tau_mean /= nPoin_z;
//	cout << "u_tau_mean = " << u_tau_mean << endl; //Uncomment only if u_tau_mean is needed to specify Re_tau flat plate

	/*--- Compute average of R ---*/
	if (sum_weight == 0){
		R = sum_tmp_R / (sum_weight + 1e-15); //add small number
	}
	else {
		R = sum_tmp_R / sum_weight;
	}

	su2double Re_tau_flat_plate = config->Get_Stokes_Re_tau_flat_plate();
	su2double flat_plate_viscous_drag_coeff = config->Get_Stokes_flat_plate_viscous_drag_coeff();

	if (rank == MASTER_NODE && config->GetInnerIter() % config->GetScreen_Wrt_Freq(2) == 0){
		cout << config->GetInnerIter() << " R = " << R << endl;
	}

	R = ReynoldsScalingRicco(R, Re_tau_flat_plate); // result is in percentage (%)

//	if (rank == MASTER_NODE){
//		cout << "R2 = " << R << endl;
//	}


//	//FOR VALIDATION ONLY: For each slice, plot the FFT of the equivalent wall oscillation
//	/*--- Uncomment if need to debug ---*/
//	ofstream myfile_fft;
//
//	if (rank == MASTER_NODE){
//		myfile_fft.open ("7_fft_validation/fft_wm_plus.dat", ios::out);
//		for (iPoin = 0; iPoin<fft_npoin/2-1; iPoin++){
//			for (kPoin = 0; kPoin<nPoin_z; kPoin++){
//				myfile_fft << fft_wmplus[iPoin][kPoin]<< ", ";
//			}
//			myfile_fft << fft_wmplus[fft_npoin/2-1][kPoin] << endl;
//		}
//		myfile_fft.close();
//	}
//
//	if (rank == MASTER_NODE){
//		myfile_fft.open ("7_fft_validation/fft_freq.dat", ios::out);
//		for (iPoin = 0; iPoin<fft_npoin/2-1; iPoin++){
//			for (kPoin = 0; kPoin<nPoin_z; kPoin++){
//				myfile_fft << fft_freq[iPoin][kPoin]<< ", ";
//			}
//			myfile_fft << fft_freq[fft_npoin/2-1][kPoin] << endl;
//		}
//		myfile_fft.close();
//	}
//
//	if (rank == MASTER_NODE){
//		myfile_fft.open ("7_fft_validation/fft_period.dat", ios::out);
//		for (iPoin = 0; iPoin<fft_npoin/2-1; iPoin++){
//			for (kPoin = 0; kPoin<nPoin_z; kPoin++){
//				myfile_fft << fft_period[iPoin][kPoin]<< ", ";
//			}
//			myfile_fft << fft_period[fft_npoin/2-1][kPoin] << endl;
//		}
//		myfile_fft.close();
//	}
//	//END VALIDATION

	/*--- Deallocate memory ---*/
	for (iPoin = 0; iPoin < nPoin_x; iPoin++) {
		for (jPoin = 0; jPoin < nPoin_y; jPoin++){
			delete [] local_wm_plus[iPoin][jPoin];
			delete [] wm_plus[iPoin][jPoin];
			delete [] local_y_plus[iPoin][jPoin];
			delete [] y_plus[iPoin][jPoin];
			delete [] local_x_pos[iPoin][jPoin];
			delete [] x_pos[iPoin][jPoin];
			delete [] local_tke_approx[iPoin][jPoin];
			delete [] tke_approx[iPoin][jPoin];
			delete [] local_u_streamwise[iPoin][jPoin];
			delete [] u_streamwise[iPoin][jPoin];
			delete [] local_nu_turb[iPoin][jPoin];
			delete [] nu_turb[iPoin][jPoin];
			if (jPoin < 3){
				delete [] peaks[iPoin][jPoin];
				delete [] peaks_tke[iPoin][jPoin];
			}
		  }
		delete [] local_wm_plus[iPoin];
		delete [] wm_plus[iPoin];
		delete [] local_y_plus[iPoin];
		delete [] y_plus[iPoin];
		delete [] local_x_pos[iPoin];
		delete [] x_pos[iPoin];
		delete [] local_tke_approx[iPoin];
		delete [] tke_approx[iPoin];
		delete [] local_u_streamwise[iPoin];
		delete [] u_streamwise[iPoin];
		delete [] local_nu_turb[iPoin];
		delete [] nu_turb[iPoin];
		delete [] local_u_tau[iPoin];
		delete [] u_tau[iPoin];
		delete [] peaks[iPoin];
		delete [] peaks_tke[iPoin];
		delete [] Wm_plus[iPoin];
		delete [] x_at_wall[iPoin];
		delete [] B[iPoin];
		delete [] vel_peak[iPoin];
		delete [] nu_peak[iPoin];
	  }
	delete [] local_wm_plus;
	delete [] wm_plus;
	delete [] local_y_plus;
	delete [] y_plus;
	delete [] local_x_pos;
	delete [] x_pos;
	delete [] local_tke_approx;
	delete [] tke_approx;
	delete [] local_u_streamwise;
	delete [] u_streamwise;
	delete [] local_nu_turb;
	delete [] nu_turb;
	delete [] local_u_tau;
	delete [] u_tau;
	delete [] peaks;
	delete [] peaks_tke;
	delete [] Wm_plus;
	delete [] x_at_wall;
	delete [] B;
	delete [] vel_peak;
	delete [] nu_peak;
	delete [] time;
	delete [] wm_plus_for_fft;

	for (iPoin = 0; iPoin < config->Get_nPoinx_Ricco(); iPoin++) {
		delete [] zz[iPoin];
	}
	delete [] xx;
	delete [] yy;
	delete [] zz;
	delete [] xint;

	for (iPoin=0; iPoin < fft_npoin/2; iPoin++) {
		delete [] fft_freq[iPoin];
		delete [] fft_period[iPoin];
		delete [] fft_wmplus[iPoin];
		delete [] tmp_R[iPoin];
		delete [] weight[iPoin];
	}
	delete [] fft_freq;
	delete [] fft_period;
	delete [] fft_wmplus;
	delete [] tmp_time;
	delete [] tmp_wmplus;
	delete [] tmp_R;
	delete [] weight;

	/*---Return viscous drag reduction---*/
	return (R / 100.0 ) * flat_plate_viscous_drag_coeff;  //viscous drag reduction w.r.t. flat plate.

}


//void CNSSolver::Fit_exponential(su2double ***wm_plus, su2double ***y_plus, su2double ***x_pos, unsigned long nPoin_x, unsigned long nPoin_y,
//		                        unsigned long nPoin_z, su2double **Wm_plus, su2double **x_at_wall, su2double **B) {
//
//	/*--- Main code from "Numerical Recipes: The Art of Scientific Computing, Third Edition in C++, pages 781-785" ---*/
//
//	unsigned long iPoin, jPoin, kPoin;
//	su2double ss, sx=0., sy=0., st2=0., t, sxoss;
//	su2double a, b;
//	su2double siga, sigb, chi2, q, sigdat;
//	su2double *x, *y;
//
//	su2double konst = 100.00;
//
//	x = new su2double[nPoin_y];
//	y = new su2double[nPoin_y];
//
//	for (iPoin = 0; iPoin < nPoin_x; iPoin++){		// loop over each streamwise location
//		for (kPoin = 0; kPoin < nPoin_z; kPoin++){	// loop over each slice
//
//			// Store x-coord at (closest mesh point to) the wall (jPoin=0 -> farthest from the wall; jPoin=nPoin_y-1 -> closest to the wall).
//			x_at_wall[iPoin][kPoin] = x_pos[iPoin][nPoin_y-1][kPoin];
//
//			/*--- Start with fitting ---*/
//			sx=0.0; sy=0.0; st2=0.0;
//			b=0.0;
//
//			/*--- Transform wm+ = Wm*exp(B*y+) into a linear form log(wm+) = log(Wm+) + B*y+;
//			 * i.e. y = a + bx, where y = log(wm+), a = log(Wm+), b = B, x = y+ ---*/
//			for (jPoin=0; jPoin<nPoin_y; jPoin++) {
//				x[jPoin] = y_plus[iPoin][jPoin][kPoin];
//				y[jPoin] = log(wm_plus[iPoin][jPoin][kPoin] + konst); // add konst to avoid having to take log of -ve numbers
//			}
//
//			/*--- Accumulate sums without weights. ---*/
//			for (jPoin=0; jPoin<nPoin_y; jPoin++) {
//				sx += x[jPoin];
//				sy += y[jPoin];
//			}
//			ss = nPoin_y;
//			sxoss = sx/ss;
//
//			for (jPoin=0; jPoin<nPoin_y; jPoin++) {
//				t = x[jPoin]-sxoss;
//				st2 += t*t;
//				b += t*y[jPoin];
//			}
//
//			/*--- Solve for a, b ---*/
//			b /= st2;
//			a = (sy-sx*b)/ss;
//
//
//			/*---Compute Wm+ and B ---*/
//			Wm_plus[iPoin][kPoin] = exp(a) - konst; // This is wm_plus evaluated at the wall
//			B[iPoin][kPoin] = b;
//
////			// Uncomment if debug needed
////			if (rank == MASTER_NODE)
////				cout << "B = " << B << ", Wm_plus[" << iPoin << "][" << kPoin << "]= " << Wm_plus[iPoin][kPoin] << endl;
//
//			/*--- Solve for sigma_a , and sigma_b. ---*/
////			siga = sqrt((1.0+sx*sx/(ss*st2))/ss);
////			sigb = sqrt(1.0/st2);
////
////			/*--- Calculate Chi squared ---*/
////			for (jPoin=0;jPoin<nPoin_y;jPoin++) {chi2 += (y[jPoin]-a-b*x[jPoin]) * (y[jPoin]-a-b*x[jPoin]);}
////
////			/*--- For unweighted data evaluate typical sig using chi2, and adjust the standard deviations ---*/
////			if (nPoin_y > 2) { sigdat=sqrt(chi2/(nPoin_y-2)); }
////			siga *= sigdat;
////			sigb *= sigdat;
//
//		}
//	}
//
//}

void CNSSolver::Fit_exponential(su2double ***wm_plus, su2double ***y_plus, su2double ***x_pos, unsigned long nPoin_x, unsigned long nPoin_y,
		                        unsigned long nPoin_z, su2double **Wm_plus, su2double **x_at_wall, su2double **B, unsigned long ***peaks) {

	/*--- Main code from "Numerical Recipes: The Art of Scientific Computing, Third Edition in C++, pages 781-785" ---*/

	unsigned long iPoin, jPoin, kPoin;
	su2double ss, sx=0., sy=0., st2=0., t, sxoss;
	su2double a, b;
	su2double siga, sigb, chi2, q, sigdat;
	su2double *xx, *yy;
	unsigned long  max_idx;
	su2double count;

	xx = new su2double[2];
	yy = new su2double[2];

	for (kPoin = 0; kPoin < nPoin_z; kPoin++){		// loop over each streamwise location
		for (iPoin = 0; iPoin < nPoin_x; iPoin++){	// loop over each slice

			// Store x-coord at (closest mesh point to) the wall (jPoin=0 -> farthest from the wall; jPoin=nPoin_y-1 -> closest to the wall).
			x_at_wall[iPoin][kPoin] = x_pos[iPoin][nPoin_y-1][kPoin];

			//jPoin index of the inflection point:
			max_idx = peaks[iPoin][1][kPoin];

			/*--- Start with fitting ---*/
			sx=0.0; sy=0.0; st2=0.0;
			b=0.0;

			// fit a line between inflection point and point immediately above it
			xx[0] = y_plus[iPoin][max_idx-3][kPoin];
			xx[1] = y_plus[iPoin][max_idx][kPoin];

			yy[0] = wm_plus[iPoin][max_idx-3][kPoin];
			yy[1] = wm_plus[iPoin][max_idx][kPoin];

			/*--- Accumulate sums without weights. ---*/
			count = 0;
			for (jPoin=0; jPoin<2; jPoin++) {
				sx += xx[jPoin];
				sy += yy[jPoin];
				count++;
			}
			ss = count;
			sxoss = sx/ss;

			for (jPoin=0; jPoin<2; jPoin++) {
				t = xx[jPoin]-sxoss;
				st2 += t*t;
				b += t*yy[jPoin];
			}

			/*--- Solve for a, b ---*/
			b /= st2;
			a = (sy-sx*b)/ss;

			/*---Compute Wm+ and B ---*/
			Wm_plus[iPoin][kPoin] = a; // This is wm_plus evaluated at the wall
			B[iPoin][kPoin] = b;

		}
	}

	/*--- Deallocate memory ---*/
	delete [] xx;
	delete [] yy;

}


void CNSSolver::Find_peak_closest_to_wall(su2double ***data, 					/* input data */
		                                   unsigned long nPoin_x,				/* row count of data */
										   unsigned long nPoin_y,
										   unsigned long nPoin_z,
										   su2double delta,						/* delta used for distinguishing peaks */
										   unsigned long ***peaks){

	/*--- Main code from: https://github.com/xuphys/peakdetect/blob/master/peakdetect.c ---*/

    unsigned long iPoin, jPoin, kPoin;
    su2double  mx, mn;
    unsigned long mx_pos, mn_pos;
    unsigned long mx_loc, mn_loc;
    unsigned short is_detecting_pk; // start detecting local peaks

    /*--- Loop over all slices ---*/
    for (kPoin=0; kPoin < nPoin_z; kPoin++){
    	for (iPoin=0; iPoin < nPoin_x; iPoin++){

			/*--- Initialize max and min of the input data. Note y coords are sorted in descending order so that jPoin = 0 corresponds to y = max(y). ---*/
			mx = data[iPoin][0][kPoin];
			mn = data[iPoin][0][kPoin];
			is_detecting_pk = 1; // start detecting local peaks
			mx_pos = 0; mn_pos = 0;
			mx_loc = 0; mn_loc = 0;

			/*--- Loop over the input data, skipping jPoin = 0 =, i.e. where y = max(y)---*/
			for(jPoin = 1; jPoin < nPoin_y; ++jPoin){

				if(data[iPoin][jPoin][kPoin] > mx){
					mx_pos = jPoin;
					mx = data[iPoin][jPoin][kPoin];
				}

				if(data[iPoin][jPoin][kPoin] < mn){
					mn_pos = jPoin;
					mn = data[iPoin][jPoin][kPoin];
				}

				if(is_detecting_pk && data[iPoin][jPoin][kPoin] < mx - delta){

					is_detecting_pk = 0;
					jPoin = mx_pos - 1;

					mn = data[iPoin][mx_pos][kPoin];
					mn_loc = mx_pos;
				}
				else if((!is_detecting_pk) &&  data[iPoin][jPoin][kPoin] > mn + delta){

					is_detecting_pk = 1;
					jPoin = mn_pos - 1;

					mx = data[iPoin][mn_pos][kPoin];
					mx_loc = mn_pos;
				}
			}

			/*--- Store mn_pos/max_pos value ---*/
			//if mx_pos/mn_pos is close to top of domain, then wm+ is a straight line -> take value close to wall
			//to avoid (max_idx-3) being -ve in Fit_exponential -> (Segmentation fault).

			if (mx_loc > mn_loc){ //i.e. if mx_loc is closer to wall than mn_loc (since y coords are sorted in descending order)
				peaks[iPoin][0][kPoin] = 1;
				if (mx_loc < 3){ peaks[iPoin][1][kPoin] = nPoin_y-1; }
				else { peaks[iPoin][1][kPoin] = mx_loc; }
			}
			else{
				peaks[iPoin][0][kPoin] = 0;
				if (mn_loc < 3){ peaks[iPoin][1][kPoin] = nPoin_y-1; }
				else {peaks[iPoin][1][kPoin] = mn_loc;}
			}
    	}
    }

}

void CNSSolver::Find_change_of_sign(su2double ***wm_plus, 					/* input data */
		                                   unsigned long nPoin_x,
										   unsigned long nPoin_y,
										   unsigned long nPoin_z,
										   unsigned long ***peaks){					/*  location of closest inflection point to the wall */

	unsigned long iPoin, jPoin, kPoin;

	for(kPoin=0; kPoin<nPoin_z; ++kPoin){
		for(iPoin=0; iPoin<nPoin_x; ++iPoin){

			if ( peaks[iPoin][0][kPoin] == 1 ){ // if wm_plus at inflection point is +ve, look for first -ve wm_plus
				for(jPoin=peaks[iPoin][1][kPoin]; jPoin>0; jPoin--){ //Loop backwards (because wall is at jPoin = nPoin_y, start from inflection point.
					if ( wm_plus[iPoin][jPoin][kPoin] < 0){
						peaks[iPoin][2][kPoin] = jPoin;
						break;
					}
					else {
						peaks[iPoin][2][kPoin] = 0; // if there is no sign change, then  assign a default index of 0.
					}
				}
			}

			else if ( peaks[iPoin][0][kPoin] == 0 ){ // if wm_plus at inflection point is -ve, look for first +ve wm_plus
				for(jPoin=peaks[iPoin][1][kPoin]; jPoin>0; jPoin--){ //Loop backwards (because wall is at jPoin = nPoin_y, start from inflection point.
					if ( wm_plus[iPoin][jPoin][kPoin] > 0){
						peaks[iPoin][2][kPoin] = jPoin;
						break;
					}
					else {
						peaks[iPoin][2][kPoin] = 0; // if there is no sign change, then  assign a default index of 0.
					}
				}
			}

		}
	}

}

unsigned long CNSSolver::LinearInt_locate(su2double *xx, unsigned long n, su2double x){

	/*--- Main code from "Numerical Recipes: The Art of Scientific Computing, Third Edition in C++, pg. 115" ---*/

	/* Given a value x, return a value j such that x is (insofar as possible) centered in the subrange
	xx[j..j+mm-1], where xx is the stored pointer. The values in xx must be monotonic, either
	increasing or decreasing. The returned value is not less than 0, nor greater than n-1.*/

	/* INPUTS:
	 * xx : Ordered 1D array
	 * n : size of xx
	 * x : location where to interpolate
	 */

	unsigned long mm = 2; //HARDCODED FOR LINIEAR INTERPOLATION

	unsigned long ju,jm,jl;
	if (n < 2 || mm < 2 || mm > n) throw("locate size error");
	bool ascnd = (xx[n-1] >= xx[0]);	// True if ascending order of table, false otherwise.

	jl = 0; 							// Initialize lower ...
	ju = n-1; 							//	...and upper limits.

	while (ju-jl > 1) {					// If we are not yet done, ...
		jm = (ju+jl) >> 1;				// ...  compute a midpoint, ...
		if (x >= xx[jm] == ascnd)		// ... and replace either the lower limit...
			jl = jm;
		else							// ... or the upper limit, as appropriate.
			ju = jm;
	}									//	Repeat until the test condition is satisfied.

	unsigned long aaa, bbb = 0;
	aaa = min(n-mm, jl-((mm-2)>>1));

	return max( bbb, aaa );				//return the location of the lower side of xx containing x, i.e. xx[lower] < x < xx[lower+1]

}

su2double CNSSolver::BilinearInterp(su2double *xx, unsigned long nx, su2double *yy, unsigned long ny, su2double **zz, su2double *xint){

	/*--- Main code from "Numerical Recipes: The Art of Scientific Computing, Third Edition in C++, pg. 132-134" ---*/

	/* bilinear interpolation on a matrix. Construct with a vector of xx values, a vector of
	yy values, and a matrix of tabulated function values zz . Then call interp for interpolated
	values.*/

	/* INPUTS:
	 * xx : Ordered 1D array
	 * yy : Ordered 1D array
	 * zz : 2D array with the function values f(xx,yy)
	 * nx, ny : size of xx, yy.
	 * xint : (x,y) location where to interpolate
	 */

	unsigned long i,j;
	su2double x,y,val,u,t;

	x = xint[0];
	y = xint[1];

	/*--- Find the grid square ---*/
	i = LinearInt_locate(xx, nx, x);
	j = LinearInt_locate(yy, ny, y);

	/*--- Interpolate ---*/
	t = (x - xx[i]) / (xx[i+1] - xx[i]);
	u = (y - yy[j]) / (yy[j+1] - yy[j]);

	val = (1.0 - t)*(1.0 - u)*zz[i][j] + t*(1.0 - u)*zz[i+1][j]	+ (1.0 - t)*u*zz[i][j+1] + t*u*zz[i+1][j+1];

	return val;

}

su2double CNSSolver::LinearInterp(su2double *xx, unsigned long nx, su2double *yy, su2double xint){

	unsigned long i;

	/*--- Find the interpolating bracket---*/
	i = LinearInt_locate(xx, nx, xint);

	/*--- Interpolate ---*/
	if (xx[i] == xx[i+1])
		return yy[i];
	else
		return yy[i] + ((xint-xx[i]) / (xx[i+1]-xx[i])) * (yy[i+1]-yy[i]);

}

su2double CNSSolver::ReynoldsScalingRicco(su2double R, su2double Re_tau){

	/*--- Code based on original script by M. van Nesselrooij ---*/

	bool mirrored = false;
	su2double Rsearch, Re_tau_in, R_new, err, A, B;
	su2double *a, *b, *R_in;
	unsigned long ii;

	a = new su2double [6];
	b = new su2double [6];
	R_in = new su2double [6];

	if (R > 0){
		R = -1.0 * R;
		mirrored = true;
	}

	Rsearch   = -1.0 * R/100;
	Re_tau_in = 200; // why??? Ricco's data are at Re_tau = 200.

	a[0] = 0.8991; b[0] = -0.0839;
	a[1] = 0.7676; b[1] = -0.09349;
	a[2] = 0.6107; b[2] = -0.1022;
	a[3] = 0.43;   b[3] = -0.1104;
	a[4] = 0.2263; b[4] = -0.1182;
	a[5] = 0;      b[5] = 0;

	for (ii=0; ii<6; ii++){
		R_in[ii] = a[ii] * pow(Re_tau_in, b[ii]);
	}

	/*--- Interpolate ---*/
	A = LinearInterp(R_in, 6, a, Rsearch);
	B = LinearInterp(R_in, 6, b, Rsearch);

	/*--- Compute friction drag reduction ---*/
	R_new = A * pow(Re_tau, B);

//	if (rank == MASTER_NODE){
//		cout << "Rnew = " << R_new << endl;
//	}

	/*--- Remove line offset if it exists ---*/
	err = Rsearch - A * pow(Re_tau_in, B);
	R_new += err;

	R_new *= -100;
	if (mirrored)
		R_new = -1.0 * R_new;

	/*--- Deallocate memory ---*/
	delete [] a;
	delete [] b;
	delete [] R_in;

	return R_new;

}

void CNSSolver::realft(su2double *data, su2double isign, unsigned long data_size) {
	/*---Calculates the Fourier transform of a set of n real-valued data points. Replaces these data
	(which are stored in array data[0..n-1]) by the positive frequency half of their complex Fourier
	transform. The real-valued first and last components of the complex transform are returned
	as elements data[0] and data[1], respectively. n must be a power of 2. This routine also
	calculates the inverse transform of a complex data array if it is the transform of real data.
	(Result in this case must be multiplied by 2/n.)---*/

	unsigned long i,i1,i2,i3,i4;
	unsigned long n = data_size;
	su2double c1=0.5,c2,h1r,h1i,h2r,h2i,wr,wi,wpr,wpi,wtemp;
	su2double theta=3.141592653589793238/su2double(n>>1); //Initialize the recurrence.

	if (isign == 1) {
		c2 = -0.5;
		four1(data, isign, data_size);		//The forward transform is here.
	} else {
		c2 = 0.5;
		theta = -theta;		//Otherwise set up for an inverse transform.
	}

	wtemp = sin(0.5*theta);
	wpr = -2.0*wtemp*wtemp;
	wpi = sin(theta);
	wr = 1.0+wpr;
	wi = wpi;

	for (i=1; i<(n>>2); i++) {			//Case i=0 done separately below
		i2 = 1+(i1=i+i);
		i4 = 1+(i3=n-i1);
		h1r = c1*(data[i1]+data[i3]);	//The two separate transforms are separated out of data
		h1i = c1*(data[i2]-data[i4]);
		h2r = -c2*(data[i2]+data[i4]);
		h2i = c2*(data[i1]-data[i3]);
		data[i1] = h1r+wr*h2r-wi*h2i;	//Here they are recombined to form the true transform of the original real data
		data[i2] = h1i+wr*h2i+wi*h2r;
		data[i3] = h1r-wr*h2r+wi*h2i;
		data[i4] = -h1i+wr*h2i+wi*h2r;
		wr = (wtemp=wr)*wpr-wi*wpi+wr;
		wi = wi*wpr+wtemp*wpi+wi;
	}

	if (isign == 1) {					//Squeeze the first and last data together to get them all within the original array.
		data[0] = (h1r = data[0])+data[1];
		data[1] = h1r - data[1];
	} else {
		data[0] = c1*((h1r = data[0])+data[1]);
		data[1] = c1*(h1r-data[1]);
		four1(data, isign, data_size);					//This is the inverse transform for the case isign=-1.
	}

}

void CNSSolver::four1(su2double *data, su2double isign, unsigned long data_size) {
	/*---Replaces data[0..2*n-1] by its discrete Fourier transform, if isign is input as 1; or replaces
	data[0..2*n-1] by n times its inverse discrete Fourier transform, if isign is input as -1. data
	is a complex array of length n stored as a real array of length 2*n. n must be an integer power	of 2.---*/

	unsigned long nn,mmax,m,j,istep,i;
	unsigned long n = data_size/2;		//Only valid for vector of real-valued data points!
	su2double wtemp,wr,wpr,wpi,wi,theta,tempr,tempi;

	if (n<2 || n&(n-1)) throw("n must be power of 2 in four1");

	nn = n << 1;
	j = 1;

	for (i=1; i<nn; i+=2) {					//This is the bit-reversal section of the routine.
		if (j > i) {
			swap(data[j-1],data[i-1]);		//Exchange the two complex numbers.
			swap(data[j],data[i]);
		}

		m = n;

		while (m >= 2 && j > m) {
			j -= m;
			m >>= 1;
		}
		j += m;
	}

	/*---Here begins the Danielson-Lanczos section of the routine.---*/
	mmax=2;

	while (nn > mmax) {								//Outer loop executed log 2 n times.
		istep = mmax << 1;
		theta = isign*(6.28318530717959/mmax);		//Initialize the trigonometric recurrence.
		wtemp = sin(0.5*theta);
		wpr = -2.0*wtemp*wtemp;
		wpi = sin(theta);
		wr = 1.0;
		wi = 0.0;

		for (m=1;m<mmax;m+=2) {						//Here are the two nested inner loops.
			for (i=m;i<=nn;i+=istep) {
				j = i+mmax;
				tempr = wr*data[j-1]-wi*data[j];	//This is the Danielson-Lanczos formula:
				tempi = wr*data[j]+wi*data[j-1];
				data[j-1] = data[i-1]-tempr;
				data[j] = data[i]-tempi;
				data[i-1] += tempr;
				data[i] += tempi;
			}
			wr = (wtemp=wr)*wpr-wi*wpi+wr;			//Trigonometric recurrence.
			wi=wi*wpr+wtemp*wpi+wi;
		}
		mmax=istep;
	}
}
