#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <memory>
#include <ctime>
#include <cmath>
#include <limits>
#include <string>
#include <filesystem>

// parameters
namespace {
  const int X = 0;
  const int Y = 1;
  const int NX = 128;
  const int NY = 128;
  const double DT = 0.01;
  const int MAX_STEP = 100000;
  const int DIM = 2;
  const double TOLERANCE = 1e-5;
  const double SOR_COEFF = 1.0;
  const int SEED = 37;
}

int main(int argc, char** argu)
{
  auto time_start = std::chrono::system_clock::now(); //

  std::srand(SEED);


  int itr_write = 0;
  std::filesystem::create_directory("dat");

  // allocation
  auto u_alloc   = std::make_unique<double[]>( 2 * (NX+2) * (NY+2) * DIM );
  auto p_alloc   = std::make_unique<double[]>( 2 * (NX+2) * (NY+2)       );
  auto phi_alloc = std::make_unique<double[]>( 2 * (NX+2) * (NY+2)       );
  auto mu_alloc  = std::make_unique<double[]>( 2 * (NX+2) * (NY+2)       );
  auto u   = reinterpret_cast<double(&)[2][NX+2][NY+2][DIM]>(*  u_alloc.get());
  auto p   = reinterpret_cast<double(&)[2][NX+2][NY+2]     >(*  p_alloc.get());
  auto phi = reinterpret_cast<double(&)[2][NX+2][NY+2]     >(*phi_alloc.get());
  auto mu  = reinterpret_cast<double(&)[2][NX+2][NY+2]     >(* mu_alloc.get());


  // initialize
  for(int i = 0; i < 2*(NX+2)*(NY+2); i++){
    p_alloc.get()[i]   = 0.0;
    phi_alloc.get()[i] = 0.0;
    mu_alloc.get()[i] = 0.0;
  }
  double phiave = 0.0;
  for(int i = 1; i <= NX; i++) {
    for(int j = 1; j <= NY; j++) {
      double x = 0.5+(i-1);
      double y = 0.5+(j-1);
      u[0][i][j][X] =   std::cos(x/NX*2*M_PI)*std::sin(y/NY*2*M_PI);
      u[0][i][j][Y] = - std::sin(x/NX*2*M_PI)*std::cos(y/NY*2*M_PI);
      double tmp_phi = drand48()*0.01;
      phiave += tmp_phi;
      phi[0][i][j] = tmp_phi;
    }
  }
  phiave /= NX*NY;
  for(int i = 1; i <= NX; i++) {
    for(int j = 1; j <= NY; j++) {
      phi[0][i][j] -= phiave;
    }
  }
  // boundary condition
  for(int i = 1; i <= NX; i++) {    
    phi[0][i][0   ] = phi[0][i][NY];
    phi[0][i][NY+1] = phi[0][i][1 ];
    mu[0][i][0   ] = mu[0][i][NY];
    mu[0][i][NY+1] = mu[0][i][1 ];
    for(int d = 0; d < DIM; d++) {
      u[0][i][0   ][d] = u[0][i][NY][d];
      u[0][i][NY+1][d] = u[0][i][1 ][d];
    }
  }
  for(int j = 1; j <= NY; j++) {
    phi[0][0   ][j] = phi[0][NX][j];
    phi[0][NX+1][j] = phi[0][1 ][j];
    mu[0][0   ][j] = mu[0][NX][j];
    mu[0][NX+1][j] = mu[0][1 ][j];
    for(int d = 0; d < DIM; d++) {
      u[0][0   ][j][d] = u[0][NX][j][d];
      u[0][NX+1][j][d] = u[0][1 ][j][d];
    }
  }

  // main loop
  double time = 0.0;
  for(int step = 0; step < MAX_STEP; step++) {
    // std::cout << step << std::endl;
    const int tn = step%2 == 1;
    const int tp = step%2 == 0;
    auto& un = u[tn];
    auto& up = u[tp];

    // eval free energy (Ginzburg Langau)
    for(int i = 1; i <= NX; i++){
      for(int j = 1; j <= NY; j++){
        mu[tn][i][j] = 
  phi[tn][i][j]*phi[tn][i][j]*phi[tn][i][j] - phi[tn][i][j]
  - (- 4.0 * phi[tn][i][j] + phi[tn][i-1][j] + phi[tn][i+1][j] + phi[tn][i][j-1] + phi[tn][i][j+1]);
      }
    }

    // boundary condition
    for(int i = 1; i <= NX; i++){
      mu[tn][i][0   ] = mu[tn][i][NY];
      mu[tn][i][NY+1] = mu[tn][i][1 ];
    }
    for(int j = 1; j <= NY; j++){
      mu[tn][0   ][j] = mu[tn][NX][j];
      mu[tn][NX+1][j] = mu[tn][1 ][j];
    }

    // fractional step w/o pressure term
    for(int i = 1; i <= NX; i++){
      for(int j = 1; j <= NY; j++){
  up[i][j][X] = un[i][j][X] 
      + DT * ( 
  // diffusion
  - 4.0 * un[i][j][X] + un[i-1][j][X] + un[i+1][j][X] + un[i][j-1][X] + un[i][j+1][X] 
  // advection (div. form)
  - 0.5 * ( un[i+1][j][X]*un[i+1][j][X] - un[i-1][j][X]*un[i-1][j][X]) // d/dx(ux^2) 
  - 0.5 * ( un[i][j+1][X]*un[i][j+1][Y] - un[i][j-1][X]*un[i][j-1][Y]) // d/dy(ux*uy)
  // order parameter flux
  - phi[tn][i][j] * 0.5*( mu[tn][i+1][j] - mu[tn][i-1][j] )
            );
        up[i][j][Y] = un[i][j][Y] 
            + DT * ( 
  // diffusion
  - 4.0 * un[i][j][Y] + un[i-1][j][Y] + un[i+1][j][Y] + un[i][j-1][Y] + un[i][j+1][Y] 
  // advection (div. form)
  - 0.5 * ( un[i+1][j][X]*un[i+1][j][Y] - un[i-1][j][X]*un[i-1][j][Y]) // d/dx(ux*uy) 
  - 0.5 * ( un[i][j+1][Y]*un[i][j+1][Y] - un[i][j-1][Y]*un[i][j-1][Y]) // d/dy(uy^2)
  // order parameter flux
  - phi[tn][i][j] * 0.5*( mu[tn][i][j+1] - mu[tn][i][j-1] )
            );
      }
    }

    // boundary condition of tentative velocity
    for(int i = 1; i <= NX; i++){
      for(int d = 0; d < DIM; d++){
        up[i][0   ][d] = up[i][NY][d];
        up[i][NY+1][d] = up[i][1 ][d];
      }
    }
    for(int j = 1; j <= NY; j++){
      for(int d = 0; d < DIM; d++){
        up[0   ][j][d] = up[NX][j][d];
        up[NX+1][j][d] = up[1 ][j][d];
      }
    }

    // solve poisson equation with jacobi method
    const int max_iter = 100000;
    for (int itr = 0; itr < max_iter; itr++) {    
      const double COEFF = -0.25 * 0.5 / DT;
      for(int i = 1; i <= NX; i++){
        for(int j = 1; j <= NY; j++){
  p[tp][i][j] = 
    + COEFF * (up[i+1][j][X] - up[i-1][j][X] + up[i][j+1][Y] - up[i][j-1][Y])  // b/D
    + 0.25  * ( p[tn][i+1][j] + p[tn][i-1][j] + p[tn][i][j+1] + p[tn][i][j-1] ); // -A*x/D
        }
      }
      double residue = 0.0;
      for(int i = 1; i <= NX; i++){
        for(int j = 1; j <= NY; j++){
          double pnew = (1.0-SOR_COEFF)*p[tn][i][j] + SOR_COEFF*p[tp][i][j];
          double e = pnew - p[tn][i][j];
          residue += e*e;
          p[tn][i][j] = pnew;
        }
      }
      residue = std::sqrt(residue);
      // boundary condition of pressure
      for(int i = 1; i <= NX; i++){
        p[tn][i][0   ] = p[tn][i][NY];
        p[tn][i][NY+1] = p[tn][i][1 ];
      }
      for(int j = 1; j <= NY; j++){
        p[tn][0   ][j] = p[tn][NX][j];
        p[tn][NX+1][j] = p[tn][1 ][j];
      }
      if( residue < TOLERANCE ) break;
      if( (itr == max_iter-1) || !std::isfinite(residue) ) {
        std::cerr << "ERROR: not converged: poisson solver" << std::endl;
        std::cerr << "step: " << step << std::endl;
        std::cerr << "iterator: " << itr << std::endl;
        std::cerr << "residue: " << residue << std::endl;
        return 1;
      }
    }
    for(int i = 0; i < NX+2; i++){
      for(int j = 0; j < NY+2; j++){
        p[tp][i][j] = p[tn][i][j];
      }
    }

    // next velocity from tmp 
    for(int i = 1; i <= NX; i++) {
      for(int j = 1; j <= NY; j++) {
        up[i][j][X] += - DT * 0.5*(p[tp][i+1][j] - p[tp][i-1][j]);
        up[i][j][Y] += - DT * 0.5*(p[tp][i][j+1] - p[tp][i][j-1]);
      }
    }

    // Cahn-Hilliard Equation
    for(int i = 1; i <= NX; i++){
      for(int j = 1; j <= NY; j++){
  phi[tp][i][j] = phi[tn][i][j] + DT * (
    // advection
    - un[i][j][X] * 0.5*(phi[tn][i+1][j]-phi[tn][i-1][j]) 
    - un[i][j][Y] * 0.5*(phi[tn][i][j+1]-phi[tn][i][j-1]) 
    // diffusion
    - 4.0 * mu[tn][i][j] + mu[tn][i-1][j] + mu[tn][i+1][j] + mu[tn][i][j-1] + mu[tn][i][j+1] 
    );
      }
    }

    // boundary condition
    for(int i = 1; i <= NX; i++){
      phi[tp][i][0   ] = phi[tp][i][NY];
      phi[tp][i][NY+1] = phi[tp][i][1 ];
      for(int d = 0; d < DIM; d++){
        up[i][0   ][d] = up[i][NY][d];
        up[i][NY+1][d] = up[i][1 ][d];
      }
    }
    for(int j = 1; j <= NY; j++){
      phi[tp][0   ][j] = phi[tp][NX][j];
      phi[tp][NX+1][j] = phi[tp][1 ][j];
      for(int d = 0; d < DIM; d++){
        up[0   ][j][d] = up[NX][j][d];
        up[NX+1][j][d] = up[1 ][j][d];
      }
    }


    time += DT;

    double phi2 = 0.0;
    for(int i = 1; i <= NX; i++) {
      for(int j = 1; j <= NY; j++) {
        phi2 += phi[tp][i][j]*phi[tp][i][j];
      }
    } phi2 /= (NX*NY);
    std::fprintf(stdout,"%lf %.8lf\n",time,phi2);

    // write
    if(((step+1) % (MAX_STEP/200))==0){
      std::string fname = "dat/fluid_"+std::to_string(itr_write++)+".dat";
      FILE* fp = std::fopen(fname.c_str(),"w");
      std::fprintf(fp,"# time = %lf, phi2 = %.8lf\n",time,phi2);
      for(int i = 1; i <= NX; i++) {
        for(int j = 1; j <= NY; j++) {
          const double x = 0.5+(i-1), y = 0.5+(j-1);
          std::fprintf(fp,"%lf %lf %lf %lf %lf\n",x,y,u[tp][i][j][X],u[tp][i][j][Y],phi[tp][i][j]); 
        }
      }
      std::fclose(fp);
      std::fprintf(stderr,"step = %d, time = %lf\n",step+1,time);
    }// fi

  } // for step


  auto time_end = std::chrono::system_clock::now();  // 
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end-time_start).count();
  std::fprintf(stderr,"time = %.3f [s]",elapsed/1000.0);
  return 0;
}