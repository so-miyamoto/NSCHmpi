#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <ctime>
#include <cmath>
#include <limits>

// parameters
namespace {
  static const int X = 0;
  static const int Y = 1;
  static const int NX = 128;
  static const int NY = 128;
  static const double DT = 0.001;
  static const int MAX_STEP = 1000;
  static const int DIM = 2;
  static const double TOLERANCE = 1e-8;
  static const double SOR_COEFF = 1.0;
}

int main(int argc, char** argu)
{
  auto time_start = std::chrono::system_clock::now(); //

  // allocation
  auto u_alloc = std::make_unique<double[]>( 2 * (NX+2) * (NY+2) * DIM );
  auto p_alloc = std::make_unique<double[]>( 2 * (NX+2) * (NY+2)       );
  auto u = reinterpret_cast<double(&)[2][NX+2][NY+2][DIM]>(*u_alloc.get());
  auto p = reinterpret_cast<double(&)[2][NX+2][NY+2]     >(*p_alloc.get());

  // initialize
  for(int i = 0; i < 2*(NX+2)*(NY+2)*DIM; i++){
    u_alloc.get()[i] = 0.01;
  }   
  for(int i = 0; i < 2*(NX+2)*(NY+2); i++){
    p_alloc.get()[i] = 0.0;
  }
  for(int i = 1; i <= NX; i++) {
    for(int j = 1; j <= NY; j++) {
      double x = 0.5+(i-1);
      double y = 0.5+(j-1);
      u[0][i][j][X] =   std::cos(x/NX*2*M_PI)*std::sin(y/NY*2*M_PI);
      u[0][i][j][Y] = - std::sin(x/NX*2*M_PI)*std::cos(y/NY*2*M_PI);
    }
  }
  // boundary condition
  for(int i = 1; i <= NX; i++) {
    for(int d = 0; d < DIM; d++) {
      u[0][i][0   ][d] = u[0][i][NY][d];
      u[0][i][NY+1][d] = u[0][i][1 ][d];
    }
  }
  for(int j = 1; j <= NY; j++) {
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
            );
        up[i][j][Y] = un[i][j][Y] 
            + DT * ( 
  // diffusion
  - 4.0 * un[i][j][Y] + un[i-1][j][Y] + un[i+1][j][Y] + un[i][j-1][Y] + un[i][j+1][Y] 
  // advection (div. form)
  - 0.5 * ( un[i+1][j][X]*un[i+1][j][Y] - un[i-1][j][X]*un[i-1][j][Y]) // d/dx(ux*uy) 
  - 0.5 * ( un[i][j+1][Y]*un[i][j+1][Y] - un[i][j-1][Y]*un[i][j-1][Y]) // d/dy(uy^2)
            );
      }
    }
    // boundary condition
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
    const int max_iter = 10000;
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
      // boundary condition
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

    // update velocity
    for(int i = 1; i <= NX; i++) {
      for(int j = 1; j <= NY; j++) {
        up[i][j][X] += - DT * 0.5*(p[tp][i+1][j] - p[tp][i-1][j]);
        up[i][j][Y] += - DT * 0.5*(p[tp][i][j+1] - p[tp][i][j-1]);
      }
    }

    // boundary condition
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

    time += DT;
    // if(((step+1) % (MAX_STEP/10))==0){
    //   std::cerr << step+1 << std::endl;
    // }

    // write
    double err = 0.0;
    for(int i = 1; i <= NX; i++) {
      for(int j = 1; j <= NY; j++) {
        double x = 0.5+(i-1);
        double y = 0.5+(j-1);
        double aux =   std::cos(x/NX*2*M_PI)*std::sin(y/NY*2*M_PI)*std::exp(-2.0*time/NX);
        double auy = - std::sin(x/NX*2*M_PI)*std::cos(y/NY*2*M_PI)*std::exp(-2.0*time/NX);
        // std::cout << 0.5+(i-1) << " " << 0.5+(j-1) << " "
        //   << u[0][i][j][0] << " " << u[0][i][j][1] << " "
        //   << aux << " " << auy << std::endl;
        err +=  (up[i][j][0]-aux)*(up[i][j][0]-aux)
        + (up[i][j][1]-auy)*(up[i][j][1]-auy);
      }
    }
    std::cout << time << " " << std::sqrt(err/(NX*NY)) << std::endl;;
  } // for step


  auto time_end = std::chrono::system_clock::now();  // 
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end-time_start).count();
  std::cerr << "time = " << elapsed << " [ms]" << std::endl;
  return 0;
}