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

#include "mpi.h"

// parameters
namespace {
  const int X = 0;
  const int Y = 1;
  const int NX = 256;
  const int NY = 256;
  const double DT = 0.01;
  const int MAX_STEP = 1000000;
  const int DIM = 2;
  const double TOLERANCE = 1e-6;
  const double SOR_COEFF = 1.0;
  const int SEED = 37;
}

int main(int argc, char** argv)
{
  MPI_Init(&argc,&argv);
  int num_procs;
  int my_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);  
  const int rank_r = (my_rank+1)%num_procs;
  const int rank_l = (my_rank+num_procs-1)%num_procs;

  if(NX%num_procs!=0){
    if( my_rank == 0 ) std::fprintf(stderr,"NX/num_procs != 0\n");
    return 1;
  }
  const int NXloc = NX/num_procs;

  auto time_start = std::chrono::system_clock::now(); //

  std::srand(SEED+my_rank);


  int itr_write = 0;
  std::filesystem::create_directory("dat");

  // allocation
  auto u_alloc   = std::make_unique<double[]>( 2 * (NXloc+2) * (NY+2) * DIM );
  auto p_alloc   = std::make_unique<double[]>( 2 * (NXloc+2) * (NY+2)       );
  auto phi_alloc = std::make_unique<double[]>( 2 * (NXloc+2) * (NY+2)       );
  auto mu_alloc  = std::make_unique<double[]>( 2 * (NXloc+2) * (NY+2)       );
  auto u   = reinterpret_cast<double(&)[2][NXloc+2][NY+2][DIM]>(*  u_alloc.get());
  auto p   = reinterpret_cast<double(&)[2][NXloc+2][NY+2]     >(*  p_alloc.get());
  auto phi = reinterpret_cast<double(&)[2][NXloc+2][NY+2]     >(*phi_alloc.get());
  auto mu  = reinterpret_cast<double(&)[2][NXloc+2][NY+2]     >(* mu_alloc.get());


  // initialize
  for(int i = 0; i < 2*(NXloc+2)*(NY+2); i++){
    p_alloc.get()[i]   = 0.0;
    phi_alloc.get()[i] = 0.0;
    mu_alloc.get()[i] = 0.0;
  }
  double phigen = 0.0;
  for(int i = 1; i <= NXloc; i++) {
    for(int j = 1; j <= NY; j++) {
      double x = 0.5+(i-1) + my_rank * NXloc;
      double y = 0.5+(j-1);
      u[0][i][j][X] =   std::cos(x/NX*2*M_PI)*std::sin(y/NY*2*M_PI);
      u[0][i][j][Y] = - std::sin(x/NX*2*M_PI)*std::cos(y/NY*2*M_PI);
      double tmp_phi = drand48()*0.01;
      // double tmp_phi = 0.1*(int(x+y)%3-1);
      phigen += tmp_phi;
      phi[0][i][j] = tmp_phi;
    }
  }
  double phiave = 0.0;
MPI_Allreduce(&phigen,&phiave,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
  phiave /= NX*NY;
  for(int i = 1; i <= NXloc; i++) {
    for(int j = 1; j <= NY; j++) {
      phi[0][i][j] -= phiave;
    }
  }
  // boundary condition
  for(int i = 1; i <= NXloc; i++) {    
    phi[0][i][0   ] = phi[0][i][NY];
    phi[0][i][NY+1] = phi[0][i][1 ];
    mu[0][i][0   ] = mu[0][i][NY];
    mu[0][i][NY+1] = mu[0][i][1 ];
    for(int d = 0; d < DIM; d++) {
      u[0][i][0   ][d] = u[0][i][NY][d];
      u[0][i][NY+1][d] = u[0][i][1 ][d];
    }
  }
MPI_Sendrecv( &phi[0][NXloc  ],NY+2,MPI_DOUBLE,rank_r,0,
              &phi[0][0      ],NY+2,MPI_DOUBLE,rank_l,0,MPI_COMM_WORLD,MPI_STATUSES_IGNORE);
MPI_Sendrecv( &phi[0][1      ],NY+2,MPI_DOUBLE,rank_l,0,
              &phi[0][NXloc+1],NY+2,MPI_DOUBLE,rank_r,0,MPI_COMM_WORLD,MPI_STATUSES_IGNORE);
MPI_Sendrecv( &mu [0][NXloc  ],NY+2,MPI_DOUBLE,rank_r ,0,
              &mu [0][0      ],NY+2,MPI_DOUBLE,rank_l,0,MPI_COMM_WORLD,MPI_STATUSES_IGNORE);
MPI_Sendrecv( &mu [0][1      ],NY+2,MPI_DOUBLE,rank_l ,0,
              &mu [0][NXloc+1],NY+2,MPI_DOUBLE,rank_r,0,MPI_COMM_WORLD,MPI_STATUSES_IGNORE);
MPI_Sendrecv( &u  [0][NXloc  ],2*(NY+2),MPI_DOUBLE,rank_r,0,
              &u  [0][0      ],2*(NY+2),MPI_DOUBLE,rank_l,0,MPI_COMM_WORLD,MPI_STATUSES_IGNORE);
MPI_Sendrecv( &u  [0][1      ],2*(NY+2),MPI_DOUBLE,rank_l,0,
              &u  [0][NXloc+1],2*(NY+2),MPI_DOUBLE,rank_r,0,MPI_COMM_WORLD,MPI_STATUSES_IGNORE);

  // main loop
  double time = 0.0;
  for(int step = 0; step < MAX_STEP; step++) {
    // std::cout << step << std::endl;
    const int tn = step%2 == 1;
    const int tp = step%2 == 0;
    auto& un = u[tn];
    auto& up = u[tp];

    // eval free energy (Ginzburg Langau)
    for(int i = 1; i <= NXloc; i++){
      for(int j = 1; j <= NY; j++){
        mu[tn][i][j] = 
  phi[tn][i][j]*phi[tn][i][j]*phi[tn][i][j] - phi[tn][i][j]
  - (- 4.0 * phi[tn][i][j] + phi[tn][i-1][j] + phi[tn][i+1][j] + phi[tn][i][j-1] + phi[tn][i][j+1]);
      }
    }

    // boundary condition
    for(int i = 1; i <= NXloc; i++){
      mu[tn][i][0   ] = mu[tn][i][NY];
      mu[tn][i][NY+1] = mu[tn][i][1 ];
    }
MPI_Sendrecv( &mu [tn][NXloc  ],NY+2,MPI_DOUBLE,rank_r,0,
              &mu [tn][0      ],NY+2,MPI_DOUBLE,rank_l,0,MPI_COMM_WORLD,MPI_STATUSES_IGNORE);
MPI_Sendrecv( &mu [tn][1      ],NY+2,MPI_DOUBLE,rank_l,0,
              &mu [tn][NXloc+1],NY+2,MPI_DOUBLE,rank_r,0,MPI_COMM_WORLD,MPI_STATUSES_IGNORE);

    // fractional step w/o pressure term
    for(int i = 1; i <= NXloc; i++){
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
    for(int i = 1; i <= NXloc; i++){
      for(int d = 0; d < DIM; d++){
        up[i][0   ][d] = up[i][NY][d];
        up[i][NY+1][d] = up[i][1 ][d];
      }
    }
MPI_Sendrecv( &up[NXloc  ],2*(NY+2),MPI_DOUBLE,rank_r,0,
              &up[0      ],2*(NY+2),MPI_DOUBLE,rank_l,0,MPI_COMM_WORLD,MPI_STATUSES_IGNORE);
MPI_Sendrecv( &up[1      ],2*(NY+2),MPI_DOUBLE,rank_l,0,
              &up[NXloc+1],2*(NY+2),MPI_DOUBLE,rank_r,0,MPI_COMM_WORLD,MPI_STATUSES_IGNORE);

    // solve poisson equation with jacobi method
    const int max_iter = 100000;
    for (int itr = 0; itr < max_iter; itr++) {    
      const double COEFF = -0.25 * 0.5 / DT;
      for(int i = 1; i <= NXloc; i++){
        for(int j = 1; j <= NY; j++){
  p[tp][i][j] = 
    + COEFF * (up[i+1][j][X] - up[i-1][j][X] + up[i][j+1][Y] - up[i][j-1][Y])  // b/D
    + 0.25  * ( p[tn][i+1][j] + p[tn][i-1][j] + p[tn][i][j+1] + p[tn][i][j-1] ); // -A*x/D
        }
      }
      double residue = 0.0;
      for(int i = 1; i <= NXloc; i++){
        for(int j = 1; j <= NY; j++){
          double e = p[tp][i][j] - p[tn][i][j];
          residue += e*e;
          p[tn][i][j] = (1.0-SOR_COEFF)*p[tn][i][j] + SOR_COEFF*p[tp][i][j];
        }
      }
      double residue_sum = 0.0; MPI_Allreduce(&residue_sum,&residue,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
      residue_sum = std::sqrt(residue_sum);
      // boundary condition of pressure
      for(int i = 1; i <= NXloc; i++){
        p[tn][i][0   ] = p[tn][i][NY];
        p[tn][i][NY+1] = p[tn][i][1 ];
      }
MPI_Sendrecv( &p[tn][NXloc  ],NY+2,MPI_DOUBLE,rank_r,0,
              &p[tn][0      ],NY+2,MPI_DOUBLE,rank_l,0,MPI_COMM_WORLD,MPI_STATUSES_IGNORE);
MPI_Sendrecv( &p[tn][1      ],NY+2,MPI_DOUBLE,rank_l,0,
              &p[tn][NXloc+1],NY+2,MPI_DOUBLE,rank_r,0,MPI_COMM_WORLD,MPI_STATUSES_IGNORE);

      if( residue_sum < TOLERANCE ) break;
      if( (itr == max_iter-1) || !std::isfinite(residue) ) {
        std::cerr << "ERROR: not converged: poisson solver" << std::endl;
        std::cerr << "step: " << step << std::endl;
        std::cerr << "iterator: " << itr << std::endl;
        std::cerr << "residue: " << residue << std::endl;
        return 1;
      }
    }
    for(int i = 0; i < NXloc+2; i++){
      for(int j = 0; j < NY+2; j++){
        p[tp][i][j] = p[tn][i][j];
      }
    }

    // next velocity from tmp 
    for(int i = 1; i <= NXloc; i++) {
      for(int j = 1; j <= NY; j++) {
        up[i][j][X] += - DT * 0.5*(p[tp][i+1][j] - p[tp][i-1][j]);
        up[i][j][Y] += - DT * 0.5*(p[tp][i][j+1] - p[tp][i][j-1]);
      }
    }

    // Cahn-Hilliard Equation
    for(int i = 1; i <= NXloc; i++){
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
    for(int i = 1; i <= NXloc; i++){
      phi[tp][i][0   ] = phi[tp][i][NY];
      phi[tp][i][NY+1] = phi[tp][i][1 ];
      for(int d = 0; d < DIM; d++){
        up[i][0   ][d] = up[i][NY][d];
        up[i][NY+1][d] = up[i][1 ][d];
      }
    }
MPI_Sendrecv( &phi[tp][NXloc  ],NY+2,MPI_DOUBLE,rank_r,0,
              &phi[tp][0      ],NY+2,MPI_DOUBLE,rank_l,0,MPI_COMM_WORLD,MPI_STATUSES_IGNORE);
MPI_Sendrecv( &phi[tp][1      ],NY+2,MPI_DOUBLE,rank_l,0,
              &phi[tp][NXloc+1],NY+2,MPI_DOUBLE,rank_r,0,MPI_COMM_WORLD,MPI_STATUSES_IGNORE);
MPI_Sendrecv( &up[NXloc  ],2*(NY+2),MPI_DOUBLE,rank_r,0,
              &up[0      ],2*(NY+2),MPI_DOUBLE,rank_l,0,MPI_COMM_WORLD,MPI_STATUSES_IGNORE);
MPI_Sendrecv( &up[1      ],2*(NY+2),MPI_DOUBLE,rank_l,0,
              &up[NXloc+1],2*(NY+2),MPI_DOUBLE,rank_r,0,MPI_COMM_WORLD,MPI_STATUSES_IGNORE);


    time += DT;

    double phi2 = 0.0;
    for(int i = 1; i <= NXloc; i++) {
      for(int j = 1; j <= NY; j++) {
        phi2 += phi[tp][i][j]*phi[tp][i][j];
      }
    } 
    double phi2sum = 0.0; MPI_Allreduce(&phi2,&phi2sum,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
    phi2sum /= (NX*NY);
    if( my_rank == 0 )
      std::fprintf(stdout,"%lf %.8lf\n",time,phi2sum);

    // write
    if(((step+1) % (MAX_STEP/200))==0){
      std::string fname = "dat/fluid_"+std::to_string(itr_write++)+"_"+std::to_string(my_rank)+".dat";
      FILE* fp = std::fopen(fname.c_str(),"w");
      std::fprintf(fp,"# time = %lf, phi2 = %.8lf\n",time,phi2);
      for(int i = 1; i <= NXloc; i++) {
        for(int j = 1; j <= NY; j++) {
          const double x = 0.5+(i-1) + my_rank*NXloc, y = 0.5+(j-1);
          std::fprintf(fp,"%lf %lf %lf %lf %lf\n",x,y,u[tp][i][j][X],u[tp][i][j][Y],phi[tp][i][j]); 
        }
      }
      std::fclose(fp);
      if(my_rank==0)
        std::fprintf(stderr,"step = %d, time = %lf\n",step+1,time);
    }// fi

  } // for step


  if(my_rank==0){
  auto time_end = std::chrono::system_clock::now();  // 
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end-time_start).count();
  std::fprintf(stderr,"time = %.3f [s]",elapsed/1000.0);
  }
  MPI_Finalize();
  return 0;
}