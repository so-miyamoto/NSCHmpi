#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <ctime>

// parameters
namespace {
  static const int NX = 100;
  static const int NY = 100;
  static const double DT = 0.1;
  static int MAX_STEP = 100000;
}

int main(int argc, char** argv)
{
  auto time_start = std::chrono::system_clock::now(); //

  // allocation
  auto v_alloc = std::make_unique<double[]>( 2 * (NX+2) * (NY+2) );
  auto v = reinterpret_cast<double(&)[2][NX+2][NY+2]>(*v_alloc.get());

  // initialize
  for(int i = 0; i < (NX+2)*(NY+2); i++){
    v_alloc.get()[i] = 0.0;
  }   
  for(int i = 1; i <= NX; i++){
    v[0][i][NY/2] = 100.0;
  }

  // main loop
  for(int step = 0; step < MAX_STEP; step++){
    const int tn = step%2 == 1;
    const int tp = step%2 == 0;

    // boundary condition
    for(int i = 1; i <= NX; i++){
      v[tn][i][0   ] = v[tn][i][NY];
      v[tn][i][NY+1] = v[tn][i][1 ];
    }
    for(int j = 1; j <= NY; j++){
      v[tn][0   ][j] = v[tn][NX][j];
      v[tn][NX+1][j] = v[tn][1 ][j];
    }

    // diffusion eq.
    for(int i = 1; i <= NX; i++){
      for(int j = 1; j <= NY; j++){
          v[tp][i][j] = v[tn][i][j] 
                    + DT * ( - 4.0*v[tn][i][j]
                             + v[tn][i-1][j] + v[tn][i+1][j]
                             + v[tn][i][j-1] + v[tn][i][j+1]);
      }
    }

  } // for step

  // write
  double sumv = 0.0;
  for(int i = 1; i <= NX; i++) {
    for(int j = 1; j <= NY; j++) {
      sumv += v[0][i][j];
      std::cout << v[0][i][j] << std::endl;
    }
  }
  std::cout << "sum = " << sumv << std::endl;;

  auto time_end = std::chrono::system_clock::now();  // 
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end-time_start).count();
  std::cout << "time = " << elapsed << " [ms]" << std::endl;
  return 0;
}