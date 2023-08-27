
# 流体系相分離のシミュレーション: NS-CH方程式を解く [C,C++]
2023-0827

車輪の再発明をします．

ここでは，Navie-Stokes方程式とChanHilliard方程式を解く数値解析コードを構築します．

## 結果

流速$u$にTaylor-Green Vortexの初期条件，適当に小さなオーダーパラメータ$\phi$を与える．



## NS方程式

非圧縮NS方程式は，流速uについての偏微分方程式：
$$\frac{\partial u}{\partial t}+(u\cdot\nabla)u=\frac{\eta}{\rho}\nabla^2 u - \frac{1}{\rho}\nabla p -\phi\nabla\mu$$
です．ここで，$p$は圧力，$\eta$は粘度，$\phi$はオーダーパラメータ，$\mu$は化学ポテンシャルです．
密度$\rho$は一定とします．
簡単のため，$\rho=1,\eta=1$とします.

拘束条件として$\nabla\cdot u=0$を満たす．

NS方程式部分のベンチマーク問題としては，Taylor-Green Vortexを用いることができる．2次元の場合，$0\leq x\leq 2\pi,0\leq y\leq 2\pi$で
$$u_x=\cos x\sin y e^{-2\nu t},\quad u_y=-\sin x\cos y e^{-2\nu t}$$

## NS方程式の数値解法

コロケート格子，つまりコントロールボリュームの真ん中に物理量$u,p$が配置されているとします．

空間的には中心差分で2次精度，時間的に陽的Euler法で1次精度で解きます．

時間積分にはFractional Step法を用います．

1. $n$step目の速度場$u^n$から，圧力項を除き，仮の速度場$u^*$を求めます．
$$\frac{u^*-u^n}{\Delta t}=-(u^n\cdot\nabla)u^n + \nabla^2 u^n -\phi\nabla\mu $$

2. 非圧縮条件を満たすように圧力を決定します．Jacobiの反復法を用います．

$$\nabla^2 p = - \nabla \cdot u^*$$

3. 圧力項の寄与分を足します．
$$\frac{u^{n+1}-u^*}{\Delta t}=-\nabla p$$

## CH方程式

オーダーパラメータ$\phi$の時間発展は，化学ポテンシャル$\mu$を用いて，
$$\frac{\partial \phi}{\partial t}+(u\cdot\nabla)\phi=\frac{D}{\alpha}\nabla^2\mu$$
ここで，Ginzburg-Landau型の二重井戸型ポテンシャルから，
$$\mu = \alpha(\phi^3-\phi-\gamma\nabla^2\phi)$$
である．簡単のため，パラメータ$D=\alpha=\gamma=1$とする．

これも上のNS方程式と同様に陽的Euler法で解く．

## 実装

計算ドメインは周期境界条件で$N_x\times N_y$点のRegular Gridとする．ここで$\Delta x=\Delta y=1$．

ここで，MPI実装のため，領域を$x$方向にプロセス数$N_p$等分する．
つまり，rank $i$(0-indexed)のプロセスは$[iN_x/N_p,(i+1)N_x/N_p]$の領域を計算する．
各プロセスで1列分の"のりしろ"となる領域を確保するため$(N_x/N+2)_p\times N_y$だけallocationする．
~~~C++
MPI_Sendrecv( &phi[0][NXloc  ],NY+2,MPI_DOUBLE,rank_r,0,
              &phi[0][0      ],NY+2,MPI_DOUBLE,rank_l,0,MPI_COMM_WORLD,MPI_STATUSES_IGNORE);
MPI_Sendrecv( &phi[0][1      ],NY+2,MPI_DOUBLE,rank_l,0,
              &phi[0][NXloc+1],NY+2,MPI_DOUBLE,rank_r,0,MPI_COMM_WORLD,MPI_STATUSES_IGNORE);
~~~


配列は一次元配列で確保し，その後多次元配列にキャストする．
C++っぽく，ポインタを`unique_ptr`で管理し．多次元配列にキャストして利用する．
~~~C++
  auto phi_alloc = std::make_unique<double[]>( 2 * (NXloc+2) * (NY+2)       );
  auto phi = reinterpret_cast<double(&)[2][NXloc+2][NY+2]     >(*phi_alloc.get());
  // phi is 3d array
~~~


# 実装

Github: https://github.com/so-miyamoto/NSCHmpi


# 参考

Naso, Aurore, and Lennon Ó. Náraigh. 2018. “A Flow-Pattern Map for Phase Separation Using the Navier–Stokes–Cahn–Hilliard Model.” European Journal of Mechanics - B/Fluids 72 (November): 576–585.


