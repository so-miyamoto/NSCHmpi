
set terminal png
set output "phi.png"
set pm3d map
set cbrange[-1:1]
set palette defined (-1 "blue", 0 "white", 1 "red")
stat "stdout.dat"
splot "stdout.dat" u 1:2:3 w pm3d

