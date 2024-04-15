#include <iostream>

#include "dbscan/capi.h"
#include "dbscan/point.h"
#include "dbscan/geometryIO.h"
#include "dbscan/pbbs/parallel.h"
#include "dbscan/pbbs/parseCommandLine.h"


int main(int argc, char* argv[]) {
  parlay::internal::start_scheduler();

  commandLine P(argc,argv,"[-o <outFile>] [-eps <p_epsilon>] [-minpts <p_minpts>] <inFile>");
  char* iFile = P.getArgument(0);
  char* oFile = P.getOptionValue("-o");
  size_t rounds = P.getOptionIntValue("-r",1);
  double p_epsilon = P.getOptionDoubleValue("-eps",1);
  size_t p_minpts = P.getOptionIntValue("-minpts",1);
  double p_rho = P.getOptionDoubleValue("-rho",-1);

  // int dim = readHeader(iFile);
  int dim = P.getOptionIntValue("-dim",2);
  std::cout << "dim " << dim << std::endl;
  _seq<double> PIn = readDoubleFromFile(iFile, dim);

  std::cout << "n " << PIn.n << std::endl;

  int ct = 0;
  for (int i=0;i < PIn.n; ++ i){
    if (PIn.A[i] != 0){ std::cout << PIn.A[i] << " "; ct++;}
    if (ct > 100) break;
  }
  std::cout << std::endl;


  bool* coreFlag = new bool[PIn.n / dim];
  int* cluster = new int[PIn.n / dim];
  double* data = PIn.A;

  if (DBSCAN(dim, PIn.n / dim, data, p_epsilon, p_minpts, coreFlag, cluster))
    cout << "Error: dimension >20 is not supported." << endl;

  if (oFile != NULL) {
    writeArrayToFile("cluster-id", cluster, PIn.n / dim, oFile);
  }

  PIn.del();
  delete[] coreFlag;
  delete[] cluster;

  parlay::internal::stop_scheduler();

  return 0;
}
